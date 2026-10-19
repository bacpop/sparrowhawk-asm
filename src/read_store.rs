//! Whole reads packed once, and borrowed in place by later counting passes.

use crate::bit_encoding::{decode_base, encode_base, valid_base};
use crate::qual_profile::MAX_GROUPS;

/// Original reads per block and counting batch. Bounded buffers avoid reallocating the whole store.
pub(crate) const BLOCK_READS: usize = 8192;
/// Class reserved for a base that no k-mer may contain.
pub const UNUSABLE: u8 = 3;
const _: () = assert!(MAX_GROUPS <= UNUSABLE as usize);

/// Packed sequences, with the first base in the lowest bits: four bits per read base by default
/// (class over code), or two bits per base for Q0 reads and contigs.
pub struct PackedSeqs {
    two_bit: bool,
    same_length: bool,
    common_len: usize,
    /// Set by `finish`: the writing phase is over, and uniform-length offsets may be gone.
    finished: bool,
    blocks: Vec<Block>,
}

/// One bounded allocation and counting batch.
struct Block {
    words: Vec<u32>,
    // Possible future compression: consecutive equal-length sequences in one u32, with low 10 bits
    // for length and upper 22 for count: (count << 10) | length. Start another entry on a length
    // change or at count 0x3F_FFFF. Lengths above 1023 would need an extension before adopting this.
    /// Exclusive sequence ends, in bases from the start of the block. Released at finalisation when
    /// all lengths are equal: each block then starts its sequences at multiples of the common length.
    ends: Vec<u32>,
    /// Sequences in the block, kept apart from `ends`, which finalisation may release.
    seq_count: usize,
}

/// A borrowed sequence within a block; its start and length are measured in bases, not words.
#[derive(Clone, Copy, Debug)]
pub struct PackedSeq<'a> {
    words: &'a [u32],
    start: usize,
    len: usize,
    two_bit: bool,
}

impl PackedSeq<'_> {
    /// Length in bases.
    pub fn len(&self) -> usize {
        self.len
    }
    /// Whether the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Whether bases use two bits, without quality classes.
    pub fn is_two_bit(&self) -> bool {
        self.two_bit
    }
    /// Two-bit base code at `i`.
    #[inline(always)]
    pub fn code(&self, i: usize) -> u8 {
        let at = self.start + i;
        ((self.words[at / 16] >> (2 * (at % 16))) & 3) as u8
    }
    /// Four-bit entry at `i`: the quality class over the base code.
    #[inline(always)]
    pub fn nibble(&self, i: usize) -> u8 {
        let at = self.start + i;
        ((self.words[at / 8] >> (4 * (at % 8))) & 0xF) as u8
    }
    /// Decode for inspection and tests; production counting and FASTA writing use bounded access.
    pub fn ascii(&self) -> Vec<u8> {
        (0..self.len)
            .map(|i| {
                if self.two_bit {
                    decode_base(self.code(i))
                } else {
                    let n = self.nibble(i);
                    if n >> 2 == UNUSABLE {
                        b'N'
                    } else {
                        decode_base(n & 3)
                    }
                }
            })
            .collect()
    }
}

impl Block {
    fn seqs(
        &self,
        two_bit: bool,
        same_length: bool,
        common_len: usize,
    ) -> impl Iterator<Item = PackedSeq<'_>> {
        (0..self.seq_count).map(move |i| {
            let (start, len) = if same_length {
                (i * common_len, common_len)
            } else {
                let start = if i == 0 { 0 } else { self.ends[i - 1] as usize };
                (start, self.ends[i] as usize - start)
            };
            PackedSeq {
                words: &self.words,
                start,
                len,
                two_bit,
            }
        })
    }
}

impl PackedSeqs {
    /// Empty collection in the requested encoding.
    pub fn new(two_bit: bool) -> Self {
        Self {
            two_bit,
            same_length: false,
            common_len: 0,
            finished: false,
            blocks: Vec::new(),
        }
    }
    /// Whether the collection stores two-bit codes.
    pub fn is_two_bit(&self) -> bool {
        self.two_bit
    }
    /// Number of sequences, including empty ones.
    pub fn len(&self) -> usize {
        self.blocks.iter().map(|b| b.seq_count).sum()
    }
    /// Whether it contains no sequences.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
    /// All sequences in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = PackedSeq<'_>> {
        self.batches().flatten()
    }
    /// Borrowed sequence iterators, one per original counting batch.
    pub fn batches(&self) -> impl Iterator<Item = impl Iterator<Item = PackedSeq<'_>>> {
        self.blocks
            .iter()
            .map(|b| b.seqs(self.two_bit, self.same_length, self.common_len))
    }
    /// Append one contig's base codes before finalisation.
    pub fn push_contig(&mut self, codes: &[u8]) {
        assert!(self.two_bit, "contigs need two-bit storage");
        self.push(codes.len(), codes.iter().copied());
    }
    /// Append one whole sequence, detecting length variation while its bases are packed.
    fn push(&mut self, len: usize, values: impl IntoIterator<Item = u8>) {
        assert!(!self.finished, "a finished store was written to");
        if self.blocks.is_empty() {
            self.common_len = len;
            self.same_length = true;
        } else if len != self.common_len {
            self.same_length = false;
        }
        if self
            .blocks
            .last()
            .is_none_or(|b| b.seq_count == BLOCK_READS)
        {
            let size = self.blocks.last_mut().map_or(0, |b| {
                b.words.shrink_to_fit();
                b.ends.shrink_to_fit();
                b.words.len()
            });
            self.blocks.push(Block {
                words: Vec::with_capacity(size),
                ends: Vec::with_capacity(BLOCK_READS),
                seq_count: 0,
            });
        }
        let block = self.blocks.last_mut().expect("an open block exists");
        let start = block.ends.last().copied().unwrap_or(0) as usize;
        let (bits, per_word) = if self.two_bit { (2, 16) } else { (4, 8) };
        let mut pos = start;
        for value in values {
            let shift = bits * (pos % per_word);
            if shift == 0 {
                block.words.push(u32::from(value));
            } else {
                *block.words.last_mut().expect("a word is open") |= u32::from(value) << shift;
            }
            pos += 1;
        }
        assert_eq!(
            pos - start,
            len,
            "packed length differs from sequence length"
        );
        block
            .ends
            .push(u32::try_from(pos).expect("a sequence block exceeds 4 Gbases"));
        block.seq_count += 1;
    }
    /// End the writing phase: release uniform-length offsets and trim the remaining buffers.
    /// Returns retained payload and offset capacity in bytes, without rescanning sequence lengths.
    pub(crate) fn finish(&mut self) -> usize {
        self.finished = true;
        if self.same_length {
            for block in &mut self.blocks {
                block.ends = Vec::new();
            }
        } else if let Some(last) = self.blocks.last_mut() {
            last.ends.shrink_to_fit();
        }
        if let Some(last) = self.blocks.last_mut() {
            last.words.shrink_to_fit();
        }
        self.blocks
            .iter()
            .map(|b| 4 * (b.words.capacity() + b.ends.capacity()))
            .sum()
    }
}

/// Whole reads, with qualities reduced to the strictest floor each base clears; Q0-only in two bits.
pub struct ReadStore {
    floors: Vec<u8>,
    class_of: [u8; 256],
    seqs: PackedSeqs,
    stats: ReadStats,
}

/// How many reads a pass saw and how long they were: what a multi-k ladder is settled from.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReadStats {
    reads: u64,
    bases: u64,
    max_read_len: usize,
}

impl ReadStats {
    fn record(&mut self, seq: &[u8]) {
        self.reads += 1;
        self.bases += seq.len() as u64;
        self.max_read_len = self.max_read_len.max(seq.len());
    }

    /// Longest read seen, in bases.
    pub fn max_read_len(&self) -> usize {
        self.max_read_len
    }

    /// Arithmetic mean read length, or `None` before any read.
    pub fn average_read_len(&self) -> Option<f64> {
        (self.reads > 0).then(|| self.bases as f64 / self.reads as f64)
    }
}

/// What a first pass leaves behind of every read: the whole read, or only its length.
pub trait RecordSink {
    /// Take note of one read.
    fn take(&mut self, seq: &[u8], qual: Option<&[u8]>);
}

impl RecordSink for ReadStore {
    fn take(&mut self, seq: &[u8], qual: Option<&[u8]>) {
        self.push(seq, qual);
    }
}

impl RecordSink for ReadStats {
    fn take(&mut self, seq: &[u8], _qual: Option<&[u8]>) {
        self.record(seq);
    }
}

impl ReadStore {
    /// Empty store for a quality ladder or fixed floor. Two-bit storage requires Q0 alone.
    pub fn new(floors: &[u8], two_bit: bool) -> Self {
        assert!(
            !floors.is_empty() && floors.len() <= MAX_GROUPS,
            "not a quality ladder: {floors:?}"
        );
        assert!(!two_bit || floors == [0], "two-bit reads require Q0 alone");
        let mut class_of = [UNUSABLE; 256];
        for (byte, class) in class_of.iter_mut().enumerate() {
            let phred = (byte as u8).saturating_sub(33);
            if let Some(floor) = floors.iter().rposition(|&f| phred >= f) {
                *class = floor as u8;
            }
        }
        Self {
            floors: floors.to_vec(),
            class_of,
            seqs: PackedSeqs::new(two_bit),
            stats: ReadStats::default(),
        }
    }
    /// Floors represented by the stored classes.
    pub fn floors(&self) -> &[u8] {
        &self.floors
    }
    /// Longest original read.
    pub fn max_read_len(&self) -> usize {
        self.stats.max_read_len()
    }
    /// Arithmetic mean original-read length.
    pub fn average_read_len(&self) -> Option<f64> {
        self.stats.average_read_len()
    }
    /// Original-read statistics.
    pub fn stats(&self) -> ReadStats {
        self.stats
    }
    /// Packed reads in input order.
    pub fn seqs(&self) -> &PackedSeqs {
        &self.seqs
    }
    /// Stored class for a floor of this ladder.
    pub fn class_of_floor(&self, floor: u8) -> u8 {
        self.floors
            .iter()
            .position(|&f| f == floor)
            .expect("floor must belong to the stored ladder") as u8
    }
    /// Pack a whole read. Two-bit mode rejects non-ACGT symbols rather than splitting reads.
    pub fn push(&mut self, seq: &[u8], qual: Option<&[u8]>) {
        let strict = (self.floors.len() - 1) as u8;
        let class_of = &self.class_of;
        if self.seqs.is_two_bit() {
            let read = self.stats.reads + 1;
            self.seqs.push(
                seq.len(),
                seq.iter().enumerate().map(|(i, &base)| {
                    if !valid_base(base) {
                        // This runs in the reader thread, whose panic reaches only stderr: log it too.
                        let message = format!(
                            "two-bit read {read}, base {}: non-ACGT symbol {:?}",
                            i + 1,
                            char::from(base)
                        );
                        log::error!("{message}");
                        panic!("{message}");
                    }
                    encode_base(base)
                }),
            );
        } else {
            self.seqs.push(
                seq.len(),
                seq.iter().enumerate().map(|(i, &base)| {
                    let class = if !valid_base(base) {
                        UNUSABLE
                    } else {
                        qual.map_or(strict, |q| {
                            q.get(i).map_or(UNUSABLE, |&b| class_of[b as usize])
                        })
                    };
                    (class << 2)
                        | if class == UNUSABLE {
                            0
                        } else {
                            encode_base(base)
                        }
                }),
            );
        }
        self.stats.record(seq);
    }
    /// Finish storage, releasing offsets if lengths stayed uniform, and report retained capacity.
    pub fn finish(&mut self) {
        let heap = self.seqs.finish();
        log::info!(
            "Read store: {} reads, {} bases, {:.1} MB, {}-bit, uniform length: {}",
            self.stats.reads,
            self.stats.bases,
            heap as f64 / 1e6,
            if self.seqs.is_two_bit() { 2 } else { 4 },
            self.seqs.same_length
        );
    }
}

/// Passes records through unchanged, leaving each in `sink` on the way.
pub struct Tee<'s, I, S> {
    inner: I,
    sink: &'s mut S,
}

impl<'s, I, S> Tee<'s, I, S> {
    /// Wrap `inner`, leaving its records in `sink`.
    pub fn new(inner: I, sink: &'s mut S) -> Self {
        Self { inner, sink }
    }
}

impl<I: Iterator<Item = (Vec<u8>, Option<Vec<u8>>)>, S: RecordSink> Iterator for Tee<'_, I, S> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let record = self.inner.next()?;
        self.sink.take(&record.0, record.1.as_deref());
        Some(record)
    }
}

//
//
//
//
// ===================================================================
// ============================================================= TESTS
// ==================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        bit_encoding::{UInt, U256, U512},
        kmer::{Bases, Kmer},
        qual_profile::{window_classes, window_groups},
    };
    use std::borrow::Cow;

    fn walk<T: for<'a> UInt<'a>, B: Bases>(
        it: Option<Kmer<'_, T, B>>,
    ) -> Vec<(u64, u64, u8, T, usize)> {
        let Some(mut it) = it else { return Vec::new() };
        let mut out = Vec::new();
        loop {
            let (hc, hnc, b, km) = it.get_hashes_and_bases_and_kmer();
            out.push((hc, hnc, b, km, it.end_index()));
            if it.get_next_hash_and_bases().is_none() {
                break;
            }
        }
        out
    }
    fn fixture() -> Vec<(Vec<u8>, Option<Vec<u8>>)> {
        let seq = b"ACGTNacgtACGGTTACGATCGATTNNACGGCATCARGTACAGGTTACAGGATC".to_vec();
        let qual = (0..seq.len())
            .map(|i| 33 + [2, 11, 25, 37, 30, 14][i % 6])
            .collect();
        vec![
            (seq.clone(), Some(qual)),
            (seq, None),
            (b"AC".to_vec(), Some(b"II".to_vec())),
        ]
    }
    fn check(floors: &[u8]) {
        let reads = fixture();
        let mut store = ReadStore::new(floors, false);
        for (seq, qual) in &reads {
            store.push(seq, qual.as_deref());
        }
        store.finish();
        assert_eq!(store.seqs.len(), reads.len());
        for ((seq, qual), packed) in reads.iter().zip(store.seqs.iter()) {
            for k in [3, 5, 7] {
                for (class, &floor) in floors.iter().enumerate() {
                    assert_eq!(
                        walk(Kmer::<u64>::new(
                            Cow::Borrowed(seq),
                            seq.len(),
                            qual.as_deref(),
                            k,
                            floor,
                            true
                        )),
                        walk(Kmer::<u64, _>::from_four_bit(packed, k, class as u8, true))
                    );
                    assert_eq!(store.class_of_floor(floor), class as u8);
                }
                if let Some(qual) = qual {
                    for top in 0..floors.len() {
                        let (mut text, mut classes) = (Vec::new(), Vec::new());
                        window_groups(seq, qual, k, &floors[..=top], &mut text);
                        window_classes(packed, k, top as u8, &mut classes);
                        assert_eq!(text, classes);
                    }
                }
            }
        }
    }
    #[test]
    fn replay_admits_the_same_kmers_at_every_floor() {
        check(&[0, 11, 25]);
    }
    #[test]
    fn a_fixed_floor_replays_what_the_iterator_keeps() {
        check(&[20]);
    }

    #[test]
    fn replay_returns_every_read_in_order_across_blocks() {
        let reads: Vec<Vec<u8>> = (0..BLOCK_READS + 3)
            .map(|i| {
                b"ACGTTGCA"
                    .iter()
                    .cycle()
                    .skip(i % 8)
                    .take(i % 17)
                    .copied()
                    .collect()
            })
            .collect();
        for two_bit in [false, true] {
            let mut store = ReadStore::new(&[0], two_bit);
            for read in &reads {
                store.push(read, None);
            }
            assert!(!store.seqs.same_length);
            store.finish();
            assert_eq!(store.seqs.blocks.len(), 2);
            assert_eq!(
                store.seqs.iter().map(|s| s.ascii()).collect::<Vec<_>>(),
                reads
            );
            assert_eq!(store.max_read_len(), 16);
        }
    }
    #[test]
    fn uniform_lengths_are_detected_during_insertion_and_offsets_are_freed() {
        for two_bit in [false, true] {
            let mut store = ReadStore::new(&[0], two_bit);
            assert!(!store.seqs.same_length);
            assert!(store.seqs.is_empty());
            for _ in 0..BLOCK_READS + 3 {
                store.push(b"aCGta", None);
                assert!(store.seqs.same_length);
            }
            let before: Vec<_> = store.seqs.iter().map(|s| s.ascii()).collect();
            assert!(store.seqs.blocks.iter().all(|b| !b.ends.is_empty()));
            store.finish();
            assert_eq!(store.seqs.common_len, 5);
            assert!(store.seqs.blocks.iter().all(|b| b.ends.capacity() == 0));
            assert_eq!(store.seqs.len(), BLOCK_READS + 3);
            assert!(!store.seqs.is_empty());
            assert_eq!(
                store.seqs.iter().map(|s| s.ascii()).collect::<Vec<_>>(),
                before
            );
        }
    }
    #[test]
    fn a_late_length_difference_is_permanent() {
        let mut store = ReadStore::new(&[0], false);
        for _ in 0..BLOCK_READS {
            store.push(b"ACG", None);
        }
        assert!(store.seqs.same_length);
        store.push(b"AC", None);
        assert!(!store.seqs.same_length);
        store.push(b"ACG", None);
        assert!(!store.seqs.same_length);
        let before: Vec<_> = store.seqs.iter().map(|s| s.ascii()).collect();
        store.finish();
        assert!(store.seqs.blocks.iter().all(|b| !b.ends.is_empty()));
        assert_eq!(
            store.seqs.iter().map(|s| s.len()).sum::<usize>(),
            BLOCK_READS * 3 + 5
        );
        assert_eq!(
            store.seqs.iter().map(|s| s.ascii()).collect::<Vec<_>>(),
            before
        );
    }
    #[test]
    #[should_panic(expected = "a finished store was written to")]
    fn a_finished_store_refuses_writes() {
        let mut seqs = PackedSeqs::new(true);
        seqs.push_contig(&[0, 1, 2]);
        seqs.finish();
        seqs.push_contig(&[0, 1, 2]);
    }
    #[test]
    fn empty_single_and_long_sequences_survive_finalisation() {
        let mut seqs = PackedSeqs::new(true);
        seqs.finish();
        assert!(seqs.is_empty());
        assert!(!seqs.same_length);
        let mut seqs = PackedSeqs::new(true);
        seqs.push_contig(&[]);
        seqs.finish();
        assert_eq!(seqs.len(), 1);
        assert!(seqs.iter().next().unwrap().is_empty());
        assert_eq!(seqs.blocks[0].ends.capacity(), 0);
        let mut seqs = PackedSeqs::new(true);
        seqs.push_contig(&vec![0; 1024]);
        seqs.push_contig(&vec![1; 4097]);
        seqs.finish();
        assert!(!seqs.same_length);
        assert_eq!(
            seqs.iter().map(|s| s.len()).collect::<Vec<_>>(),
            [1024, 4097]
        );
    }
    #[test]
    fn packed_walks_agree_across_widths_and_orientations() {
        fn check<T: for<'a> UInt<'a>>(k: usize) {
            let forward: Vec<u8> = b"acGTTGCAT".iter().cycle().take(800).copied().collect();
            let reverse: Vec<u8> = forward
                .iter()
                .rev()
                .map(|&b| decode_base(encode_base(b) ^ 2))
                .collect();
            for seq in [forward, reverse] {
                let mut qual = vec![70; seq.len()];
                qual[500] = 33;
                let mut four = ReadStore::new(&[0, 20], false);
                four.push(&seq, Some(&qual));
                four.finish();
                let mut two = ReadStore::new(&[0], true);
                two.push(&seq, Some(&qual));
                two.finish();
                for (class, floor) in [(0, 0), (1, 20)] {
                    assert_eq!(
                        walk(Kmer::<T>::new(
                            Cow::Borrowed(&seq),
                            seq.len(),
                            Some(&qual),
                            k,
                            floor,
                            true
                        )),
                        walk(Kmer::<T, _>::from_four_bit(
                            four.seqs.iter().next().unwrap(),
                            k,
                            class,
                            true
                        ))
                    );
                }
                assert_eq!(
                    walk(Kmer::<T>::new(
                        Cow::Borrowed(&seq),
                        seq.len(),
                        Some(&qual),
                        k,
                        0,
                        true
                    )),
                    walk(Kmer::<T, _>::from_two_bit(
                        two.seqs.iter().next().unwrap(),
                        k,
                        true
                    ))
                );
            }
        }
        for k in [3, 21, 31] {
            check::<u64>(k);
        }
        for k in [33, 51, 63] {
            check::<u128>(k);
        }
        for k in [65, 91, 127] {
            check::<U256>(k);
        }
        for k in [129, 191, 255] {
            check::<U512>(k);
        }
    }
    #[test]
    #[should_panic(expected = "two-bit read 1, base 3")]
    fn two_bit_reads_reject_ns() {
        ReadStore::new(&[0], true).push(b"ACNTA", None);
    }
    #[test]
    #[should_panic(expected = "non-ACGT")]
    fn two_bit_reads_reject_other_iupac() {
        ReadStore::new(&[0], true).push(b"ACrTA", None);
    }
    #[test]
    fn packed_reads_keep_exact_k_suffixes_after_unusable_bases() {
        let mut store = ReadStore::new(&[0], false);
        store.push(b"ANACG", None);
        store.finish();
        assert_eq!(
            walk(Kmer::<u64, _>::from_four_bit(
                store.seqs.iter().next().unwrap(),
                3,
                0,
                true
            ))
            .len(),
            1
        );
    }
}
