//! Every read packed once in memory, so a multi-k run parses and decompresses its input a single time.

use crate::bit_encoding::{decode_base, encode_base};
use crate::qual_profile::MAX_GROUPS;

/// Reads per block. Splitting into blocks means a big buffer is never reallocated, which would briefly
/// double it while counting is at its peak.
const BLOCK_READS: usize = 8192;
/// Class of a base no k-mer may contain (N, or anything else not A, C, G or T): the one 2-bit value
/// the floor indices, at most `MAX_GROUPS` of them, leave free.
const INVALID: u8 = 3;
const _: () = assert!(MAX_GROUPS <= INVALID as usize);

/// One block of reads, eight bases per `u32` word (first base in the low nibble), as WGSL buffers hold
/// them. Each nibble is a 2-bit base code with the 2-bit quality class above it.
struct Block {
    words: Vec<u32>,
    /// Exclusive end of each read, in bases from the start of the block.
    ends: Vec<u32>,
}

/// The reads, each base's quality reduced to the strictest ladder floor it clears. That is all counting
/// ever reads from a quality, so replaying the store counts exactly what streaming the files did.
pub struct ReadStore {
    /// Ascending ladder. A valid base's class is the index of the strictest floor it clears, the same
    /// index `window_groups` gives a window.
    floors: Vec<u8>,
    /// Class of a valid base, by its raw quality byte.
    class_of: [u8; 256],
    /// Full blocks, shrunk to fit, then the one being filled.
    blocks: Vec<Block>,
    reads: u64,
    bases: u64,
    max_read_len: usize,
}

impl ReadStore {
    /// An empty store for a ladder as `qual_profile::floors_from` builds it (`0` first, at most 3 rungs).
    pub fn new(floors: &[u8]) -> Self {
        assert!(
            !floors.is_empty() && floors.len() <= MAX_GROUPS && floors[0] == 0,
            "not a quality ladder: {floors:?}"
        );
        let mut class_of = [0u8; 256];
        for (byte, class) in class_of.iter_mut().enumerate() {
            let phred = (byte as u8).saturating_sub(33);
            // `floors[0]` is 0, so every valid base clears at least that one.
            *class = floors.iter().rposition(|&f| phred >= f).unwrap_or(0) as u8;
        }
        Self {
            floors: floors.to_vec(),
            class_of,
            blocks: Vec::new(),
            reads: 0,
            bases: 0,
            max_read_len: 0,
        }
    }

    /// The ladder the classes refer to.
    pub fn floors(&self) -> &[u8] {
        &self.floors
    }

    /// Longest read seen, in bases.
    pub fn max_read_len(&self) -> usize {
        self.max_read_len
    }

    /// Arithmetic mean read length over all stored records, or `None` for an empty store.
    pub fn average_read_len(&self) -> Option<f64> {
        (self.reads > 0).then(|| self.bases as f64 / self.reads as f64)
    }

    /// Floor of 90% of the arithmetic mean read length, measured over every stored record.
    ///
    /// An empty store returns zero. Integer arithmetic keeps the ladder boundary deterministic.
    pub fn ninety_percent_average_read_len(&self) -> usize {
        if self.reads == 0 {
            return 0;
        }
        let target = (u128::from(self.bases) * 9) / (u128::from(self.reads) * 10);
        usize::try_from(target).unwrap_or(usize::MAX)
    }

    /// Pack one read into the open block. A FASTA read has no qualities and clears every floor, as it
    /// does when streamed.
    pub fn push(&mut self, seq: &[u8], qual: Option<&[u8]>) {
        if self
            .blocks
            .last()
            .is_none_or(|b| b.ends.len() == BLOCK_READS)
        {
            // Close the full block at its exact size; the next one starts at about the same size.
            let size = self.blocks.last_mut().map_or(0, |full| {
                full.words.shrink_to_fit();
                full.words.len()
            });
            self.blocks.push(Block {
                words: Vec::with_capacity(size),
                ends: Vec::with_capacity(BLOCK_READS),
            });
        }
        let strict = (self.floors.len() - 1) as u8;
        let class_of = &self.class_of;
        let block = self
            .blocks
            .last_mut()
            .expect("an open block was just ensured");
        let mut pos = block.ends.last().map_or(0, |&end| end as usize);
        for (i, &base) in seq.iter().enumerate() {
            let class = if !matches!(base, b'A' | b'C' | b'G' | b'T' | b'a' | b'c' | b'g' | b't') {
                INVALID
            } else {
                qual.map_or(strict, |q| {
                    q.get(i).map_or(INVALID, |&b| class_of[b as usize])
                })
            };
            let code = (class << 2)
                | if class == INVALID {
                    0
                } else {
                    encode_base(base)
                };
            // Every eighth base opens a word; the others fill the open word's next nibble up.
            let shift = 4 * (pos % 8);
            if shift == 0 {
                block.words.push(u32::from(code));
            } else {
                *block.words.last_mut().expect("a part-filled word is open") |=
                    u32::from(code) << shift;
            }
            pos += 1;
        }
        block
            .ends
            .push(u32::try_from(pos).expect("a read block exceeds 4 Gbases"));
        self.reads += 1;
        self.bases += seq.len() as u64;
        self.max_read_len = self.max_read_len.max(seq.len());
    }

    /// Shrink the last, partial block to fit and report the store's size. Call once the first k's
    /// pass is over.
    pub fn finish(&mut self) {
        if let Some(last) = self.blocks.last_mut() {
            last.words.shrink_to_fit();
            last.ends.shrink_to_fit();
        }
        let heap: usize = self
            .blocks
            .iter()
            .map(|b| 4 * (b.words.len() + b.ends.len()))
            .sum();
        log::info!(
            "Read store: {} reads, {} bases, {:.1} MB",
            self.reads,
            self.bases,
            heap as f64 / 1e6
        );
    }

    /// Every read in input order, decoded to bases and one quality byte per ladder class.
    pub fn records(&self) -> Records<'_> {
        Records {
            store: self,
            block: 0,
            read: 0,
        }
    }

    /// A class decodes to its own floor's quality, so it clears exactly the floors it cleared before.
    fn decode(&self, block: &Block, start: usize, end: usize) -> (Vec<u8>, Option<Vec<u8>>) {
        let mut seq = Vec::with_capacity(end - start);
        let mut qual = Vec::with_capacity(end - start);
        for i in start..end {
            let code = ((block.words[i / 8] >> (4 * (i % 8))) & 0xF) as u8;
            match code >> 2 {
                INVALID => {
                    seq.push(b'N');
                    qual.push(33);
                }
                class => {
                    seq.push(decode_base(code & 3));
                    qual.push(33 + self.floors[class as usize]);
                }
            }
        }
        (seq, Some(qual))
    }
}

/// The store's reads in input order, as `(sequence, qualities)`, like a FASTQ parser yields them.
pub struct Records<'a> {
    store: &'a ReadStore,
    block: usize,
    read: usize,
}

impl Iterator for Records<'_> {
    type Item = (Vec<u8>, Option<Vec<u8>>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let block = self.store.blocks.get(self.block)?;
            if let Some(&end) = block.ends.get(self.read) {
                let start = if self.read == 0 {
                    0
                } else {
                    block.ends[self.read - 1]
                };
                self.read += 1;
                return Some(self.store.decode(block, start as usize, end as usize));
            }
            self.block += 1;
            self.read = 0;
        }
    }
}

/// Passes records through unchanged, packing each into the store on the way.
pub struct Tee<'s, I> {
    inner: I,
    store: &'s mut ReadStore,
}

impl<'s, I> Tee<'s, I> {
    /// Wrap `inner`, packing its records into `store`.
    pub fn new(inner: I, store: &'s mut ReadStore) -> Self {
        Self { inner, store }
    }
}

impl<I: Iterator<Item = (Vec<u8>, Option<Vec<u8>>)>> Iterator for Tee<'_, I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let record = self.inner.next()?;
        self.store.push(&record.0, record.1.as_deref());
        Some(record)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kmer::Kmer;
    use crate::qual_profile::window_groups;
    use std::borrow::Cow;

    /// Canonical hashes the iterator emits at `floor`, as the counter sees them.
    fn kmers_at(seq: &[u8], qual: Option<&[u8]>, k: usize, floor: u8) -> Vec<u64> {
        let Some(mut it) = Kmer::<u64>::new(Cow::Borrowed(seq), seq.len(), qual, k, floor, true)
        else {
            return Vec::new();
        };
        let mut out = vec![it.get_curr_hash_and_bases().0];
        while let Some((hc, _, _)) = it.get_next_hash_and_bases() {
            out.push(hc);
        }
        out
    }

    /// Ns, lowercase, every quality bin, a FASTA record, and a read shorter than every k.
    fn fixture() -> Vec<(Vec<u8>, Option<Vec<u8>>)> {
        let seq = b"ACGTNacgtACGGTTACGATCGATTNNACGGCATCAGGTACAGGTTACAGGATC".to_vec();
        let qual: Vec<u8> = (0..seq.len())
            .map(|i| 33 + [2u8, 11, 25, 37, 30, 14][i % 6])
            .collect();
        vec![
            (seq.clone(), Some(qual)),
            (seq, None),
            (b"AC".to_vec(), Some(b"II".to_vec())),
        ]
    }

    /// The invariant the store rests on: every floor admits exactly the same windows after a replay.
    #[test]
    fn replay_admits_the_same_kmers_at_every_floor() {
        let floors = [0u8, 11, 25];
        let reads = fixture();
        let mut store = ReadStore::new(&floors);
        for (seq, qual) in &reads {
            store.push(seq, qual.as_deref());
        }
        store.finish();
        let replayed: Vec<_> = store.records().collect();
        assert_eq!(replayed.len(), reads.len());
        for ((seq, qual), (rseq, rqual)) in reads.iter().zip(&replayed) {
            for k in [3usize, 5, 7] {
                if let Some(qual) = qual {
                    let (mut a, mut b) = (Vec::new(), Vec::new());
                    window_groups(seq, qual, k, &floors, &mut a);
                    window_groups(rseq, rqual.as_deref().unwrap(), k, &floors, &mut b);
                    assert_eq!(a, b, "k={k}");
                }
                for &f in &floors {
                    assert_eq!(
                        kmers_at(seq, qual.as_deref(), k, f),
                        kmers_at(rseq, rqual.as_deref(), k, f),
                        "k={k} floor={f}"
                    );
                }
            }
        }
    }

    /// Lengths that are not multiples of 8 straddle words, and a full block closes mid-stream; neither
    /// may change the reads.
    #[test]
    fn replay_returns_every_read_in_order_across_blocks() {
        let reads: Vec<Vec<u8>> = (0..BLOCK_READS + 3)
            .map(|i| {
                b"ACGTTGCA"
                    .iter()
                    .cycle()
                    .skip(i % 8)
                    .take(1 + i % 13)
                    .copied()
                    .collect()
            })
            .collect();
        let mut store = ReadStore::new(&[0, 20]);
        for read in &reads {
            store.push(read, None);
        }
        store.finish();
        let back: Vec<Vec<u8>> = store.records().map(|(seq, _)| seq).collect();
        assert_eq!(back, reads);
        assert_eq!(store.max_read_len(), 13);
    }

    #[test]
    fn ninety_percent_average_length_uses_all_records_and_floors_the_result() {
        let mut store = ReadStore::new(&[0, 20]);
        store.push(b"ACGTACGTAC", None); // 10 bases
        store.push(b"ACGTACGTACGT", None); // 12 bases; mean 11, target 9
        assert_eq!(store.average_read_len(), Some(11.0));
        assert_eq!(store.ninety_percent_average_read_len(), 9);
    }

    #[test]
    fn ninety_percent_average_length_is_zero_for_an_empty_store() {
        let store = ReadStore::new(&[0, 20]);
        assert_eq!(store.average_read_len(), None);
        assert_eq!(store.ninety_percent_average_read_len(), 0);
    }
}
