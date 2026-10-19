//! Some docs here, but I think the file name is already self-explicative enough
//! Blatantly copied from ska.rust

use std::borrow::Cow;

use super::bit_encoding::*;
use super::nthash::NtHashIterator;

/// Struct to generate all k-mers from an input sequence
///
/// Holds reference to input sequence, current encoded k-mer, and other
/// information (k-mer size, masks etc.)
#[derive(Debug)]
pub struct Kmer<'a, IntT> {
    /// K-mer size
    k: usize,
    /// Reference to input sequence
    seq: Cow<'a, [u8]>,
    /// Size of seq
    seq_len: usize,
    /// Reference to sequence quality scores
    qual: Option<&'a [u8]>,
    /// Minimum quality score to allow in a k-mer
    min_qual: u8,
    /// Current index in input sequence
    index: usize,
    /// Current k-mer
    kmer: IntT,
    /// Hash generator for reads
    hash_gen: NtHashIterator,
    /// rc or Not
    rc: bool,
}

impl<'a, IntT: for<'b> UInt<'b>> Kmer<'a, IntT> {
    /// Quality score is at least minimum. Inclusive, so `--min-qual 20` keeps Q20; saturating because a byte below b'!' = 33, where the ASCII encoding starts, would otherwise underflow.
    #[inline(always)]
    fn valid_qual(idx: usize, qual: Option<&'a [u8]>, min_qual: u8) -> bool {
        match qual {
            Some(qual_seq) => qual_seq[idx].saturating_sub(33) >= min_qual,
            None => true,
        }
    }

    /// Build a new k-mer at the given index.
    ///
    #[allow(clippy::too_many_arguments)]
    fn build(
        seq: &[u8],
        seq_len: usize,
        qual: Option<&'a [u8]>,
        k: usize,
        idx: &mut usize,
        min_qual: u8,
        rc: bool,
    ) -> Option<(IntT, NtHashIterator)> {
        // A k-mer starting at `idx` spans [idx, idx + k - 1], so it exists iff idx + k <= seq_len.
        if *idx + k > seq_len {
            return None;
        }
        let mut kmer = IntT::zero_init();
        let mut i = 0;
        while i < k {
            if valid_base(seq[i + *idx]) && (Self::valid_qual(i + *idx, qual, min_qual)) {
                // Checks for N or n
                let next_base = encode_base(seq[i + *idx]);
                kmer <<= 2;
                kmer |= IntT::from_encoded_base(next_base);
                i += 1;
            } else {
                // Start again, skipping over N
                *idx += i + 1;
                if *idx + k > seq_len {
                    return None;
                }
                kmer = IntT::zero_init();
                i = 0;
            }
        }
        let hash_gen = NtHashIterator::new(&seq[*idx..(*idx + k)], k, rc);
        *idx += k - 1;
        Some((kmer, hash_gen))
    }

    /// Move forward to the next valid k-mer
    ///
    /// Usually the next base, but if an N skips over it.
    /// If end of sequence encountered then returns `false`.
    #[inline]
    fn roll_fwd(&mut self) -> bool {
        let mut success = false;
        self.index += 1;
        if self.index >= self.seq_len {
            return success;
        }
        let base = self.seq[self.index];
        if !valid_base(base) || !Self::valid_qual(self.index, self.qual, self.min_qual) {
            let new_kmer = Self::build(
                &self.seq,
                self.seq_len,
                self.qual,
                self.k,
                &mut self.index,
                self.min_qual,
                self.rc,
            );
            if let Some(kmer_tuple) = new_kmer {
                (self.kmer, self.hash_gen) = kmer_tuple;
                success = true;
            }
        } else {
            let new_base = encode_base(base);
            let old_base = (self.kmer >> ((self.k - 1) * 2)).as_u8() & 3;
            self.hash_gen.roll_fwd(old_base, new_base);

            // Update the k-mer
            let cleanbits = self.kmer.n_bits() as usize - (self.k - 1) * 2;
            self.kmer = (((self.kmer << cleanbits) >> cleanbits) << 2)
                | (IntT::from_encoded_base(new_base));
            success = true;
        }
        success
    }

    /// Create a [`Kmer`] iterator given reference to sequence input.
    ///
    /// Sequence, length and quality come from [`needletail`].
    ///
    /// Returns [`None`] if no valid k-mers found in input (e.g. too short,
    /// no sequence, too many Ns).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        seq: Cow<'a, [u8]>,
        seq_len: usize,
        qual: Option<&'a [u8]>,
        k: usize,
        min_qual: u8,
        rc: bool,
    ) -> Option<Self> {
        let mut index = 0;
        let first_kmer = Self::build(&seq, seq_len, qual, k, &mut index, min_qual, rc);
        if let Some((kmer, hash_gen)) = first_kmer {
            let kmerobj = Self {
                k,
                seq,
                seq_len,
                qual,
                min_qual,
                index,
                kmer,
                hash_gen,
                rc,
            };
            Some(kmerobj)
        } else {
            None
        }
    }

    // The current k-mer

    /// The current k-mer's canonical and non-canonical hashes, endpoint bases and packed bases, all in
    /// canonical orientation. The endpoint byte stores the first base in bits 3–2 and the last in bits
    /// 1–0; bits 7–4 are unused.
    #[inline]
    pub fn get_hashes_and_bases_and_kmer(&self) -> (u64, u64, u8, IntT) {
        let (canonical, non_canonical, bases) = self.get_hashes_and_bases();
        (canonical, non_canonical, bases, self.get_kmer())
    }

    /// Returns `(canonical_hash, non_canonical_hash, endpoint_bases)` without computing the full
    /// k-mer bits. The endpoint byte uses the same canonical layout as
    /// [`Self::get_hashes_and_bases_and_kmer`].
    #[inline]
    pub fn get_hashes_and_bases(&self) -> (u64, u64, u8) {
        let (canhash, notcanhash, isittherevcomp) = self.hash_gen.hashes_and_order();
        let thebases = if isittherevcomp {
            (rc_base((self.kmer >> ((self.k - 1) * 2)).as_u8()) & 3)
                | (rc_base((self.kmer).as_u8() & 3) << 2)
        } else {
            ((self.kmer >> ((self.k - 1) * 2)) << 2).as_u8() | ((self.kmer).as_u8() & 3)
        };
        (canhash, notcanhash, thebases)
    }

    /// Returns the current k-mer bits in canonical orientation.
    /// Call after `get_hashes_and_bases` when the k-mer is also needed.
    #[inline]
    pub fn get_kmer(&self) -> IntT {
        let (_, _, isittherevcomp) = self.hash_gen.hashes_and_order();
        if isittherevcomp {
            self.kmer.rev_comp(self.k)
        } else {
            self.kmer
        }
    }

    /// Index of the **last** base of the current k-mer, so `end_index() + 1 - k` is its first base.
    #[inline]
    pub fn end_index(&self) -> usize {
        self.index
    }

    // The next k-mer

    /// Move to the next k-mer and return [`Self::get_hashes_and_bases_and_kmer`], or `None` at the
    /// end of the read.
    #[inline]
    pub fn get_next_kmer(&mut self) -> Option<(u64, u64, u8, IntT)> {
        if self.roll_fwd() {
            Some(self.get_hashes_and_bases_and_kmer())
        } else {
            None
        }
    }

    /// Move to the next k-mer and return [`Self::get_hashes_and_bases`], or `None` at the end of the
    /// read: lighter than [`Self::get_next_kmer`], as it skips the packed bases.
    #[inline]
    pub fn get_next_hash_and_bases(&mut self) -> Option<(u64, u64, u8)> {
        if self.roll_fwd() {
            Some(self.get_hashes_and_bases())
        } else {
            None
        }
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
    use std::borrow::Cow;

    fn count_kmers(seq: &[u8], k: usize) -> usize {
        match Kmer::<u64>::new(Cow::Borrowed(seq), seq.len(), None, k, 0, true) {
            None => 0,
            Some(mut it) => {
                let mut c = 1;
                while it.get_next_kmer().is_some() {
                    c += 1;
                }
                c
            }
        }
    }

    #[test]
    fn kmer_count_basic() {
        // 8-mer sequence with k=3 → 8-3+1 = 6 k-mers
        assert_eq!(count_kmers(b"ACGTACGT", 3), 6);
    }

    #[test]
    fn kmer_short_seq_returns_none() {
        assert!(Kmer::<u64>::new(Cow::Borrowed(b"AC"), 2, None, 3, 0, true).is_none());
    }

    #[test]
    fn kmer_exact_length_yields_one() {
        // seq_len == k: the whole sequence is the single k-mer (seq_len - k + 1 = 1)
        assert_eq!(count_kmers(b"ACG", 3), 1);
    }

    #[test]
    fn kmer_exact_length_suffix_after_n() {
        // "ANACG" k=3: the N restarts the window at idx=2, leaving a suffix of exactly k.
        // That suffix is a valid k-mer ("ACG") and must not be dropped.
        assert_eq!(count_kmers(b"ANACG", 3), 1);
    }

    #[test]
    fn kmer_minimum_valid_length() {
        // seq_len = k+1: exactly two k-mers (initial + one roll)
        assert_eq!(count_kmers(b"ACGT", 3), 2);
    }

    #[test]
    fn kmer_n_invalidates_window() {
        // "ACNGT" k=3: N at pos 2, restart skips to idx=3, leaving only "GT" (< k) → None
        assert_eq!(count_kmers(b"ACNGT", 3), 0);
    }

    #[test]
    fn kmer_n_at_start_skipped() {
        // "NNNACGT" k=3: three leading Ns skipped, yields "ACG" + "CGT" = 2
        assert_eq!(count_kmers(b"NNNACGT", 3), 2);
    }

    #[test]
    fn kmer_n_in_middle_splits() {
        // "ACGTNACGT" k=3: "ACG"+"CGT" before N, "ACG"+"CGT" after N = 4
        assert_eq!(count_kmers(b"ACGTNACGT", 3), 4);
        // Any other IUPAC letter breaks the window as an N does.
        assert_eq!(count_kmers(b"ACGTRACGT", 3), 4);
    }

    #[test]
    fn kmer_quality_filter_reduces_count() {
        let seq = b"ACGTACGT";
        // All-passing quality: b'I'=73, (73-33)=40 >= 30 ✓
        let good_qual = vec![b'I'; seq.len()];
        // Bad quality at position 1 (C): b'!'=33, (33-33)=0 < 30 → fails
        let mut bad_qual = good_qual.clone();
        bad_qual[1] = b'!';

        let good_count = count_kmers(seq, 3); // no quality filter → 6

        let bad_count = match Kmer::<u64>::new(
            Cow::Borrowed(seq.as_slice()),
            seq.len(),
            Some(&bad_qual),
            3,
            30,
            true,
        ) {
            None => 0,
            Some(mut it) => {
                let mut c = 1;
                while it.get_next_kmer().is_some() {
                    c += 1;
                }
                c
            }
        };

        assert!(
            bad_count < good_count,
            "bad_count={bad_count} good_count={good_count}"
        );
    }

    /// The floor is inclusive, so a read sitting exactly on it keeps every k-mer. This is the
    /// regression test for the `>` that used to reject Q20 under `--min-qual 20`.
    #[test]
    fn a_base_at_exactly_the_floor_is_kept() {
        let seq = b"ACGTACGTACGT";
        let k = 5;
        let qual = vec![20 + 33; seq.len()];
        let mut it = Kmer::<u64>::new(
            Cow::Borrowed(seq.as_slice()),
            seq.len(),
            Some(&qual),
            k,
            20,
            true,
        )
        .expect("a read at exactly the floor still yields k-mers");
        let mut n = 1;
        while it.get_next_hash_and_bases().is_some() {
            n += 1;
        }
        assert_eq!(n, seq.len() - k + 1);
    }

    /// `end_index` must name the last base of the current window, since Step 2 uses it to index a
    /// per-position array of floors.
    #[test]
    fn end_index_lines_up_with_the_window() {
        let seq = b"ACGTACGTACGT";
        let k = 5;
        let mut it =
            Kmer::<u64>::new(Cow::Borrowed(seq.as_slice()), seq.len(), None, k, 0, true).unwrap();
        assert_eq!(it.end_index() + 1 - k, 0);
        it.get_next_hash_and_bases().unwrap();
        assert_eq!(it.end_index() + 1 - k, 1);
    }

    /// Test to see if the u8 store the first and last bases as they should
    #[test]
    fn endpoint_byte_stores_canonical_first_and_last_bases() {
        let alphabet = *b"ACTG";
        let k = 3;
        let mut orientations_seen = [false; 2];

        for encoded in 0..64 {
            let seq = [
                alphabet[(encoded >> 4) & 3],
                alphabet[(encoded >> 2) & 3],
                alphabet[encoded & 3],
            ];
            let it = Kmer::<u64>::new(Cow::Borrowed(&seq), seq.len(), None, k, 0, true).unwrap();
            let (_, _, is_reverse_complement) = it.hash_gen.hashes_and_order();
            let (_, _, endpoint_bases, canonical) = it.get_hashes_and_bases_and_kmer();

            orientations_seen[is_reverse_complement as usize] = true;
            assert_eq!(
                (endpoint_bases >> 2) & 3,
                ((canonical >> (2 * (k - 1))) as u8) & 3
            );
            assert_eq!(endpoint_bases & 3, canonical as u8 & 3);
            assert_eq!(endpoint_bases & 0xf0, 0);
        }

        assert!(orientations_seen.into_iter().all(|seen| seen));
    }
}
