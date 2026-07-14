//! Turning a walk of canonical k-mer hashes back into nucleotides.
//!
//! The graph stores each k-mer once, under its *canonical* hash `hc = min(fwd, rc)`, so the packed
//! k-mer in `thedict` may be either the k-mer as it reads along the walk or its reverse complement.
//! Spelling therefore means recovering, for each k-mer in turn, which of the two strands is the one
//! that continues the previous k-mer — which we can do because consecutive k-mers in a walk overlap by
//! exactly `k-1` bases.
//!
//! Two consumers: the contig writer (`save_functions`) and the multi-k evidence oracle, which needs the
//! *full* sequence of a candidate path so it can re-hash it at a larger k.

use nohash_hasher::NoHashHasher;
use std::{collections::HashMap, fmt, hash::BuildHasherDefault};

use crate::bit_encoding::UInt;

/// Why a walk could not be spelled.
///
/// Every variant is an invariant violation that should be impossible for a walk taken from the graph:
/// an edge exists *iff* the `k-1` overlap holds, so a genuine walk always overlaps. Reporting rather
/// than papering over them is the point — the previous code counted the second case and emitted a base
/// from the failing orientation anyway.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpellError {
    /// The walk was empty.
    Empty,
    /// A hash in the walk is absent from the dictionary.
    UnknownKmer {
        /// Position in the walk.
        index: usize,
        /// The offending canonical hash.
        hash: u64,
    },
    /// Consecutive k-mers overlap in neither orientation: the "walk" is not a walk.
    NotAWalk {
        /// Position of the k-mer that failed to follow its predecessor.
        index: usize,
    },
    /// Both orientations of a k-mer overlap its predecessor, so the strand is ambiguous. This needs the
    /// `k-1` junction to be a reverse-complement palindrome; `k-1` is even, so it is not impossible a
    /// priori, but it has never been observed.
    AmbiguousOverlap {
        /// Position of the ambiguous k-mer.
        index: usize,
    },
}

impl fmt::Display for SpellError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "the walk is empty"),
            Self::UnknownKmer { index, hash } => {
                write!(f, "k-mer {index} (hash {hash}) is not in the dictionary")
            }
            Self::NotAWalk { index } => write!(
                f,
                "k-mer {index} does not overlap its predecessor in either orientation"
            ),
            Self::AmbiguousOverlap { index } => write!(
                f,
                "k-mer {index} overlaps its predecessor in both orientations"
            ),
        }
    }
}

/// Spell the nucleotides of a walk of canonical k-mer hashes.
///
/// Returns the **full** sequence, of length `hashes.len() + k - 1`. Note the contig writer deliberately
/// trims `k-1` from each end of this; the oracle does not, because it wants every base the walk covers.
///
/// The caller only has to get the *order* of the hashes right — reversing a unitig's `abs_ind` if it is
/// traversed backwards. It does **not** have to track each k-mer's strand: that is recovered here, from
/// the overlap.
///
/// The spelled sequence may come out as the reverse complement of the genomic orientation, since what is
/// pinned is the walk's *direction*, not its strand. That is harmless for canonical-hash lookups, which
/// are reverse-complement invariant.
pub fn spell_path<IntT>(
    hashes: &[u64],
    dict: &HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    k: usize,
) -> Result<Vec<u8>, SpellError>
where
    IntT: for<'a> UInt<'a>,
{
    if hashes.is_empty() {
        return Err(SpellError::Empty);
    }

    let get = |i: usize| -> Result<IntT, SpellError> {
        dict.get(&hashes[i])
            .copied()
            .ok_or(SpellError::UnknownKmer {
                index: i,
                hash: hashes[i],
            })
    };

    let mut prev = get(0)?;

    // Number of high bits to shift out to leave the last `k-1` bases. This is a property of the integer
    // type, not of k alone, so it is correct for u64/u128/U256/U512 alike.
    let clear_high = prev.n_bits() as usize - 2 * (k - 1);
    let suffix = |kmer: IntT| -> IntT { (kmer << clear_high) >> clear_high };
    // The first `k-1` bases of a k-mer: drop the last base.
    let prefix = |kmer: IntT| -> IntT { kmer >> 2 };

    // The first k-mer's strand is not determined by itself — only by whether it can be continued. Pick
    // the orientation that the second k-mer follows.
    if hashes.len() > 1 {
        let next = get(1)?;
        let (nf, nr) = (prefix(next), prefix(next.rev_comp(k)));
        if suffix(prev) != nf && suffix(prev) != nr {
            prev = prev.rev_comp(k);
            if suffix(prev) != nf && suffix(prev) != nr {
                return Err(SpellError::NotAWalk { index: 1 });
            }
        }
    }

    let mut out: Vec<u8> = Vec::with_capacity(hashes.len() + k - 1);

    // All k bases of the first k-mer. `get_one_nucleotide` indexes from the low end, and packing is
    // MSB-first, so index k-1 is base 0 and index 0 is the last base.
    for inc in 0..k {
        out.push(prev.get_one_nucleotide(k - 1 - inc));
    }

    // Thereafter each k-mer contributes exactly one new base: its last.
    for i in 1..hashes.len() {
        let mut curr = get(i)?;
        let psuf = suffix(prev);

        if psuf == prefix(curr) && psuf == prefix(curr.rev_comp(k)) {
            return Err(SpellError::AmbiguousOverlap { index: i });
        }
        if psuf != prefix(curr) {
            curr = curr.rev_comp(k);
            if psuf != prefix(curr) {
                return Err(SpellError::NotAWalk { index: i });
            }
        }

        out.push(curr.get_one_nucleotide(0));
        prev = curr;
    }

    debug_assert_eq!(out.len(), hashes.len() + k - 1);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kmer::Kmer;
    use std::borrow::Cow;

    /// A deterministic ACGT sequence; an LCG keeps it reproducible without pulling in an rng crate.
    fn pseudo_seq(len: usize, seed: u64) -> Vec<u8> {
        const BASES: [u8; 4] = [b'A', b'C', b'G', b'T'];
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                BASES[((state >> 33) & 3) as usize]
            })
            .collect()
    }

    /// Build the (dict, walk) pair that preprocessing would hand us for a sequence: `dict` maps
    /// canonical hash -> canonical packed k-mer, and the walk is the ordered canonical hashes.
    #[allow(clippy::type_complexity)]
    fn dict_and_walk<IntT>(
        seq: &[u8],
        k: usize,
    ) -> (
        HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
        Vec<u64>,
    )
    where
        IntT: for<'a> UInt<'a>,
    {
        let mut dict = HashMap::with_hasher(BuildHasherDefault::default());
        let mut walk = Vec::new();
        let mut it = Kmer::<IntT>::new(Cow::Borrowed(seq), seq.len(), None, k, 0, true).unwrap();
        let (hc, _, _, km) = it.get_curr_kmerhash_and_bases_and_kmer();
        dict.insert(hc, km);
        walk.push(hc);
        while let Some((hc, _, _, km)) = it.get_next_kmer_and_give_us_things() {
            dict.insert(hc, km);
            walk.push(hc);
        }
        (dict, walk)
    }

    /// The core contract: spelling the k-mers of S gives back S, in full.
    #[test]
    fn round_trips_the_original_sequence() {
        for k in [5, 15, 31] {
            for len in [k, k + 1, 60, 300] {
                let seq = pseudo_seq(len, 0x2545_F491_4F6C_DD1D ^ (len as u64));
                let (dict, walk) = dict_and_walk::<u64>(&seq, k);
                assert_eq!(walk.len(), len - k + 1, "k={k} len={len}: k-mer count");

                let spelled = spell_path(&walk, &dict, k).expect("valid walk");
                assert_eq!(
                    spelled.len(),
                    walk.len() + k - 1,
                    "k={k} len={len}: spelled length"
                );
                assert_eq!(
                    String::from_utf8(spelled).unwrap(),
                    String::from_utf8(seq).unwrap(),
                    "k={k} len={len}: spelled sequence"
                );
            }
        }
    }

    /// The packed-k-mer width must not change what is spelled — multi-k sizes IntT from the LARGER k,
    /// so a k=31 graph is spelled through u128 rather than u64.
    #[test]
    fn width_of_intt_does_not_change_the_spelling() {
        let k = 31;
        let seq = pseudo_seq(500, 99);
        let (d64, w64) = dict_and_walk::<u64>(&seq, k);
        let (d128, w128) = dict_and_walk::<u128>(&seq, k);
        assert_eq!(w64, w128, "hashes are independent of the packing width");
        assert_eq!(
            spell_path(&w64, &d64, k).unwrap(),
            spell_path(&w128, &d128, k).unwrap()
        );
    }

    /// A lone k-mer has no successor to fix its strand, so it spells its canonical form: k bases.
    #[test]
    fn single_kmer_spells_exactly_k_bases() {
        let k = 31;
        let seq = pseudo_seq(k, 7);
        let (dict, walk) = dict_and_walk::<u64>(&seq, k);
        assert_eq!(walk.len(), 1);
        let spelled = spell_path(&walk, &dict, k).unwrap();
        assert_eq!(spelled.len(), k);
        // Either strand is a legitimate answer here; the sequence itself is one of them.
        let rc: Vec<u8> = seq
            .iter()
            .rev()
            .map(|b| match b {
                b'A' => b'T',
                b'C' => b'G',
                b'G' => b'C',
                _ => b'A',
            })
            .collect();
        assert!(spelled == seq || spelled == rc, "spelled an unrelated k-mer");
    }

    #[test]
    fn empty_walk_is_an_error() {
        let dict: HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>> = HashMap::default();
        assert_eq!(spell_path(&[], &dict, 31), Err(SpellError::Empty));
    }

    #[test]
    fn a_hash_missing_from_the_dict_is_reported_with_its_position() {
        let k = 31;
        let seq = pseudo_seq(200, 3);
        let (mut dict, walk) = dict_and_walk::<u64>(&seq, k);
        dict.remove(&walk[5]);
        assert_eq!(
            spell_path(&walk, &dict, k),
            Err(SpellError::UnknownKmer {
                index: 5,
                hash: walk[5]
            })
        );
    }

    /// Two k-mers that do not overlap are not a walk. The old code counted this and emitted a base
    /// anyway; it is now a hard error.
    #[test]
    fn a_non_overlapping_pair_is_not_a_walk() {
        let k = 31;
        let a = pseudo_seq(k, 11);
        let b = pseudo_seq(k, 22);
        let (mut dict, wa) = dict_and_walk::<u64>(&a, k);
        let (db, wb) = dict_and_walk::<u64>(&b, k);
        dict.extend(db);

        let walk = vec![wa[0], wb[0]];
        assert_eq!(
            spell_path(&walk, &dict, k),
            Err(SpellError::NotAWalk { index: 1 })
        );
    }

    /// Splicing an unrelated k-mer into the middle of a good walk is caught at exactly that index.
    #[test]
    fn a_break_mid_walk_is_caught_at_its_index() {
        let k = 31;
        let seq = pseudo_seq(200, 5);
        let other = pseudo_seq(k, 4242);
        let (mut dict, mut walk) = dict_and_walk::<u64>(&seq, k);
        let (dother, wother) = dict_and_walk::<u64>(&other, k);
        dict.extend(dother);

        walk[10] = wother[0];
        assert_eq!(
            spell_path(&walk, &dict, k),
            Err(SpellError::NotAWalk { index: 10 })
        );
    }
}
