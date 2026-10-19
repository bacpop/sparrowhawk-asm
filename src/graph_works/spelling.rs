//! Spell graph walks into base codes and packed contig sequences.

use super::Contigs;
use crate::preprocessing::read_store::PackedSeqs;
use crate::utils::bit_encoding::UInt;
use std::fmt;

/// Why a walk could not be spelled.
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
        }
    }
}

/// Spell the two-bit base codes of a walk of canonical k-mer hashes, looking each packed k-mer up with `lookup`.
pub fn spell_path<IntT>(
    hashes: &[u64],
    lookup: impl Fn(u64) -> Option<IntT>,
    k: usize,
) -> Result<Vec<u8>, SpellError>
where
    IntT: for<'a> UInt<'a>,
{
    if hashes.is_empty() {
        return Err(SpellError::Empty);
    }

    let get = |i: usize| -> Result<IntT, SpellError> {
        lookup(hashes[i]).ok_or(SpellError::UnknownKmer {
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

    // All k bases of the first k-mer. `get_one_code` indexes from the low end, and packing is
    // MSB-first, so index k-1 is base 0 and index 0 is the last base.
    for inc in 0..k {
        out.push(prev.get_one_code(k - 1 - inc));
    }

    // Thereafter each k-mer contributes exactly one new base: its last.
    for i in 1..hashes.len() {
        let mut curr = get(i)?;
        let psuf = suffix(prev);

        // If both orientations overlapped we would have to guess; that needs the whole k-mer to be a
        // reverse-complement palindrome, which needs even k, which the CLI rejects (`valid_kmer`).
        // So the forward orientation below is unambiguous whenever it matches.
        if psuf != prefix(curr) {
            curr = curr.rev_comp(k);
            if psuf != prefix(curr) {
                return Err(SpellError::NotAWalk { index: i });
            }
        }

        out.push(curr.get_one_code(0));
        prev = curr;
    }

    debug_assert_eq!(out.len(), hashes.len() + k - 1);
    Ok(out)
}

/// Transcribes every contig into 2-bit storage, looking each packed k-mer up with `lookup`. A multi-k
/// step carries all of them to the next k, so none is length-filtered.
pub(crate) fn spell_contigs<IntT>(
    contigs: &Contigs,
    lookup: impl Fn(u64) -> Option<IntT>,
    k: usize,
) -> PackedSeqs
where
    IntT: for<'a> UInt<'a>,
{
    let mut spelled = PackedSeqs::new(true);
    for (i, contig) in contigs.serialized_contigs.iter().enumerate() {
        assert_eq!(contig.len(), 1, "contig {i} was not shrunk to one node");
        let codes = spell_path(&contig[0].abs_ind, &lookup, k)
            .unwrap_or_else(|e| panic!("contig {i} is not a valid walk: {e}"));
        spelled.push_contig(&codes);
    }
    spelled.finish();
    spelled
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessing::kmer::Kmer;
    use nohash_hasher::NoHashHasher;
    use sparrowhawk_graph::NodeStruct;
    use std::{borrow::Cow, collections::HashMap, hash::BuildHasherDefault};

    // ── spelling ─────────────────────────────────────────────────────────
    /// A deterministic ACGT sequence; an LCG keeps it reproducible without pulling in an rng crate.
    fn pseudo_seq(len: usize, seed: u64) -> Vec<u8> {
        const BASES: [u8; 4] = *b"ACGT";
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
        let (hc, _, _, km) = it.get_hashes_and_bases_and_kmer();
        dict.insert(hc, km);
        walk.push(hc);
        while let Some((hc, _, _, km)) = it.get_next_kmer() {
            dict.insert(hc, km);
            walk.push(hc);
        }
        (dict, walk)
    }

    /// The reverse complement of an ASCII sequence.
    fn reverse_complement(seq: &[u8]) -> Vec<u8> {
        seq.iter()
            .rev()
            .map(|b| match b {
                b'A' => b'T',
                b'C' => b'G',
                b'G' => b'C',
                _ => b'A',
            })
            .collect()
    }

    /// The core contract: spelling the k-mers of S gives back S, in full. A single k-mer (`len == k`)
    /// has no successor to fix its strand and is spelled in its canonical orientation, so there
    /// either strand is right.
    #[test]
    fn round_trips_the_original_sequence() {
        for k in [5, 15, 31] {
            for len in [k, k + 1, 60, 300] {
                let seq = pseudo_seq(len, 0x2545_F491_4F6C_DD1D ^ (len as u64));
                let (dict, walk) = dict_and_walk::<u64>(&seq, k);
                assert_eq!(walk.len(), len - k + 1, "k={k} len={len}: k-mer count");

                let spelled =
                    spell_path(&walk, |hash| dict.get(&hash).copied(), k).expect("valid walk");
                assert_eq!(
                    spelled.len(),
                    walk.len() + k - 1,
                    "k={k} len={len}: spelled length"
                );
                let spelled: Vec<u8> = spelled
                    .into_iter()
                    .map(crate::utils::bit_encoding::decode_base)
                    .collect();
                if len == k && spelled != seq {
                    assert_eq!(
                        spelled,
                        reverse_complement(&seq),
                        "k={k} len={len}: a lone k-mer spells one strand or the other"
                    );
                } else {
                    assert_eq!(
                        String::from_utf8(spelled).unwrap(),
                        String::from_utf8(seq).unwrap(),
                        "k={k} len={len}: spelled sequence"
                    );
                }
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
            spell_path(&w64, |hash| d64.get(&hash).copied(), k).unwrap(),
            spell_path(&w128, |hash| d128.get(&hash).copied(), k).unwrap()
        );
    }

    /// A lone k-mer has no successor to fix its strand, so it spells its canonical form: k bases.
    #[test]
    fn single_kmer_spells_exactly_k_bases() {
        let k = 31;
        let seq = pseudo_seq(k, 7);
        let (dict, walk) = dict_and_walk::<u64>(&seq, k);
        assert_eq!(walk.len(), 1);
        let spelled = spell_path(&walk, |hash| dict.get(&hash).copied(), k).unwrap();
        assert_eq!(spelled.len(), k);
        // Either strand is a legitimate answer here; the sequence itself is one of them.
        let rc = reverse_complement(&seq);
        let spelled: Vec<_> = spelled
            .into_iter()
            .map(crate::utils::bit_encoding::decode_base)
            .collect();
        assert!(
            spelled == seq || spelled == rc,
            "spelled an unrelated k-mer"
        );
    }

    #[test]
    fn empty_walk_is_an_error() {
        let dict: HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>> = HashMap::default();
        assert_eq!(
            spell_path(&[], |hash| dict.get(&hash).copied(), 31),
            Err(SpellError::Empty)
        );
    }

    #[test]
    fn a_hash_missing_from_the_dict_is_reported_with_its_position() {
        let k = 31;
        let seq = pseudo_seq(200, 3);
        let (mut dict, walk) = dict_and_walk::<u64>(&seq, k);
        dict.remove(&walk[5]);
        assert_eq!(
            spell_path(&walk, |hash| dict.get(&hash).copied(), k),
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
            spell_path(&walk, |hash| dict.get(&hash).copied(), k),
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
            spell_path(&walk, |hash| dict.get(&hash).copied(), k),
            Err(SpellError::NotAWalk { index: 10 })
        );
    }

    /// The seed of the sequences the contig tests spell.
    const CONTIG_SEED: u64 = 0x2545_F491_4F6C_DD1D;

    fn one_contig(path: Vec<u64>) -> Contigs {
        Contigs::new(vec![vec![NodeStruct {
            counts: 1,
            abs_ind: path,
            innerdir: None,
        }]])
    }

    /// A contig is the whole sequence its k-mer path spells, so it must come back as `S` itself, with
    /// both terminal k-mers: that guards the end-symmetry the old trim had to be fixed for, where an
    /// off-by-one at either end would clip exactly one of them.
    #[test]
    fn contig_preserves_the_full_spelled_path() {
        let k = 31;
        let seq = pseudo_seq(300, CONTIG_SEED);
        let n = seq.len() - k + 1; // number of k-mers
        let (dict, path) = dict_and_walk::<u64>(&seq, k);
        assert_eq!(path.len(), n);

        let sequences = spell_contigs(&one_contig(path), |hash| dict.get(&hash).copied(), k);
        let got = sequences.iter().next().unwrap().ascii();
        assert_eq!(got.len(), seq.len(), "contig length");
        assert_eq!(&got[..k], &seq[..k], "the first terminal k-mer");
        assert_eq!(&got[got.len() - k..], &seq[seq.len() - k..], "the last one");
        assert_eq!(got, seq, "contig sequence");
    }

    /// A path of fewer than k k-mers used to be dropped whatever the minimum, by a `2*k-1` guard that
    /// outlived the trim it protected. It cost ~104 contigs a run.
    #[test]
    fn a_path_shorter_than_k_kmers_is_still_written() {
        let k = 31;
        let seq = pseudo_seq(k + 4, CONTIG_SEED); // 5 k-mers, far fewer than k
        let (dict, path) = dict_and_walk::<u64>(&seq, k);
        assert!(path.len() < k, "fixture must have fewer than k k-mers");

        let sequences = spell_contigs(&one_contig(path), |hash| dict.get(&hash).copied(), k);
        assert_eq!(sequences.len(), 1, "the short path must still be written");
        assert_eq!(sequences.iter().next().unwrap().ascii(), seq);
    }
}
