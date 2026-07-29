use nohash_hasher::NoHashHasher;
#[cfg(not(target_family = "wasm"))]
use rayon::prelude::*;
use std::{collections::HashMap, hash::BuildHasherDefault};

#[cfg(not(target_family = "wasm"))]
use std::{io::Write, path::PathBuf, time::Instant};

#[cfg(not(target_family = "wasm"))]
use super::io_utils::*;

#[cfg(not(target_family = "wasm"))]
use needletail::parser::write_fasta;

use crate::algorithms::collapser::Collapsable;
use crate::algorithms::corrector::Correctable;
#[cfg(not(target_family = "wasm"))]
use crate::algorithms::corrector::pop_bubbles_by_coverage;
use crate::algorithms::shrinker::Shrinkable;
use crate::bit_encoding::UInt;
use crate::nthash;
use std::fmt;

use crate::bit_encoding::rc_base;
use crate::logw;
#[cfg(target_family = "wasm")]
use crate::post_state;

use sparrowhawk_graph::{DbgGraph, EdgeType, HashInfoSimple, SerializedContigs};

/// Get backwards neighbours, i.e. incoming edges to either the canonical or non-canonical hashes
pub fn check_bkg(
    // backwards here mean INCOMING edges, whether from the rev.-comp. strand, or the direct one
    hc: u64,
    hnc: u64,
    k: usize,
    bases: u8,
    thedict: &HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
    maxmindict: &HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
) -> Vec<(u64, EdgeType)> {
    let mut outvec = Vec::new();
    let thecbase = bases & 3;
    let thencbase = (bases >> 2) & 3;
    for i in 0..4 {
        let tmphashc = nthash::swapbits_18_31_42_51_58_63(
            (hc ^ nthash::HASH_LOOKUP[thecbase as usize]
                ^ (nthash::MS_TAB_5LL[(i as usize * 5) + (k % 5)]
                    | nthash::MS_TAB_7L[(i as usize * 7) + (k % 7)]
                    | nthash::MS_TAB_9LC[(i as usize * 9) + (k % 9)]
                    | nthash::MS_TAB_11CR[(i as usize * 11) + (k % 11)]
                    | nthash::MS_TAB_13R[(i as usize * 13) + (k % 13)]
                    | nthash::MS_TAB_19RR[(i as usize * 19) + (k % 19)]))
                .rotate_right(1u32),
        );

        if thedict.contains_key(&tmphashc) {
            outvec.push((tmphashc, EdgeType::MinToMin));
        } else {
            let poth = maxmindict.get(&tmphashc);
            if poth.is_some_and(|x| thedict.contains_key(x)) {
                outvec.push((*poth.unwrap(), EdgeType::MaxToMin));
            }
        }

        let mut tmphashnc = hnc
            ^ (nthash::MS_TAB_5LL[(rc_base(i) as usize * 5) + (k % 5)]
                | nthash::MS_TAB_7L[(rc_base(i) as usize * 7) + (k % 7)]
                | nthash::MS_TAB_9LC[(rc_base(i) as usize * 9) + (k % 9)]
                | nthash::MS_TAB_11CR[(rc_base(i) as usize * 11) + (k % 11)]
                | nthash::MS_TAB_13R[(rc_base(i) as usize * 13) + (k % 13)]
                | nthash::MS_TAB_19RR[(rc_base(i) as usize * 19) + (k % 19)]);
        tmphashnc ^= nthash::RC_HASH_LOOKUP[thencbase as usize];
        tmphashnc = tmphashnc.rotate_right(1_u32);
        tmphashnc = nthash::swapbits_18_31_42_51_58_63(tmphashnc);

        if thedict.contains_key(&tmphashnc) {
            outvec.push((tmphashnc, EdgeType::MinToMax));
        } else {
            let poth = maxmindict.get(&tmphashnc);
            if poth.is_some_and(|x| thedict.contains_key(x)) {
                outvec.push((*poth.unwrap(), EdgeType::MaxToMax));
            }
        }
    }

    outvec
}

/// Get forward neighbours, i.e. outgoing edges from either the canonical or non-canonical hashes
pub fn check_fwd(
    hc: u64,
    hnc: u64,
    k: usize,
    bases: u8,
    thedict: &HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
    maxmindict: &HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
) -> Vec<(u64, EdgeType)> {
    let mut outvec = Vec::new();

    let thecbase = (bases >> 2) & 3;
    let thencbase = bases & 3;
    for i in 0..4 {
        let mut tmphashc = hc.rotate_left(1);
        tmphashc = nthash::swapbits_0_19_32_43_52_59(tmphashc);
        tmphashc ^= nthash::HASH_LOOKUP[i as usize];
        tmphashc ^= nthash::MS_TAB_5LL[(thecbase as usize * 5) + (k % 5)]
            | nthash::MS_TAB_7L[(thecbase as usize * 7) + (k % 7)]
            | nthash::MS_TAB_9LC[(thecbase as usize * 9) + (k % 9)]
            | nthash::MS_TAB_11CR[(thecbase as usize * 11) + (k % 11)]
            | nthash::MS_TAB_13R[(thecbase as usize * 13) + (k % 13)]
            | nthash::MS_TAB_19RR[(thecbase as usize * 19) + (k % 19)];

        if thedict.contains_key(&tmphashc) {
            outvec.push((tmphashc, EdgeType::MinToMin));
        } else {
            let poth = maxmindict.get(&tmphashc);
            if poth.is_some_and(|x| thedict.contains_key(x)) {
                outvec.push((*poth.unwrap(), EdgeType::MinToMax));
            }
        }

        let tmphashnc = nthash::swapbits_0_19_32_43_52_59(hnc.rotate_left(1u32))
            ^ nthash::RC_HASH_LOOKUP[i as usize]
            ^ (nthash::MS_TAB_5LL[(rc_base(thencbase) as usize * 5) + (k % 5)]
                | nthash::MS_TAB_7L[(rc_base(thencbase) as usize * 7) + (k % 7)]
                | nthash::MS_TAB_9LC[(rc_base(thencbase) as usize * 9) + (k % 9)]
                | nthash::MS_TAB_11CR[(rc_base(thencbase) as usize * 11) + (k % 11)]
                | nthash::MS_TAB_13R[(rc_base(thencbase) as usize * 13) + (k % 13)]
                | nthash::MS_TAB_19RR[(rc_base(thencbase) as usize * 19) + (k % 19)]);

        if thedict.contains_key(&tmphashnc) {
            outvec.push((tmphashnc, EdgeType::MaxToMin));
        } else {
            let poth = maxmindict.get(&tmphashnc);
            if poth.is_some_and(|x| thedict.contains_key(x)) {
                outvec.push((*poth.unwrap(), EdgeType::MaxToMax));
            }
        }
    }

    outvec
}

pub(crate) fn populate_neighbours(
    k: usize,
    indict: &mut HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
    maxmindict: &HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
) -> (usize, usize, usize) {
    // Serial on wasm: rayon there falls back to a single-threaded registry, so `par_iter` buys nothing,
    // and its unindexed `collect` builds an intermediate linked list of `Vec`s — allocation we cannot
    // afford under the 4 GiB linear-memory cap.
    let updates: Vec<(u64, Vec<(u64, EdgeType)>, Vec<(u64, EdgeType)>)> = {
        let dict: &HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>> = indict;
        let search = |(h, hi): (&u64, &HashInfoSimple)| {
            let pre = check_bkg(*h, hi.hnc, k, hi.b, dict, maxmindict);
            let post = check_fwd(*h, hi.hnc, k, hi.b, dict, maxmindict);
            (*h, pre, post)
        };

        // Serial on wasm: rayon there falls back to a single-threaded registry, so `par_iter` buys nothing, and its unindexed `collect` builds an intermediate linked list of `Vec`s, so just add the gate to avoid consuming more memory.
        #[cfg(not(target_family = "wasm"))]
        {
            dict.par_iter().map(search).collect()
        }
        #[cfg(target_family = "wasm")]
        {
            dict.iter().map(search).collect()
        }
    };

    let mut nkmers = 0;
    let mut nalone = 0;
    let mut directed_edge_refs = 0;

    for (h, pre, post) in updates {
        nkmers += 1;
        directed_edge_refs += pre.len() + post.len();
        if pre.is_empty() && post.is_empty() {
            nalone += 1;
        }

        let entry = indict.get_mut(&h).unwrap();
        entry.pre = pre;
        entry.post = post;
    }

    (nkmers, nalone, directed_edge_refs)
}

////////////////////////////////////////////////////////////////////////
/// Output from the assembler.
#[derive(Default)]
pub struct Contigs {
    /// Serialized contigs.
    pub serialized_contigs: SerializedContigs,

    /// Sequence of the contigs.
    pub contig_sequences: Option<Vec<Vec<u8>>>,
}

impl Contigs {
    /// Create new `Contigs`.
    pub fn new(serialized: SerializedContigs) -> Contigs {
        Contigs {
            serialized_contigs: serialized,
            contig_sequences: None,
        }
    }

    /// Temporal and historical function to simplify contigs. To be removed in the future
    pub fn shrink(&mut self) {
        for ic in 0..self.serialized_contigs.len() {
            let mut tmpv = self.serialized_contigs[ic][0].abs_ind.clone();
            let contiglen = self.serialized_contigs[ic].len();
            for j in 1..contiglen {
                tmpv.extend(self.serialized_contigs[ic][j].abs_ind.clone());
            }
            self.serialized_contigs[ic].first_mut().unwrap().abs_ind = tmpv;
            self.serialized_contigs[ic].drain(1..);
        }
    }

    /// Save contigs in a file
    #[cfg(not(target_family = "wasm"))]
    pub fn write_fasta<W: Write>(&self, f: &mut W) {
        for i in 0..self.contig_sequences.as_ref().unwrap().len() {
            let _ = write_fasta(
                i.to_string().as_bytes(),
                &self.contig_sequences.as_ref().unwrap()[i][..],
                f,
                needletail::parser::LineEnding::Unix,
            );
        }
    }
}

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

/// Spell the nucleotides of a walk of canonical k-mer hashes.
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

        // If both orientations overlapped we would have to guess; that needs the whole k-mer to be a
        // reverse-complement palindrome, which needs even k, which the CLI rejects (`valid_kmer`).
        // So the forward orientation below is unambiguous whenever it matches.
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
    use nohash_hasher::NoHashHasher;
    use std::{borrow::Cow, collections::HashMap, hash::BuildHasherDefault};

    fn empty_thedict() -> HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>> {
        HashMap::with_hasher(BuildHasherDefault::default())
    }

    fn empty_maxmindict() -> HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>> {
        HashMap::with_hasher(BuildHasherDefault::default())
    }

    #[test]
    fn check_fwd_empty_dict_returns_empty() {
        let dict = empty_thedict();
        let maxmin = empty_maxmindict();
        assert!(check_fwd(0xDEAD_BEEF, 0xCAFE_BABE, 3, 0b0101, &dict, &maxmin).is_empty());
    }

    #[test]
    fn check_bkg_empty_dict_returns_empty() {
        let dict = empty_thedict();
        let maxmin = empty_maxmindict();
        assert!(check_bkg(0xDEAD_BEEF, 0xCAFE_BABE, 3, 0b0101, &dict, &maxmin).is_empty());
    }

    #[test]
    fn check_fwd_finds_consecutive_kmer() {
        let seq = b"ACGTACGTACGT";
        let k = 5;
        let mut it =
            Kmer::<u64>::new(Cow::Borrowed(seq.as_slice()), seq.len(), None, k, 0, true).unwrap();
        let (hc1, hnc1, b1, _) = it.get_curr_kmerhash_and_bases_and_kmer();
        let (hc2, hnc2, b2, _) = it.get_next_kmer_and_give_us_things().unwrap();

        let mut dict = empty_thedict();
        dict.insert(
            hc2,
            HashInfoSimple {
                hnc: hnc2,
                b: b2,
                pre: vec![],
                post: vec![],
                counts: 1,
            },
        );
        let mut maxmin = empty_maxmindict();
        maxmin.insert(hnc2, hc2);

        let result = check_fwd(hc1, hnc1, k, b1, &dict, &maxmin);
        assert!(
            !result.is_empty(),
            "check_fwd should find the consecutive kmer"
        );
        assert!(result.iter().any(|(h, _)| *h == hc2));
    }

    #[test]
    fn check_bkg_finds_preceding_kmer() {
        let seq = b"ACGTACGTACGT";
        let k = 5;
        let mut it =
            Kmer::<u64>::new(Cow::Borrowed(seq.as_slice()), seq.len(), None, k, 0, true).unwrap();
        let (hc1, hnc1, b1, _) = it.get_curr_kmerhash_and_bases_and_kmer();
        let (hc2, hnc2, b2, _) = it.get_next_kmer_and_give_us_things().unwrap();

        let mut dict = empty_thedict();
        dict.insert(
            hc1,
            HashInfoSimple {
                hnc: hnc1,
                b: b1,
                pre: vec![],
                post: vec![],
                counts: 1,
            },
        );
        let mut maxmin = empty_maxmindict();
        maxmin.insert(hnc1, hc1);

        let result = check_bkg(hc2, hnc2, k, b2, &dict, &maxmin);
        assert!(
            !result.is_empty(),
            "check_bkg should find the preceding kmer"
        );
        assert!(result.iter().any(|(h, _)| *h == hc1));
    }

    #[test]
    fn populate_neighbours_counts_new_edges() {
        let seq = b"ACGTACGTACGT";
        let k = 5;
        let mut it =
            Kmer::<u64>::new(Cow::Borrowed(seq.as_slice()), seq.len(), None, k, 0, true).unwrap();
        let (hc1, hnc1, b1, _) = it.get_curr_kmerhash_and_bases_and_kmer();
        let (hc2, hnc2, b2, _) = it.get_next_kmer_and_give_us_things().unwrap();

        let mut dict = empty_thedict();
        dict.insert(
            hc1,
            HashInfoSimple {
                hnc: hnc1,
                b: b1,
                pre: vec![],
                post: vec![],
                counts: 1,
            },
        );
        dict.insert(
            hc2,
            HashInfoSimple {
                hnc: hnc2,
                b: b2,
                pre: vec![],
                post: vec![],
                counts: 1,
            },
        );

        let mut maxmin = empty_maxmindict();
        maxmin.insert(hnc1, hc1);
        maxmin.insert(hnc2, hc2);

        let (nkmers, nalone, directed_edge_refs) = populate_neighbours(k, &mut dict, &maxmin);

        assert_eq!(nkmers, 2);
        assert!(
            nalone < nkmers,
            "at least one kmer should have newly computed neighbours"
        );
        assert!(directed_edge_refs > 0);
        assert!(dict
            .values()
            .any(|hi| !hi.pre.is_empty() || !hi.post.is_empty()));
    }

    // ── the edge invariant ───────────────────────────────────────────────────
    //
    // "An edge exists iff the k-1 overlap holds" (see `spelling.rs`). This used to be guarded by a
    // `remove_conflictive_links` pass that was a stub returning `false` and was never enabled; these
    // tests are what replaced it. If neighbour construction ever starts inventing links, the junction
    // walk stops spelling and these fail.

    /// A deterministic ACGT sequence, with a repeat planted twice so the graph actually branches
    /// rather than being one long chain.
    fn seq_with_repeat(k: usize) -> Vec<u8> {
        const BASES: [u8; 4] = [b'A', b'C', b'G', b'T'];
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = |n: usize| -> Vec<u8> {
            (0..n)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    BASES[((state >> 33) & 3) as usize]
                })
                .collect()
        };
        let repeat = next(3 * k);
        let mut s = next(200);
        s.extend_from_slice(&repeat);
        s.extend(next(200));
        s.extend_from_slice(&repeat);
        s.extend(next(200));
        s
    }

    /// Build (packed-k-mer dict, thedict, maxmindict) for a sequence, as preprocessing would.
    #[allow(clippy::type_complexity)]
    fn build_dicts(
        seq: &[u8],
        k: usize,
    ) -> (
        HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
        HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
        HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
    ) {
        let mut packed: HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>> =
            HashMap::with_hasher(BuildHasherDefault::default());
        let mut dict = empty_thedict();
        let mut maxmin = empty_maxmindict();

        let mut it = Kmer::<u64>::new(Cow::Borrowed(seq), seq.len(), None, k, 0, true).unwrap();
        let mut cur = Some(it.get_curr_kmerhash_and_bases_and_kmer());
        while let Some((hc, hnc, b, km)) = cur {
            packed.insert(hc, km);
            maxmin.insert(hnc, hc);
            dict.entry(hc)
                .and_modify(|hi: &mut HashInfoSimple| hi.counts += 1)
                .or_insert(HashInfoSimple {
                    hnc,
                    b,
                    pre: vec![],
                    post: vec![],
                    counts: 1,
                });
            cur = it.get_next_kmer_and_give_us_things();
        }
        (packed, dict, maxmin)
    }

    /// A node's k-mers in the order a walk on `carry` traverses them. Same rule as
    /// `collapser`: `abs_ind` is stored in `innerdir`'s strand, so reverse it
    /// when we walk the other way.
    fn oriented(
        g: &DbgGraph,
        n: sparrowhawk_graph::NodeId,
        carry: sparrowhawk_graph::CarryType,
    ) -> Vec<u64> {
        let w = g.node_weight(n).unwrap();
        let mut h = w.abs_ind.clone();
        if let Some(inn) = w.innerdir {
            if carry != inn.get_from_and_to().0 {
                h.reverse();
            }
        }
        h
    }

    /// Every edge in a real graph joins two k-mers that genuinely overlap by `k-1`.
    #[test]
    fn every_edge_is_a_valid_k_minus_one_junction() {
        use crate::algorithms::shrinker::Shrinkable;
        use super::spell_path;

        let k = 15;
        let seq = seq_with_repeat(k);
        let (packed, mut dict, maxmin) = build_dicts(&seq, k);
        populate_neighbours(k, &mut dict, &maxmin);

        let mut g = DbgGraph::from_kmer_map(k, &dict);
        g.remove_self_loops();
        g.shrink();

        let mut checked = 0usize;
        for n in g.node_indices().collect::<Vec<_>>() {
            // The node's own k-mers must be a walk.
            for carry in [
                sparrowhawk_graph::CarryType::Min,
                sparrowhawk_graph::CarryType::Max,
            ] {
                let h = oriented(&g, n, carry);
                if h.len() > 1 {
                    spell_path(&h, &packed, k)
                        .unwrap_or_else(|e| panic!("node {n:?} on {carry:?} is not a walk: {e}"));
                }
            }
            // And each outgoing edge must join two overlapping k-mers.
            for (m, et) in g.outgoing_edges(n) {
                let (sc, tc) = et.get_from_and_to();
                let from = *oriented(&g, n, sc).last().unwrap();
                let to = oriented(&g, m, tc)[0];
                spell_path(&[from, to], &packed, k).unwrap_or_else(|e| {
                    panic!("edge {n:?} -{et:?}-> {m:?} is not a k-1 junction: {e}")
                });
                checked += 1;
            }
        }
        assert!(checked > 0, "the test graph has no edges to check");
    }

    /// The companion to the above: a link that is *not* a valid junction is detected. Without this,
    /// the test above could pass vacuously if `spell_path` accepted anything.
    #[test]
    fn a_fabricated_link_fails_the_junction_check() {
        use super::{spell_path, SpellError};

        let k = 15;
        let seq = seq_with_repeat(k);
        let (packed, mut dict, maxmin) = build_dicts(&seq, k);
        populate_neighbours(k, &mut dict, &maxmin);
        let g = DbgGraph::from_kmer_map(k, &dict);

        // Two k-mers from far-apart positions cannot overlap by k-1.
        let all: Vec<u64> = g
            .node_indices()
            .map(|n| g.node_weight(n).unwrap().abs_ind[0])
            .collect();
        assert!(all.len() > 100);
        assert_eq!(
            spell_path(&[all[0], all[all.len() - 1]], &packed, k),
            Err(SpellError::NotAWalk { index: 1 }),
            "an invented junction must be rejected"
        );
    }

    // ── spelling ─────────────────────────────────────────────────────────

    // ── spelling ─────────────────────────────────────────────────────────
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

/// Public API for assemblers.
pub trait Assemble {
    #[cfg(not(target_family = "wasm"))]
    /// Assembles given data and writes results into the output file.
    fn assemble<IntT: for<'a> UInt<'a>>(
        k: usize,
        indict: &mut HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
        maxminsize: &mut HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
        timevec: &mut Vec<Instant>,
        path: &mut Option<PathBuf>,
        do_bubble_collapse: bool,
        do_dead_end_removal: bool,
        pop_ratio: f32,
    ) -> Contigs;

    #[cfg(target_family = "wasm")]
    /// Assembles given data and prepares all info for being later transferred to Javascript.
    fn assemble_wasm(
        k: usize,
        indict: &mut HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
        maxminsize: &mut HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
        do_bubble_collapse: bool,
        do_dead_end_removal: bool,
        pop_ratio: f32,
    ) -> (Contigs, String, String, String);
}

///////////////////////////////////////////////////////////
/// Basic standalone assembler.
pub struct BasicAsm {}

impl Assemble for BasicAsm {
    #[cfg(not(target_family = "wasm"))]
    fn assemble<IntT: for<'a> UInt<'a>>(
        k: usize,
        indict: &mut HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
        maxmindict: &mut HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
        timevec: &mut Vec<Instant>,
        path: &mut Option<PathBuf>,
        do_bubble_collapse: bool,
        do_dead_end_removal: bool,
        pop_ratio: f32,
    ) -> Contigs {
        logw(
            "Constructing graph. Searching for neighbours...",
            Some("info"),
        );
        timevec.push(Instant::now());

        populate_neighbours(k, indict, maxmindict);

        timevec.push(Instant::now());
        logw(
            format!(
                "Neighbours searched for in {} s. Creating graph...",
                timevec
                    .last()
                    .unwrap()
                    .duration_since(*timevec.get(timevec.len().wrapping_sub(2)).unwrap())
                    .as_secs()
            )
            .as_str(),
            Some("info"),
        );

        let mut ptgraph = DbgGraph::from_kmer_map(k, indict);

        timevec.push(Instant::now());
        logw(
            format!(
                "Graph created in {} s.",
                timevec
                    .last()
                    .unwrap()
                    .duration_since(*timevec.get(timevec.len().wrapping_sub(2)).unwrap())
                    .as_secs()
            )
            .as_str(),
            Some("info"),
        );

        logw("Starting graph correction", Some("info"));

        logw("Removing self-loops", Some("info"));
        ptgraph.remove_self_loops();

        let t_phase1 = Instant::now();
        loop {
            let compacted = ptgraph.shrink();
            let pruned = if do_dead_end_removal {
                ptgraph.remove_dead_paths()
            } else {
                false
            };
            if !compacted && !pruned {
                break;
            }
        }
        log::info!(
            "  Initial simplification and dead end removal: {} ms",
            t_phase1.elapsed().as_millis()
        );

        let t_phase3 = Instant::now();
        loop {
            let mut changed = false;
            if do_bubble_collapse {
                changed |= pop_bubbles_by_coverage(&mut ptgraph, pop_ratio);
            }
            changed |= ptgraph.shrink();
            if do_dead_end_removal {
                changed |= ptgraph.remove_dead_paths();
            }
            if !changed {
                break;
            }
        }
        log::info!(
            "  Bubble correction step (coverage popping to fixed point): {} ms",
            t_phase3.elapsed().as_millis()
        );

        timevec.push(Instant::now());
        logw(
            format!(
                "Graph correction finished in {} s.",
                timevec
                    .last()
                    .unwrap()
                    .duration_since(*timevec.get(timevec.len().wrapping_sub(2)).unwrap())
                    .as_secs()
            )
            .as_str(),
            Some("info"),
        );

        if path.is_some() {
            logw("Saving graph (post-shrink, pre-collapse, w/o one-node contigs) as DOT, GFAv1.1, and GFAv2 files...", Some("info"));
            let pathmutref = path.as_mut().unwrap();
            pathmutref.set_extension("dot");
            let mut wbufdot = set_ostream(&Some(
                pathmutref.clone().into_os_string().into_string().unwrap(),
            ));
            ptgraph.write_to_dot(&mut wbufdot);

            pathmutref.set_extension("gfa");
            let mut wbufgfa = set_ostream(&Some(
                pathmutref.clone().into_os_string().into_string().unwrap(),
            ));
            ptgraph.write_to_gfa(&mut wbufgfa);

            pathmutref.set_extension("gfa2");
            let mut wbufgfa2 = set_ostream(&Some(
                pathmutref.clone().into_os_string().into_string().unwrap(),
            ));
            ptgraph.write_to_gfa2(&mut wbufgfa2);
            logw("Done.", Some("info"));
        }

        timevec.push(Instant::now());
        let serialized_contigs = ptgraph.collapse();
        timevec.push(Instant::now());
        logw(
            format!(
                "Graph collapse finished in {} s.",
                timevec
                    .last()
                    .unwrap()
                    .duration_since(*timevec.get(timevec.len().wrapping_sub(2)).unwrap())
                    .as_secs()
            )
            .as_str(),
            Some("info"),
        );
        logw(
            format!("I created {} contigs", serialized_contigs.len()).as_str(),
            Some("info"),
        );

        let mut contigs = Contigs::new(serialized_contigs);

        // TEMPORAL RESTRICTION, WIP
        contigs.shrink();

        contigs
    }

    #[cfg(target_family = "wasm")]
    fn assemble_wasm(
        k: usize,
        indict: &mut HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
        maxmindict: &mut HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
        do_bubble_collapse: bool,
        do_dead_end_removal: bool,
        pop_ratio: f32,
    ) -> (Contigs, String, String, String) {
        logw("Starting assembler!", Some("info"));

        post_state("assembly:starting");
        post_state("assembly:create_graph");
        let (nkmers, nalone, directed_edge_refs) = populate_neighbours(k, indict, maxmindict);
        let alone_pct = if nkmers == 0 {
            0.0
        } else {
            (nalone as f64) / (nkmers as f64) * 100.0
        };

        logw(
            format!("Prop. of alone kmers: {:.1} %", alone_pct).as_str(),
            Some("trace"),
        );
        logw(
            format!("Number of edges {}", (directed_edge_refs as f64) / 2_f64).as_str(),
            Some("trace"),
        );

        let mut ptgraph = DbgGraph::from_kmer_map(k, indict);

        post_state("assembly:correct_graph");
        logw("Starting graph correction", Some("info"));

        logw("Removing self-loops", Some("info"));
        ptgraph.remove_self_loops();

        loop {
            let compacted = ptgraph.shrink();
            let pruned = if do_dead_end_removal {
                ptgraph.remove_dead_paths()
            } else {
                false
            };
            if !compacted && !pruned {
                break;
            }
        }

        loop {
            let mut changed = false;
            if do_bubble_collapse {
                changed |= ptgraph.correct_bubbles(pop_ratio);
            }
            changed |= ptgraph.shrink();
            if do_dead_end_removal {
                changed |= ptgraph.remove_dead_paths();
            }
            if !changed {
                break;
            }
        }

        logw("Shrinkage and pruning finished", Some("info"));

        let outdot = ptgraph.get_dot_string();
        let outgfa = ptgraph.get_gfa_string();
        let outgfa2 = ptgraph.get_gfa2_string();

        post_state("assembly:collapse_graph");

        let serialized_contigs = ptgraph.collapse();
        logw(
            format!("I created {} contigs", serialized_contigs.len()).as_str(),
            Some("info"),
        );
        let mut contigs = Contigs::new(serialized_contigs);

        // TEMPORAL RESTRICTION, WIP
        contigs.shrink();

        (contigs, outdot, outgfa, outgfa2)
    }
}
