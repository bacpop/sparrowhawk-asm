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
use crate::algorithms::corrector::CorrectionOpts;
#[cfg(not(target_family = "wasm"))]
use crate::algorithms::path_correction::{remove_bulges, remove_erroneous_connections};
use crate::algorithms::shrinker::Shrinkable;
use crate::bit_encoding::UInt;
#[cfg(not(target_family = "wasm"))]
use crate::indexed_kmers::{IndexedKmers, OrientedKmerIndex};
use crate::nthash;
use std::fmt;

use crate::bit_encoding::rc_base;
use crate::logw;
#[cfg(target_family = "wasm")]
use crate::post_state;

#[cfg(not(target_family = "wasm"))]
use sparrowhawk_graph::IndexedEdge;
use sparrowhawk_graph::{DbgGraph, EdgeType, HashInfoSimple, SerializedContigs};

/// Retrieves a packed canonical k-mer by its hash.
pub trait KmerLookup<IntT> {
    /// Return the packed k-mer associated with `hash`.
    fn get_kmer(&self, hash: u64) -> Option<IntT>;
}

impl<IntT: Copy> KmerLookup<IntT> for HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>> {
    fn get_kmer(&self, hash: u64) -> Option<IntT> {
        self.get(&hash).copied()
    }
}

#[cfg(not(target_family = "wasm"))]
impl<IntT: Copy> KmerLookup<IntT> for IndexedKmers<IntT> {
    fn get_kmer(&self, hash: u64) -> Option<IntT> {
        self.get_packed(hash).copied()
    }
}

/// ntHash's split-rotate tables for `base` at `k`, ORed: the term that drops or adds the far base.
#[inline]
fn ms_tab(base: u8, k: usize) -> u64 {
    let b = base as usize;
    nthash::MS_TAB_5LL[b * 5 + k % 5]
        | nthash::MS_TAB_7L[b * 7 + k % 7]
        | nthash::MS_TAB_9LC[b * 9 + k % 9]
        | nthash::MS_TAB_11CR[b * 11 + k % 11]
        | nthash::MS_TAB_13R[b * 13 + k % 13]
        | nthash::MS_TAB_19RR[b * 19 + k % 19]
}

/// A k-mer's eight candidate predecessor hashes, two per base (from the canonical strand, then from
/// the reverse complement), each with its edge type for a canonical and for a reverse-complement hit.
#[inline]
fn bkg_candidates(hc: u64, hnc: u64, k: usize, bases: u8) -> [(u64, EdgeType, EdgeType); 8] {
    let (canonical_base, reverse_base) = (bases & 3, (bases >> 2) & 3);
    let mut out = [(0, EdgeType::MinToMin, EdgeType::MinToMin); 8];
    for i in 0..4u8 {
        let canonical = nthash::swapbits_18_31_42_51_58_63(
            (hc ^ nthash::HASH_LOOKUP[canonical_base as usize] ^ ms_tab(i, k)).rotate_right(1),
        );
        let reverse = nthash::swapbits_18_31_42_51_58_63(
            (hnc ^ ms_tab(rc_base(i), k) ^ nthash::RC_HASH_LOOKUP[reverse_base as usize])
                .rotate_right(1),
        );
        out[2 * i as usize] = (canonical, EdgeType::MinToMin, EdgeType::MaxToMin);
        out[2 * i as usize + 1] = (reverse, EdgeType::MinToMax, EdgeType::MaxToMax);
    }
    out
}

/// The same for a k-mer's eight candidate successors.
#[inline]
fn fwd_candidates(hc: u64, hnc: u64, k: usize, bases: u8) -> [(u64, EdgeType, EdgeType); 8] {
    let (canonical_base, reverse_base) = ((bases >> 2) & 3, bases & 3);
    let mut out = [(0, EdgeType::MinToMin, EdgeType::MinToMin); 8];
    for i in 0..4u8 {
        let canonical = nthash::swapbits_0_19_32_43_52_59(hc.rotate_left(1))
            ^ nthash::HASH_LOOKUP[i as usize]
            ^ ms_tab(canonical_base, k);
        let reverse = nthash::swapbits_0_19_32_43_52_59(hnc.rotate_left(1))
            ^ nthash::RC_HASH_LOOKUP[i as usize]
            ^ ms_tab(rc_base(reverse_base), k);
        out[2 * i as usize] = (canonical, EdgeType::MinToMin, EdgeType::MinToMax);
        out[2 * i as usize + 1] = (reverse, EdgeType::MaxToMin, EdgeType::MaxToMax);
    }
    out
}

/// The candidates present in the legacy dictionaries: a canonical hit directly, a reverse-complement
/// one through `maxmindict`.
fn dict_hits(
    candidates: [(u64, EdgeType, EdgeType); 8],
    thedict: &HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
    maxmindict: &HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
) -> Vec<(u64, EdgeType)> {
    candidates
        .into_iter()
        .filter_map(|(hash, canonical, reverse)| {
            if thedict.contains_key(&hash) {
                Some((hash, canonical))
            } else {
                maxmindict
                    .get(&hash)
                    .filter(|h| thedict.contains_key(*h))
                    .map(|&h| (h, reverse))
            }
        })
        .collect()
}

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
    dict_hits(bkg_candidates(hc, hnc, k, bases), thedict, maxmindict)
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
    dict_hits(fwd_candidates(hc, hnc, k, bases), thedict, maxmindict)
}

#[cfg_attr(all(not(target_family = "wasm"), not(test)), allow(dead_code))]
#[allow(clippy::type_complexity)]
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

#[cfg(not(target_family = "wasm"))]
type OrientedLookup = HashMap<u64, OrientedKmerIndex, BuildHasherDefault<NoHashHasher<u64>>>;

#[cfg(not(target_family = "wasm"))]
fn indexed_edge(
    hash: u64,
    lookup: &OrientedLookup,
    canonical_type: EdgeType,
    reverse_type: EdgeType,
) -> Option<IndexedEdge> {
    lookup.get(&hash).copied().map(|entry| IndexedEdge {
        target: entry.index(),
        edge_type: if entry.is_reverse() {
            reverse_type
        } else {
            canonical_type
        },
    })
}

#[cfg(not(target_family = "wasm"))]
fn check_bkg_indexed(
    hc: u64,
    hnc: u64,
    k: usize,
    bases: u8,
    lookup: &OrientedLookup,
) -> Vec<IndexedEdge> {
    bkg_candidates(hc, hnc, k, bases)
        .into_iter()
        .filter_map(|(hash, canonical, reverse)| indexed_edge(hash, lookup, canonical, reverse))
        .collect()
}

#[cfg(not(target_family = "wasm"))]
fn check_fwd_indexed(
    hc: u64,
    hnc: u64,
    k: usize,
    bases: u8,
    lookup: &OrientedLookup,
) -> Vec<IndexedEdge> {
    fwd_candidates(hc, hnc, k, bases)
        .into_iter()
        .filter_map(|(hash, canonical, reverse)| indexed_edge(hash, lookup, canonical, reverse))
        .collect()
}

#[cfg(not(target_family = "wasm"))]
pub(crate) fn populate_indexed_neighbours<IntT>(
    k: usize,
    kmers: &mut IndexedKmers<IntT>,
) -> (usize, usize, usize) {
    kmers.initialise_neighbour_slots();

    let canonical_hashes = &kmers.canonical_hashes;
    let reverse_hashes = &kmers.reverse_hashes;
    let boundary_bases = &kmers.boundary_bases;
    let lookup = &kmers.hash_to_index;
    let neighbours = &mut kmers.neighbours;
    let predecessor_counts = &mut kmers.predecessor_counts;

    let (alone, directed_edge_refs) = neighbours
        .par_iter_mut()
        .zip(predecessor_counts.par_iter_mut())
        .enumerate()
        .map(|(index, (slot, predecessor_count))| {
            let mut edges = check_bkg_indexed(
                canonical_hashes[index],
                reverse_hashes[index],
                k,
                boundary_bases[index],
                lookup,
            );
            *predecessor_count =
                u8::try_from(edges.len()).expect("too many predecessor edges for one k-mer");
            edges.extend(check_fwd_indexed(
                canonical_hashes[index],
                reverse_hashes[index],
                k,
                boundary_bases[index],
                lookup,
            ));
            *slot = edges;
            (usize::from(slot.is_empty()), slot.len())
        })
        .reduce(
            || (0, 0),
            |left, right| (left.0 + right.0, left.1 + right.1),
        );

    (canonical_hashes.len(), alone, directed_edge_refs)
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
    dict: &impl KmerLookup<IntT>,
    k: usize,
) -> Result<Vec<u8>, SpellError>
where
    IntT: for<'a> UInt<'a>,
{
    if hashes.is_empty() {
        return Err(SpellError::Empty);
    }

    let get = |i: usize| -> Result<IntT, SpellError> {
        dict.get_kmer(hashes[i]).ok_or(SpellError::UnknownKmer {
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

/// Public API for assemblers.
pub trait Assemble {
    #[cfg(not(target_family = "wasm"))]
    /// Assembles given data and writes results into the output file.
    #[allow(clippy::too_many_arguments)]
    fn assemble<IntT: for<'a> UInt<'a>>(
        k: usize,
        kmers: &mut IndexedKmers<IntT>,
        timevec: &mut Option<&mut Vec<Instant>>,
        path: &mut Option<PathBuf>,
        correction: CorrectionOpts,
    ) -> Contigs;

    #[cfg(target_family = "wasm")]
    /// Assembles given data and prepares all info for being later transferred to Javascript.
    fn assemble_wasm(
        k: usize,
        indict: &mut HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
        maxminsize: &mut HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
        correction: CorrectionOpts,
    ) -> (Contigs, String, String, String);
}

///////////////////////////////////////////////////////////
/// Basic standalone assembler.
pub struct BasicAsm {}

impl Assemble for BasicAsm {
    #[cfg(not(target_family = "wasm"))]
    #[allow(clippy::too_many_arguments)]
    fn assemble<IntT: for<'a> UInt<'a>>(
        k: usize,
        kmers: &mut IndexedKmers<IntT>,
        timevec: &mut Option<&mut Vec<Instant>>,
        path: &mut Option<PathBuf>,
        correction: CorrectionOpts,
    ) -> Contigs {
        // Unpacked once, so the body below reads exactly as it did when these were parameters.
        let CorrectionOpts {
            do_bubble_collapse,
            do_dead_end_removal,
            pop_ratio,
            tip_nts,
            tip_rctc_nts,
            tip_rctc_cutoff,
            coverage,
            do_ec_removal,
            ec_ratio,
            ec_require_both_flanks,
        } = correction;

        logw(
            "Constructing graph. Searching for neighbours...",
            Some("info"),
        );
        if let Some(timevec) = timevec.as_mut() {
            timevec.push(Instant::now());
        }

        populate_indexed_neighbours(k, kmers);

        if let Some(timevec) = timevec.as_mut() {
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
        }

        kmers.finish_neighbour_search();
        let (hashes, counts, neighbours, predecessor_counts) = kmers.take_graph_inputs();
        let mut ptgraph =
            DbgGraph::from_indexed_kmers(k, hashes, counts, neighbours, predecessor_counts);

        if let Some(timevec) = timevec.as_mut() {
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
            )
        }

        logw("Starting graph correction", Some("info"));

        logw("Removing self-loops", Some("info"));
        ptgraph.remove_self_loops();

        let t_phase1 = Instant::now();
        loop {
            let shrink_changed_graph = ptgraph.shrink();
            let pruned = if do_dead_end_removal {
                ptgraph.remove_dead_paths(tip_nts, tip_rctc_nts, tip_rctc_cutoff)
            } else {
                false
            };
            if !shrink_changed_graph && !pruned {
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
                changed |= remove_bulges(&mut ptgraph, pop_ratio, &coverage);
            }
            if do_ec_removal {
                changed |=
                    remove_erroneous_connections(&mut ptgraph, ec_ratio, ec_require_both_flanks);
            }
            changed |= ptgraph.shrink();
            if do_dead_end_removal {
                changed |= ptgraph.remove_dead_paths(tip_nts, tip_rctc_nts, tip_rctc_cutoff);
            }
            if !changed {
                break;
            }
        }
        log::info!(
            "  Bubble correction step (coverage popping to fixed point): {} ms",
            t_phase3.elapsed().as_millis()
        );

        if let Some(timevec) = timevec.as_mut() {
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
        }

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

        if let Some(timevec) = timevec.as_mut() {
            timevec.push(Instant::now());
        }
        let serialized_contigs = ptgraph.collapse();
        if let Some(timevec) = timevec.as_mut() {
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
        }

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
        correction: CorrectionOpts,
    ) -> (Contigs, String, String, String) {
        // Unpacked once, so the body below reads exactly as it did when these were parameters.
        let CorrectionOpts {
            do_bubble_collapse,
            do_dead_end_removal,
            pop_ratio,
            tip_nts,
            tip_rctc_nts,
            tip_rctc_cutoff,
            coverage,
            // EC removal is native-only, so the browser never reads these.
            do_ec_removal: _,
            ec_ratio: _,
            ec_require_both_flanks: _,
        } = correction;

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
            let shrink_changed_graph = ptgraph.shrink();
            let pruned = if do_dead_end_removal {
                ptgraph.remove_dead_paths(tip_nts, tip_rctc_nts, tip_rctc_cutoff)
            } else {
                false
            };
            if !shrink_changed_graph && !pruned {
                break;
            }
        }

        loop {
            let mut changed = false;
            if do_bubble_collapse {
                changed |= ptgraph.correct_bubbles(pop_ratio, &coverage);
            }
            changed |= ptgraph.shrink();
            if do_dead_end_removal {
                changed |= ptgraph.remove_dead_paths(tip_nts, tip_rctc_nts, tip_rctc_cutoff);
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
    use crate::kmer::Kmer;
    use nohash_hasher::NoHashHasher;
    use std::{borrow::Cow, collections::HashMap, hash::BuildHasherDefault};

    fn empty_thedict() -> HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>> {
        HashMap::with_hasher(BuildHasherDefault::default())
    }

    fn empty_maxmindict() -> HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>> {
        HashMap::with_hasher(BuildHasherDefault::default())
    }

    fn indexed_from_dicts(
        packed: &HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
        dict: &HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
    ) -> IndexedKmers<u64> {
        let mut indexed = IndexedKmers::with_capacity(dict.len());
        for (&hash, info) in dict {
            indexed.push(
                hash,
                info.hnc,
                info.b,
                info.counts,
                *packed.get(&hash).unwrap(),
            );
        }
        indexed
    }

    /// The graph native assembly builds for a sequence, with the packed k-mers that spell it.
    #[allow(clippy::type_complexity)]
    fn indexed_graph(
        seq: &[u8],
        k: usize,
    ) -> (
        HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
        DbgGraph,
    ) {
        let (packed, dict, _) = build_dicts(seq, k);
        let mut indexed = indexed_from_dicts(&packed, &dict);
        populate_indexed_neighbours(k, &mut indexed);
        indexed.finish_neighbour_search();
        let (hashes, counts, neighbours, predecessor_counts) = indexed.take_graph_inputs();
        let graph = DbgGraph::from_indexed_kmers(k, hashes, counts, neighbours, predecessor_counts);
        (packed, graph)
    }

    #[test]
    fn indexed_neighbours_match_legacy_neighbours() {
        let k = 15;
        let seq = seq_with_repeat(k);
        let (packed, mut dict, maxmin) = build_dicts(&seq, k);
        let mut indexed = indexed_from_dicts(&packed, &dict);

        let legacy_stats = populate_neighbours(k, &mut dict, &maxmin);
        let indexed_stats = populate_indexed_neighbours(k, &mut indexed);
        assert_eq!(indexed_stats, legacy_stats);

        for (index, &hash) in indexed.canonical_hashes.iter().enumerate() {
            let split = usize::from(indexed.predecessor_counts[index]);
            let mut indexed_pre: Vec<_> = indexed.neighbours[index][..split]
                .iter()
                .map(|edge| (indexed.canonical_hashes[edge.target], edge.edge_type))
                .collect();
            let mut indexed_post: Vec<_> = indexed.neighbours[index][split..]
                .iter()
                .map(|edge| (indexed.canonical_hashes[edge.target], edge.edge_type))
                .collect();
            let legacy = dict.get(&hash).unwrap();
            let mut legacy_pre = legacy.pre.clone();
            let mut legacy_post = legacy.post.clone();
            indexed_pre.sort_unstable();
            indexed_post.sort_unstable();
            legacy_pre.sort_unstable();
            legacy_post.sort_unstable();
            assert_eq!(indexed_pre, legacy_pre, "predecessors differ for {hash}");
            assert_eq!(indexed_post, legacy_post, "successors differ for {hash}");
        }

        let legacy_graph = DbgGraph::from_kmer_map(k, &dict);
        let indexed_graph = DbgGraph::from_indexed_kmers(
            k,
            indexed.canonical_hashes,
            indexed.counts,
            indexed.neighbours,
            indexed.predecessor_counts,
        );
        assert_eq!(indexed_graph.node_count(), legacy_graph.node_count());
        assert_eq!(indexed_graph.edge_count(), legacy_graph.edge_count());
        assert!(indexed_graph.validate().is_ok());
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
        use super::spell_path;
        use crate::algorithms::shrinker::Shrinkable;

        let k = 15;
        let seq = seq_with_repeat(k);
        let (packed, mut g) = indexed_graph(&seq, k);
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
        let (packed, g) = indexed_graph(&seq, k);

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
        assert!(
            spelled == seq || spelled == rc,
            "spelled an unrelated k-mer"
        );
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
