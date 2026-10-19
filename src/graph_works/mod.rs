//! Construct, correct, simplify, and collapse DNA de Bruijn graphs.

pub mod collapser;
pub mod corrector;
pub mod indexed_kmers;
pub mod neighbours;
pub mod path_correction;
pub mod shrinker;
pub mod spelling;

use self::{
    collapser::Collapsable,
    corrector::{Correctable, CorrectionOpts},
    indexed_kmers::IndexedKmers,
    neighbours::populate_indexed_neighbours,
    path_correction::{remove_bulges, remove_erroneous_connections},
    shrinker::Shrinkable,
};
use crate::utils::{bit_encoding::UInt, clock::Instant};
use sparrowhawk_graph::{DbgGraph, SerializedContigs};

////////////////////////////////////////////////////////////////////////
/// Output from the assembler.
#[derive(Default)]
pub struct Contigs {
    /// Serialized contigs.
    pub serialized_contigs: SerializedContigs,
}

impl Contigs {
    /// Create new `Contigs`.
    pub fn new(serialized: SerializedContigs) -> Contigs {
        Contigs {
            serialized_contigs: serialized,
        }
    }

    /// Join each collapsed contig's k-mer walk into one node before sequence spelling.
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
}

/// Public API for assemblers.
pub trait Assemble {
    /// Build the graph of `kmers`, correct it, and collapse it into contigs. `graph_sink` sees the
    /// corrected graph before the collapse: natively it writes DOT and GFA files, and in the browser it
    /// keeps them as text for the page.
    fn assemble<IntT: for<'a> UInt<'a>>(
        k: usize,
        kmers: &mut IndexedKmers<IntT>,
        timevec: &mut Option<&mut Vec<Instant>>,
        graph_sink: Option<&mut dyn FnMut(&DbgGraph)>,
        correction: CorrectionOpts,
    ) -> Contigs;
}

/// Seconds between the last two stamps of `timevec`.
fn last_step_secs(timevec: &[Instant]) -> u64 {
    timevec[timevec.len() - 1]
        .duration_since(timevec[timevec.len() - 2])
        .as_secs()
}

///////////////////////////////////////////////////////////
/// Basic standalone assembler.
pub struct BasicAsm {}

impl Assemble for BasicAsm {
    fn assemble<IntT: for<'a> UInt<'a>>(
        k: usize,
        kmers: &mut IndexedKmers<IntT>,
        timevec: &mut Option<&mut Vec<Instant>>,
        graph_sink: Option<&mut dyn FnMut(&DbgGraph)>,
        correction: CorrectionOpts,
    ) -> Contigs {
        // Unpacked once, so the body below reads exactly as it did when these were parameters.
        let CorrectionOpts {
            do_bubble_collapse,
            do_dead_end_removal,
            pop_ratio,
            dead_end_nts,
            dead_end_coverage_nts,
            dead_end_coverage_ratio,
            coverage,
            do_ec_removal,
            ec_ratio,
            ec_require_both_flanks,
        } = correction;

        crate::post_state("assembly:create_graph");
        log::info!("Constructing graph. Searching for neighbours...");
        if let Some(timevec) = timevec.as_mut() {
            timevec.push(Instant::now());
        }

        populate_indexed_neighbours(k, kmers);

        if let Some(timevec) = timevec.as_mut() {
            timevec.push(Instant::now());
            log::info!(
                "Neighbours searched for in {} s. Creating graph...",
                last_step_secs(timevec)
            );
        }

        kmers.finish_neighbour_search();
        let (hashes, counts, neighbours, predecessor_counts) = kmers.take_graph_inputs();
        let mut ptgraph =
            DbgGraph::from_indexed_kmers(k, hashes, counts, neighbours, predecessor_counts);

        if let Some(timevec) = timevec.as_mut() {
            timevec.push(Instant::now());
            log::info!("Graph created in {} s.", last_step_secs(timevec));
        }

        crate::post_state("assembly:correct_graph");
        log::info!("Starting graph correction");

        log::info!("Removing self-loops");
        ptgraph.remove_self_loops();

        let t_phase1 = Instant::now();
        loop {
            let shrink_changed_graph = ptgraph.shrink();
            let pruned = if do_dead_end_removal {
                ptgraph.remove_dead_paths(
                    dead_end_nts,
                    dead_end_coverage_nts,
                    dead_end_coverage_ratio,
                )
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
                changed |= ptgraph.remove_dead_paths(
                    dead_end_nts,
                    dead_end_coverage_nts,
                    dead_end_coverage_ratio,
                );
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
            log::info!(
                "Graph correction finished in {} s.",
                last_step_secs(timevec)
            );
        }

        if let Some(sink) = graph_sink {
            sink(&ptgraph);
        }

        crate::post_state("assembly:collapse_graph");
        if let Some(timevec) = timevec.as_mut() {
            timevec.push(Instant::now());
        }
        let serialized_contigs = ptgraph.collapse();
        if let Some(timevec) = timevec.as_mut() {
            timevec.push(Instant::now());
            log::info!("Graph collapse finished in {} s.", last_step_secs(timevec));
            log::info!("I created {} contigs", serialized_contigs.len());
        }

        let mut contigs = Contigs::new(serialized_contigs);

        // Sequence spelling consumes one node containing the complete walk per contig.
        contigs.shrink();

        contigs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessing::kmer::Kmer;
    use nohash_hasher::NoHashHasher;
    use std::{borrow::Cow, collections::HashMap, hash::BuildHasherDefault};

    /// The packed k-mers of `seq`, and the `IndexedKmers` preprocessing would build from them.
    #[allow(clippy::type_complexity)]
    fn build_indexed(
        seq: &[u8],
        k: usize,
    ) -> (
        HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
        IndexedKmers<u64>,
    ) {
        let mut packed = HashMap::with_hasher(BuildHasherDefault::default());
        let mut counted: HashMap<u64, (u64, u8, u32), BuildHasherDefault<NoHashHasher<u64>>> =
            HashMap::with_hasher(BuildHasherDefault::default());
        let mut it = Kmer::<u64>::new(Cow::Borrowed(seq), seq.len(), None, k, 0, true).unwrap();
        let mut cur = Some(it.get_hashes_and_bases_and_kmer());
        while let Some((hc, hnc, b, km)) = cur {
            packed.insert(hc, km);
            counted
                .entry(hc)
                .and_modify(|e| e.2 += 1)
                .or_insert((hnc, b, 1));
            cur = it.get_next_kmer();
        }
        let mut indexed = IndexedKmers::with_capacity(counted.len());
        for (&hash, &(hnc, b, counts)) in &counted {
            indexed.push(hash, hnc, b, counts, packed[&hash]);
        }
        (packed, indexed)
    }

    /// The graph the assembler builds for a sequence, with the packed k-mers that spell it.
    #[allow(clippy::type_complexity)]
    fn indexed_graph(
        seq: &[u8],
        k: usize,
    ) -> (
        HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
        DbgGraph,
    ) {
        let (packed, mut indexed) = build_indexed(seq, k);
        populate_indexed_neighbours(k, &mut indexed);
        indexed.finish_neighbour_search();
        let (hashes, counts, neighbours, predecessor_counts) = indexed.take_graph_inputs();
        let graph = DbgGraph::from_indexed_kmers(k, hashes, counts, neighbours, predecessor_counts);
        (packed, graph)
    }

    /// A deterministic ACGT sequence, with a repeat planted twice so the graph actually branches
    /// rather than being one long chain.
    fn seq_with_repeat(k: usize) -> Vec<u8> {
        const BASES: [u8; 4] = *b"ACGT";
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
        use crate::graph_works::shrinker::Shrinkable;
        use crate::graph_works::spelling::spell_path;

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
                    spell_path(&h, |hash| packed.get(&hash).copied(), k)
                        .unwrap_or_else(|e| panic!("node {n:?} on {carry:?} is not a walk: {e}"));
                }
            }
            // And each outgoing edge must join two overlapping k-mers.
            for (m, et) in g.outgoing_edges(n) {
                let (sc, tc) = et.get_from_and_to();
                let from = *oriented(&g, n, sc).last().unwrap();
                let to = oriented(&g, m, tc)[0];
                spell_path(&[from, to], |hash| packed.get(&hash).copied(), k).unwrap_or_else(|e| {
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
        use crate::graph_works::spelling::{spell_path, SpellError};

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
            spell_path(
                &[all[0], all[all.len() - 1]],
                |hash| packed.get(&hash).copied(),
                k
            ),
            Err(SpellError::NotAWalk { index: 1 }),
            "an invented junction must be rejected"
        );
    }
}
