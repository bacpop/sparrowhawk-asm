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
    use sparrowhawk_graph::{EdgeType, NodeStruct};
    use std::{
        borrow::Cow,
        collections::{HashMap, HashSet},
        hash::BuildHasherDefault,
    };

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

    /// The two-bit codes a node reads as when entered on `carry`. A lone k-mer's carry is its
    /// strand outright: `Min` is the stored (canonical) k-mer and `Max` its reverse complement. A
    /// shrunk node's carry picks the order of its k-mers, and that order fixes the strand it spells.
    fn strand_codes(
        g: &DbgGraph,
        packed: &HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
        n: sparrowhawk_graph::NodeId,
        carry: sparrowhawk_graph::CarryType,
        k: usize,
    ) -> Vec<u8> {
        let h = oriented(g, n, carry);
        if h.len() == 1 {
            let kmer = packed[&h[0]];
            let kmer = match carry {
                sparrowhawk_graph::CarryType::Min => kmer,
                sparrowhawk_graph::CarryType::Max => kmer.rev_comp(k),
            };
            (0..k).map(|i| kmer.get_one_code(k - 1 - i)).collect()
        } else {
            crate::graph_works::spelling::spell_path(&h, |hash| packed.get(&hash).copied(), k)
                .unwrap_or_else(|e| panic!("node {n:?} on {carry:?} is not a walk: {e}"))
        }
    }

    /// Reproducible DNA without prescribing which canonical strand each k-mer uses.
    fn round_trip_sequence(len: usize) -> Vec<u8> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                b"ACGT"[((state >> 33) & 3) as usize]
            })
            .collect()
    }

    fn reverse_complement(seq: &[u8]) -> Vec<u8> {
        seq.iter()
            .rev()
            .map(|base| match base {
                b'A' => b'T',
                b'C' => b'G',
                b'G' => b'C',
                b'T' => b'A',
                _ => panic!("the fixture contains only ACGT"),
            })
            .collect()
    }

    /// Exercise the same three operations as assembly, keeping the walk as well as its bases
    /// available to the independent sequence/order assertions below.
    fn collapse_and_spell(
        graph: DbgGraph,
        packed: &HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
        k: usize,
    ) -> (Vec<u64>, Vec<u8>) {
        let mut contigs = Contigs::new(graph.collapse());
        assert_eq!(contigs.serialized_contigs.len(), 1, "one complete contig");
        let before_shrink: Vec<_> = contigs.serialized_contigs[0]
            .iter()
            .flat_map(|node| node.abs_ind.iter().copied())
            .collect();
        contigs.shrink();
        assert_eq!(contigs.serialized_contigs[0].len(), 1);
        let walk = contigs.serialized_contigs[0][0].abs_ind.clone();
        assert_eq!(
            walk, before_shrink,
            "shrinking preserves the collapsed walk"
        );
        let spelled = spelling::spell_contigs(&contigs, |hash| packed.get(&hash).copied(), k);
        assert_eq!(spelled.len(), 1);
        let bases = spelled.iter().next().unwrap().ascii();
        (walk, bases)
    }

    /// A circle has neither an external start nor a preferred rotation or strand. Collapse must
    /// still emit every k-mer once, with the closing k-1 overlap present in its spelled sequence.
    #[test]
    fn collapse_shrink_and_spell_round_trips_a_cycle() {
        let k = 15;
        let seq = round_trip_sequence(400);
        let mut closed = seq.clone();
        closed.extend_from_slice(&seq[..k - 1]);
        let (packed, graph) = indexed_graph(&closed, k);
        assert_eq!(packed.len(), seq.len(), "the circle has no repeated k-mers");
        assert!(graph.externals_bi().is_empty(), "a true circle has no ends");

        let (walk, spelled) = collapse_and_spell(graph, &packed, k);
        assert_eq!(walk.len(), seq.len());
        assert_eq!(
            walk.into_iter().collect::<HashSet<_>>(),
            packed.keys().copied().collect(),
            "every circular k-mer appears exactly once"
        );
        assert_eq!(spelled.len(), seq.len() + k - 1);
        assert_eq!(spelled[..k - 1], spelled[seq.len()..]);
        let doubled = [seq.as_slice(), seq.as_slice()].concat();
        let reverse = reverse_complement(&doubled);
        let core = &spelled[..seq.len()];
        assert!(
            doubled.windows(seq.len()).any(|window| window == core)
                || reverse.windows(seq.len()).any(|window| window == core),
            "the circle's bases are preserved up to rotation and reverse complement"
        );
    }

    /// The middle node is stored backwards. Following its Max carry must reverse its k-mer order
    /// before it can be joined to its flanks; merely concatenating stored node payloads cannot pass.
    #[test]
    fn collapse_shrink_and_spell_handles_walks_against_innerdir() {
        let k = 15;
        let seq = round_trip_sequence(180);
        let (packed, _) = build_indexed(&seq, k);
        let mut kmers = Kmer::<u64>::new(Cow::Borrowed(&seq), seq.len(), None, k, 0, true).unwrap();
        let mut walk = vec![kmers.get_hashes_and_bases_and_kmer().0];
        while let Some((hash, _, _, _)) = kmers.get_next_kmer() {
            walk.push(hash);
        }
        assert_eq!(packed.len(), walk.len(), "the fixture has unique k-mers");
        let reverse_walk: Vec<_> = walk.iter().rev().copied().collect();
        let split = walk.len() / 3;
        let stored = [
            walk[..split].to_vec(),
            walk[split..2 * split].iter().rev().copied().collect(),
            walk[2 * split..].to_vec(),
        ];
        let uncorrected: Vec<_> = stored.iter().flatten().copied().collect();
        assert_ne!(uncorrected, walk);
        assert_ne!(uncorrected, reverse_walk);

        for order in [[0, 1, 2], [2, 1, 0]] {
            let mut graph = DbgGraph::new(k);
            let mut ids = [None; 3];
            for i in order {
                ids[i] = Some(graph.add_node(NodeStruct {
                    counts: 10,
                    abs_ind: stored[i].clone(),
                    innerdir: Some(EdgeType::MinToMin),
                }));
            }
            let [a, b, c] = ids.map(Option::unwrap);
            graph.add_bi_edge(a, b, EdgeType::MinToMax);
            graph.add_bi_edge(b, c, EdgeType::MaxToMin);

            let (collapsed, spelled) = collapse_and_spell(graph, &packed, k);
            assert!(
                collapsed == walk || collapsed == reverse_walk,
                "order {order:?}"
            );
            assert_eq!(spelled.len(), seq.len(), "both terminal k-mers survive");
            assert!(
                spelled == seq || spelled == reverse_complement(&seq),
                "order {order:?}: spelling preserves the complete sequence"
            );
        }
    }

    /// Every edge in a real graph joins two k-mers that genuinely overlap by `k-1`, in the strands its
    /// type names: the source read on its carry ends with the `k-1` bases the target, read on its
    /// own, starts with. Letting `spell_path` choose a strand would also accept an edge whose type is
    /// wrong.
    #[test]
    fn every_edge_is_a_valid_k_minus_one_junction() {
        use crate::graph_works::shrinker::Shrinkable;

        let k = 15;
        let seq = seq_with_repeat(k);
        let (packed, mut g) = indexed_graph(&seq, k);
        g.remove_self_loops();
        g.shrink();

        let mut checked = 0usize;
        for n in g.node_indices().collect::<Vec<_>>() {
            for (m, et) in g.outgoing_edges(n) {
                let (sc, tc) = et.get_from_and_to();
                let from = strand_codes(&g, &packed, n, sc, k);
                let to = strand_codes(&g, &packed, m, tc, k);
                assert_eq!(
                    from[from.len() - (k - 1)..],
                    to[..k - 1],
                    "edge {n:?} -{et:?}-> {m:?} is not a k-1 junction in the strands it names"
                );
                checked += 1;
            }
        }
        assert!(checked > 0, "the test graph has no edges to check");
    }
}
