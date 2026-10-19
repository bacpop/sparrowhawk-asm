//! Corrects parts of the provided graph, if needed
use crate::preprocessing::PeakSource;
use sparrowhawk_graph::{CarryType, DbgGraph, EdgeId, EdgeType, NodeId, NodeStruct};

use crate::EdgeWeight;

use std::{
    collections::{BTreeMap, BTreeSet},
    vec::Drain,
};

/// Minimum number of k-mers a path must exceed to be worth keeping in dead-end removal.
pub(crate) fn short_path_limit(minnts: usize, k: usize) -> usize {
    (minnts + 1).saturating_sub(k) // sat_sub is compulsory, becase as these are usize, going negative my change the path to an absurd value!!!
}

/// Dead-end removal threshold in bases: the larger of a flat floor and a multiple of k, as Minia sizes it
/// (`-tip-len-topo-kmult`). A flat 100 nt is generous at k=31 and catches almost nothing at k=81.
pub(crate) fn dead_end_length_nts(flat_nts: usize, kmult: f32, k: usize) -> usize {
    if kmult <= 0.0 {
        flat_nts
    } else {
        flat_nts.max((kmult * k as f32).round() as usize)
    }
}

/// Mean coverage of a path, weighted by how many k-mers each node contributes.
///
/// Lives here rather than in `path_correction` because that module is not compiled for wasm
/// (`algorithms.rs:5`) while dead-end removal is, and both need the same accumulator.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PathCoverage {
    pub(crate) weighted_sum: u128,
    pub(crate) kmers: u64,
}

impl PathCoverage {
    pub(crate) fn add_node(&mut self, node: &NodeStruct) {
        self.add_part(node, node.abs_ind.len());
    }

    pub(crate) fn add_part(&mut self, node: &NodeStruct, kmers: usize) {
        self.weighted_sum += node.counts as u128 * kmers as u128;
        self.kmers = self.kmers.saturating_add(kmers as u64);
    }

    pub(crate) fn mean(self) -> f64 {
        debug_assert!(self.kmers > 0);
        self.weighted_sum as f64 / self.kmers as f64
    }
}

/// Mean coverage of an ordered path, weighted by each node's k-mers.
pub(crate) fn path_mean_coverage(ptgraph: &DbgGraph, path: &[NodeId]) -> f64 {
    let mut cov = PathCoverage::default();
    for n in path {
        cov.add_node(ptgraph.node_weight(*n).unwrap());
    }
    cov.mean()
}

/// A branch carrying less than this fraction of the stronger branch's coverage is noise. SKESA-inspired.
pub const DEFAULT_POP_RATIO: f32 = 0.1;

/// A connector goes when its flanks carry this many times its coverage (Minia's `_ecRCTCcutoff`).
/// Here rather than in `path_correction`: `cli` compiles for wasm and that module does not.
pub const DEFAULT_EC_COVERAGE_RATIO: f64 = 4.0;

/// A branch below this fraction of fitted single-copy coverage cannot be a copy of anything.
pub const DEFAULT_ERROR_COVERAGE_FRACTION: f32 = 0.25;

/// The same, for a fallback estimate. Smaller because the occurrence-weighted median is biased
/// *upward* by repeats, and an overstated peak raises the ceiling and deletes real sequence.
pub const FALLBACK_ERROR_COVERAGE_FRACTION: f32 = 0.15;

/// What correction knows about the library's coverage, and how far it can be trusted.
///
/// Written so that an absent peak reproduces the historical behaviour exactly: every predicate below
/// then defers to the caller's other tests rather than vetoing or forcing anything.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CoverageRef {
    /// Single-copy coverage as a k-mer count, or `None` when none was established.
    pub genomic_peak: Option<u32>,
    /// The cutoff every node in the graph already clears; it bounds what the peak can be compared to.
    pub min_count: u16,
    /// Fraction of the peak under which a branch is an error rather than a copy.
    pub error_fraction: f32,
}

impl CoverageRef {
    /// Build from what preprocessing established, matching the fraction to how far it can be trusted.
    pub fn new(peak: PeakSource, min_count: u16) -> Self {
        let (genomic_peak, error_fraction) = match peak {
            PeakSource::Fitted(p) => (Some(p), DEFAULT_ERROR_COVERAGE_FRACTION),
            PeakSource::Fallback(p) => (Some(p), FALLBACK_ERROR_COVERAGE_FRACTION),
            PeakSource::Unknown => (None, DEFAULT_ERROR_COVERAGE_FRACTION),
        };
        Self {
            genomic_peak,
            min_count,
            error_fraction,
        }
    }

    /// Override the fraction, for a caller that asked for one explicitly. `None` keeps whatever
    /// [`Self::new`] chose for this peak's provenance.
    pub fn with_error_fraction(mut self, fraction: Option<f32>) -> Self {
        if let Some(fraction) = fraction {
            self.error_fraction = fraction;
        }
        self
    }

    /// Nothing was established, so every rule falls back to the sibling-ratio test alone. Not a hole:
    /// an unresolved spectrum also drops `min_count` to 2, which is the regime where that test works.
    pub fn unknown() -> Self {
        Self::new(PeakSource::Unknown, 0)
    }

    /// The count below which a node cannot be a real single copy.
    ///
    /// The `min_count + 1` floor is the point of this: a fitted `min_count` measures 0.16-0.30 of the
    /// peak, so at the top of that range a bare fraction lands below a cutoff every surviving node
    /// already clears, leaving the predicate unsatisfiable — how the ratio rule fails today.
    #[inline]
    pub fn error_ceiling(&self) -> Option<u32> {
        let peak = self.genomic_peak?;
        Some(((self.error_fraction * peak as f32).floor() as u32).max(self.min_count as u32 + 1))
    }

    /// Too far under single-copy coverage to be a copy of anything. False when no peak is known.
    #[inline]
    pub fn is_error_like(&self, counts: u32) -> bool {
        self.error_ceiling().is_some_and(|ceiling| counts < ceiling)
    }

    /// Consistent with at least one real copy. True when no peak is known, so an unknown coverage
    /// cannot by itself veto a pop. Exact complement of [`Self::is_error_like`], which is what keeps
    /// an equal-coverage bubble unpoppable whatever the peak.
    #[inline]
    pub fn is_genomic(&self, counts: u32) -> bool {
        self.error_ceiling().is_none_or(|ceiling| counts >= ceiling)
    }
}

/// Everything the correction phase is steered by. A struct rather than four more parameters because
/// `assemble` already takes nine, and because every future correction knob lands here instead.
#[derive(Clone, Copy, Debug)]
pub struct CorrectionOpts {
    /// Collapse bubbles at all.
    pub do_bubble_collapse: bool,
    /// Remove dead-end paths at all.
    pub do_dead_end_removal: bool,
    /// Fraction of the stronger branch's coverage below which the weaker branch is popped.
    pub pop_ratio: f32,
    /// Dead-end threshold in bases, already resolved against k.
    pub dead_end_nts: usize,
    /// Upper bound of dead-end removal by coverage, already resolved against k. Zero disables it.
    pub dead_end_coverage_nts: usize,
    /// How many times better covered a junction must be than the dead end hanging off it.
    pub dead_end_coverage_ratio: f64,
    /// What is known about the library's single-copy coverage.
    pub coverage: CoverageRef,
    /// Remove erroneous connections at all.
    pub do_ec_removal: bool,
    /// A connector goes when its flanks carry this many times its coverage.
    pub ec_ratio: f64,
    /// Require *both* flanks over the ratio. Minia requires either; see `path_correction`.
    pub ec_require_both_flanks: bool,
}

/// Mark graph as correctable.
pub trait Correctable {
    /// Edge index associated with collection.
    type EdgeIdx;

    /// Node index associated with collection.
    type NodeIdx;

    /// Remove edges with weight below threshold.
    fn remove_weak_nodes(&mut self, threshold: EdgeWeight);

    /// Remove edges that are self-loops, i.e. those whose source and destination nodes are the same.
    fn remove_self_loops(&mut self);

    /// Remove dead paths shorter than `dead_end_nts` bases, and those up to `dead_end_coverage_nts` whose junction
    /// is `coverage_ratio` times better covered.
    fn remove_dead_paths(
        &mut self,
        dead_end_nts: usize,
        dead_end_coverage_nts: usize,
        coverage_ratio: f64,
    ) -> bool;
}

impl Correctable for DbgGraph {
    type EdgeIdx = EdgeId;

    type NodeIdx = NodeId;

    fn remove_weak_nodes(&mut self, threshold: EdgeWeight) {
        self.retain_nodes_by_count(threshold);
    }

    fn remove_self_loops(&mut self) {
        DbgGraph::remove_self_loops(self);
    }

    fn remove_dead_paths(
        &mut self,
        dead_end_nts: usize,
        dead_end_coverage_nts: usize,
        coverage_ratio: f64,
    ) -> bool {
        log::info!(
            "Before pruning: {} nodes and {} edges",
            self.node_count(),
            self.edge_count()
        );

        // Hoisted out of the walk: it only depends on k and the threshold, both fixed for this call.
        let limit = short_path_limit(dead_end_nts, self.k());
        // An empty band disables tier two outright: the walk bound falls back to `limit` and not one
        // comparison is made, so `--dead-end-coverage-kmult 0` reproduces today's output exactly.
        let coverage_limit = short_path_limit(dead_end_coverage_nts, self.k()).max(limit);
        let (mut by_length, mut by_coverage) = (0usize, 0usize);
        let mut dididoanything = false;
        log::info!("Starting graph pruning. Graph has {} externals, {} alone nodes, the remaining are internal.",
            self.externals_bi().len(),
            self.node_indices().filter(|n| self.out_degree(*n) == 0 && self.in_degree(*n) == 0).count());

        let mut to_remove: Vec<NodeId> = vec![];
        loop {
            let mut path_check_vec = vec![];
            let externals: Vec<_> = self
                .externals_bi()
                .into_iter()
                // A dead end with several neighbours is its own fan-out; a self-loop has no dead side.
                .filter(|n| match self.out_degree(*n) {
                    0 => false,
                    1 => true,
                    _ => !self.has_self_loop(*n),
                })
                .collect();

            log::trace!("Detected {} externals", externals.len());

            for v in externals {
                let carryedge = self.first_outgoing_edge_type(v).unwrap();
                let removed_by_coverage = check_dead_path(
                    self,
                    v,
                    &mut path_check_vec,
                    limit,
                    coverage_limit,
                    coverage_ratio,
                    carryedge,
                );
                if !path_check_vec.is_empty() {
                    dididoanything = true;
                    if removed_by_coverage {
                        by_coverage += 1;
                    } else {
                        by_length += 1;
                    }
                    to_remove.append(&mut path_check_vec);
                }
            }

            // if there are no dead paths left pruning is done
            if to_remove.is_empty() {
                log::info!(
                    "Graph is pruned: {by_length} dead ends by length (threshold {dead_end_nts} nt, \
                     {limit} k-mers), {by_coverage} by coverage (up to {dead_end_coverage_nts} nt, \
                     {coverage_limit} k-mers, cutoff {coverage_ratio})"
                );
                return dididoanything;
            }

            // reverse sort edge indices such that removal won't cause any troubles with swapped
            // edge indices (see `petgraph`'s explanation of `remove_edge`)
            // NOTE: this might be useful for the change to Graph backend, but now we don't need it.
            // to_remove.sort_by(|a, b| b.cmp(a));
            remove_paths(self, to_remove.drain(..));
        }
    }
}

/// Remove unpaired and duplicate incident edges.
///
/// At most one correctly typed reciprocal pair is retained for each connection.
/// Returns the number of directed edges removed.
pub(crate) fn prune_unpaired_edges(g: &mut DbgGraph, n: NodeId) -> usize {
    type Connection = (NodeId, NodeId, EdgeType);

    let mut outgoing = g.outgoing_edges_by_carry(n, CarryType::Min);
    outgoing.extend(g.outgoing_edges_by_carry(n, CarryType::Max));
    let incoming = g.incoming_edges(n);
    if edges_are_paired(&outgoing, &incoming) {
        return 0;
    }

    let mut incident: BTreeMap<Connection, BTreeSet<EdgeId>> = BTreeMap::new();

    for (edge_id, target, edge_type) in outgoing {
        incident
            .entry((n, target, edge_type))
            .or_default()
            .insert(edge_id);
    }

    for (source, edge_type) in incoming {
        let connection = (source, n, edge_type);
        if incident.contains_key(&connection) {
            continue;
        }

        let edge_ids = g
            .edges_between(source, n)
            .into_iter()
            .filter(|&edge_id| g.edge_weight(edge_id).unwrap().t == edge_type)
            .collect();

        incident.insert(connection, edge_ids);
    }

    let mut to_remove: BTreeMap<EdgeId, Connection> = BTreeMap::new();

    for (&(from, to, edge_type), edge_ids) in &incident {
        let reverse = (to, from, edge_type.rev());
        let retained = usize::from(incident.contains_key(&reverse));

        for &edge_id in edge_ids.iter().skip(retained) {
            to_remove.insert(edge_id, (from, to, edge_type));
        }
    }

    for (&edge_id, &(from, to, edge_type)) in &to_remove {
        log::warn!("Removing excess or unpaired edge {from:?} -{edge_type:?}-> {to:?}");
        g.remove_edge(edge_id);
    }

    to_remove.len()
}

/// Whether every edge at a node has exactly one reciprocal (opposite direction, `rev` type), so
/// [`prune_unpaired_edges`] would remove nothing. Shrinking prunes every absorbed node and this
/// is nearly always the case, so it is checked first, without that function's maps.
fn edges_are_paired(
    outgoing: &[(EdgeId, NodeId, EdgeType)],
    incoming: &[(NodeId, EdgeType)],
) -> bool {
    outgoing.len() == incoming.len()
        && outgoing.iter().all(|&(_, target, edge_type)| {
            incoming
                .iter()
                .filter(|&&(source, t)| source == target && t == edge_type.rev())
                .count()
                == 1
        })
        && incoming.iter().all(|&(source, edge_type)| {
            outgoing
                .iter()
                .filter(|&&(_, target, t)| target == source && t == edge_type.rev())
                .count()
                == 1
        })
}

/// Remove dead input path.
#[inline]
fn remove_paths(ptgraph: &mut DbgGraph, to_remove: Drain<NodeId>) {
    log::trace!("Removing {} dead paths", to_remove.len());
    for n in to_remove {
        ptgraph.remove_node(n);
    }
}

/// Whether the junction a dead end hangs off is much better covered than the dead end: its other
/// neighbours average more than `cutoff` times the dead end's mean coverage.
fn neighbourhood_outcovers_dead_end(
    ptgraph: &DbgGraph,
    junction: NodeId,
    junction_ty: CarryType,
    dead_end_last: NodeId,
    dead_end_mean: f64,
    cutoff: f64,
) -> bool {
    // Minia walks each neighbour's simple path to average it (`getMeanAbundanceOfNeighbors`); ours are
    // already unitigs, so `counts` *is* that mean. `seen` dedupes a node reachable on both sides.
    let mut seen: BTreeSet<NodeId> = BTreeSet::new();
    let mut total = 0.0f64;
    for (nb, _) in ptgraph
        .in_neighbours_bi(junction, junction_ty)
        .into_iter()
        .chain(ptgraph.out_neighbours_bi(junction, junction_ty))
    {
        if nb == dead_end_last || !seen.insert(nb) {
            continue;
        }
        total += f64::from(ptgraph.node_weight(nb).unwrap().counts);
    }
    if seen.is_empty() {
        return false; // nothing to compare against: an isolated dead end is tier one's business
    }
    // Strict `>`, and `f64` throughout — the cutoff is declared `f64` from the CLI down, as
    // `--ec-coverage-ratio` is, so there is no `f32` hop to shift the boundary the way
    // `f64::from(0.1f32)` did in `should_delete`.
    total / seen.len() as f64 > cutoff * dead_end_mean
}

/// Check if vertex initializes a dead path.
#[inline]
fn check_dead_path(
    ptgraph: &DbgGraph,
    vertex: NodeId,
    output_vec: &mut Vec<NodeId>,
    limit: usize,
    coverage_limit: usize,
    coverage_ratio: f64,
    carryedge: EdgeType,
) -> bool {
    let mut current_vertex = vertex;
    output_vec.push(current_vertex);
    let cntopt = ptgraph.node_weight(current_vertex);
    let mut cnt: usize;
    if let Some(thecntopt) = cntopt {
        cnt = thecntopt.abs_ind.len();
    } else {
        return false;
    }

    let (mut ty, _) = carryedge.get_from_and_to();

    // A dead end that fans out is its own junction, Minia's single-node dead end: its successors are the
    // whole neighbourhood.
    if ptgraph.out_neighbours_bi(current_vertex, ty).len() > 1 {
        if cnt < limit {
            return false;
        }
        if cnt < coverage_limit
            && neighbourhood_outcovers_dead_end(
                ptgraph,
                current_vertex,
                ty,
                current_vertex,
                path_mean_coverage(ptgraph, output_vec),
                coverage_ratio,
            )
        {
            return true;
        }
        output_vec.clear();
        return false;
    }

    loop {
        if cnt >= coverage_limit {
            output_vec.clear();
            return false;
        }

        let fwdneigh = ptgraph.out_neighbours_bi(current_vertex, ty);
        let nfwdn = fwdneigh.len();

        if nfwdn != 1 {
            panic!("Not expected!");
        }

        let candidate_node = fwdneigh[0].0;
        let candidate_ty = fwdneigh[0].1.get_from_and_to().1;
        let bkgneigh_c = ptgraph.in_neighbours_bi(candidate_node, candidate_ty);
        let fwdneigh_c = ptgraph.out_neighbours_bi(candidate_node, candidate_ty);
        let nfwdn_c = fwdneigh_c.len();
        let nbkgn_c = bkgneigh_c.len();

        if nbkgn_c == 0 {
            panic!("Not expected! 2");
        } else if nbkgn_c != 1 {
            if cnt < limit {
                // Tier one, unchanged: short enough to judge on length alone.
                let mut altpath: Vec<(Vec<NodeId>, usize)> = Vec::with_capacity(nbkgn_c - 1);
                let mut max_kmers = 0;
                for n in bkgneigh_c.iter() {
                    if n.0 == *output_vec.last().unwrap() {
                        continue;
                    } else {
                        let mut tmppath: Vec<NodeId> = Vec::new();
                        check_backwards_path(
                            ptgraph,
                            n.0,
                            n.1.get_from_and_to().0,
                            &mut tmppath,
                            limit,
                        );
                        let tmplen = ptgraph
                            .path_kmer_length(&tmppath)
                            .expect("backward path contains a node removed from the graph");
                        if tmplen != 0 {
                            altpath.push((tmppath, tmplen));
                            if tmplen > max_kmers {
                                max_kmers = tmplen;
                            }
                        }
                    }
                }

                // Both values are totals of represented k-mers.
                if max_kmers != 0 && cnt > max_kmers {
                    output_vec.clear();
                    for (path, _) in altpath.iter_mut() {
                        output_vec.append(path);
                    }
                }
                return false;
            }

            // Tier two: past the topological threshold, so length alone will not do it.
            if neighbourhood_outcovers_dead_end(
                ptgraph,
                candidate_node,
                candidate_ty,
                *output_vec.last().unwrap(),
                path_mean_coverage(ptgraph, output_vec),
                coverage_ratio,
            ) {
                return true;
            }
            output_vec.clear();
            return false;
        }

        if nfwdn_c == 0 {
            output_vec.push(candidate_node);
            cnt += ptgraph.node_weight(candidate_node).unwrap().abs_ind.len();

            // Stays tier one: a whole isolated component has no junction to be compared against.
            if cnt >= limit {
                output_vec.clear();
            }
            return false;
        } else if nfwdn_c == 1 {
            current_vertex = candidate_node;
            ty = candidate_ty;
            output_vec.push(current_vertex);
            cnt += ptgraph.node_weight(current_vertex).unwrap().abs_ind.len();
        } else {
            let before = *output_vec.last().unwrap();
            output_vec.push(candidate_node);
            cnt += ptgraph.node_weight(candidate_node).unwrap().abs_ind.len();

            // Keep the existing strict short-dead-end boundary and remove by topology alone below it.
            if cnt < limit {
                return false;
            }

            // Longer dead ends retain the existing relative-coverage rule, now measuring the complete
            // path through its fan-out endpoint.
            if cnt < coverage_limit
                && neighbourhood_outcovers_dead_end(
                    ptgraph,
                    candidate_node,
                    candidate_ty,
                    before,
                    path_mean_coverage(ptgraph, output_vec),
                    coverage_ratio,
                )
            {
                return true;
            }
            output_vec.clear();
            return false;
        }
    }
}

fn check_backwards_path(
    ptgraph: &DbgGraph,
    vertex: NodeId,
    mut ty: CarryType,
    output_vec: &mut Vec<NodeId>,
    kmerlimit: usize,
) {
    let mut current_vertex = vertex;
    output_vec.push(current_vertex);
    let mut cnt = ptgraph.node_weight(current_vertex).unwrap().abs_ind.len();

    loop {
        if cnt >= kmerlimit {
            output_vec.clear();
            return;
        }

        let bkgneigh = ptgraph.in_neighbours_bi(current_vertex, ty);
        let bkgneighlen = bkgneigh.len();
        if bkgneighlen == 0 {
            if ptgraph.out_neighbours_bi(current_vertex, ty).len() != 1 {
                output_vec.clear();
            }
            return;
        } else if bkgneighlen == 1 {
            if ptgraph.out_neighbours_bi(current_vertex, ty).len() != 1 {
                output_vec.clear();
                return;
            }

            current_vertex = bkgneigh[0].0;
            ty = bkgneigh[0].1.get_from_and_to().0;
            output_vec.push(current_vertex);
            cnt += ptgraph.node_weight(current_vertex).unwrap().abs_ind.len();
        } else {
            output_vec.clear();
            return;
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
    use crate::cli::DEFAULT_DEAD_END_COVERAGE_RATIO;
    use sparrowhawk_graph::{DbgGraph, NodeStruct};

    fn make_node() -> NodeStruct {
        NodeStruct {
            counts: 10,
            abs_ind: vec![0u64],
            innerdir: None,
        }
    }

    // ── dead-path limit formula ──────────────────────────────────────────────

    /// Above k = 100 the limit floors at 0 instead of underflowing (`100 - 101 + 1` on `usize` once wrapped
    /// to ~1.8e19 and emptied every k > 100 assembly), so a single k-mer, already k bases, is never dropped.
    #[test]
    fn short_path_limit_cases() {
        // (minimum bases, k, limit in k-mers)
        for (minnts, k, want) in [
            (100, 3, 98),
            (100, 31, 70),
            (100, 100, 1),
            (100, 101, 0),
            (100, 139, 0),
            (100, 255, 0),
        ] {
            assert_eq!(short_path_limit(minnts, k), want, "minnts {minnts}, k {k}");
        }
    }

    #[test]
    fn dead_path_arbitration_compares_kmer_totals() {
        let mut graph = DbgGraph::new(3);
        let dead_end_a = graph.add_node(NodeStruct {
            counts: 1,
            abs_ind: vec![0; 2],
            innerdir: None,
        });
        let middle_a = graph.add_node(NodeStruct {
            counts: 1,
            abs_ind: vec![0; 2],
            innerdir: None,
        });
        let dead_end_b = graph.add_node(NodeStruct {
            counts: 1,
            abs_ind: vec![0; 3],
            innerdir: None,
        });
        let junction = graph.add_node(make_node());

        graph.add_bi_edge(dead_end_a, middle_a, EdgeType::MinToMin);
        graph.add_bi_edge(middle_a, junction, EdgeType::MinToMin);
        graph.add_bi_edge(dead_end_b, junction, EdgeType::MinToMin);

        let limit = 98;
        let mut from_a = Vec::new();
        check_dead_path(
            &graph,
            dead_end_a,
            &mut from_a,
            limit,
            limit, // tier two disabled: an empty band, as `--no-dead-end-coverage` produces
            DEFAULT_DEAD_END_COVERAGE_RATIO,
            graph.first_outgoing_edge_type(dead_end_a).unwrap(),
        );
        assert_eq!(from_a, vec![dead_end_b]);

        let mut from_b = Vec::new();
        check_dead_path(
            &graph,
            dead_end_b,
            &mut from_b,
            limit,
            limit, // tier two disabled: an empty band, as `--no-dead-end-coverage` produces
            DEFAULT_DEAD_END_COVERAGE_RATIO,
            graph.first_outgoing_edge_type(dead_end_b).unwrap(),
        );
        assert_eq!(from_b, vec![dead_end_b]);
    }

    // ── the coverage tier (Minia's RCTC) ─────────────────────────────────────

    /// A dead end hanging off a junction, with a genomic path running in and out through it.
    ///
    /// `dead_end_kmers` sets the dead end's length so a test can place it either side of the topological
    /// threshold; `dead_end_counts` against `flank_counts` sets what the coverage tier sees.
    fn dead_end_at_junction(
        dead_end_kmers: usize,
        dead_end_counts: u32,
        flank_counts: u32,
    ) -> (DbgGraph, NodeId, NodeId) {
        let mut graph = DbgGraph::new(3);
        let dead_end = graph.add_node(NodeStruct {
            counts: dead_end_counts,
            abs_ind: vec![0; dead_end_kmers],
            innerdir: None,
        });
        // Single-k-mer by construction, as every junction is once the graph is shrunk.
        let junction = graph.add_node(NodeStruct {
            counts: flank_counts,
            abs_ind: vec![0u64],
            innerdir: None,
        });
        let upstream = graph.add_node(NodeStruct {
            counts: flank_counts,
            abs_ind: vec![0; 40],
            innerdir: None,
        });
        let downstream = graph.add_node(NodeStruct {
            counts: flank_counts,
            abs_ind: vec![0; 40],
            innerdir: None,
        });

        graph.add_bi_edge(upstream, junction, EdgeType::MinToMin);
        graph.add_bi_edge(dead_end, junction, EdgeType::MinToMin);
        graph.add_bi_edge(junction, downstream, EdgeType::MinToMin);
        (graph, dead_end, junction)
    }

    /// Walks from `dead_end` with the tier-one threshold at 10 k-mers and tier two at 100.
    fn walk_dead_end(graph: &DbgGraph, dead_end: NodeId, cutoff: f64) -> (Vec<NodeId>, bool) {
        let mut out = Vec::new();
        let removed_by_coverage = check_dead_path(
            graph,
            dead_end,
            &mut out,
            10,
            100,
            cutoff,
            graph.first_outgoing_edge_type(dead_end).unwrap(),
        );
        (out, removed_by_coverage)
    }

    /// Build a dead end ending at a two-way fan-out. The outgoing branches occupy opposite carry
    /// orientations, so the helper exercises both sides of the bidirected endpoint.
    fn dead_end_at_fanout(
        edge: EdgeType,
        dead_end_kmers: usize,
        dead_end_counts: u32,
        fanout_kmers: usize,
        fanout_counts: u32,
        branch_counts: [u32; 2],
    ) -> (DbgGraph, NodeId, NodeId, [NodeId; 2]) {
        let mut graph = DbgGraph::new(3);
        let dead_end = graph.add_node(NodeStruct {
            counts: dead_end_counts,
            abs_ind: vec![0; dead_end_kmers],
            innerdir: None,
        });
        let fanout = graph.add_node(NodeStruct {
            counts: fanout_counts,
            abs_ind: vec![0; fanout_kmers],
            innerdir: None,
        });
        let branches = branch_counts.map(|counts| {
            graph.add_node(NodeStruct {
                counts,
                abs_ind: vec![0],
                innerdir: None,
            })
        });

        graph.add_bi_edge(dead_end, fanout, edge);
        let fanout_carry = edge.get_from_and_to().1;
        for (branch, branch_carry) in branches.iter().zip([CarryType::Min, CarryType::Max]) {
            graph.add_bi_edge(
                fanout,
                *branch,
                EdgeType::from_carrytypes(fanout_carry, branch_carry),
            );
        }
        (graph, dead_end, fanout, branches)
    }

    #[test]
    fn a_short_dead_end_at_a_fanout_includes_and_removes_the_endpoint() {
        let (mut graph, dead_end, fanout, _) =
            dead_end_at_fanout(EdgeType::MinToMin, 2, 2, 1, 2, [20, 20]);
        let mut removal_path = Vec::new();
        let removed_by_coverage = check_dead_path(
            &graph,
            dead_end,
            &mut removal_path,
            4,
            100,
            2.0,
            graph.first_outgoing_edge_type(dead_end).unwrap(),
        );

        assert!(
            !removed_by_coverage,
            "a short fan-out dead end is removed by length"
        );
        assert_eq!(removal_path, vec![dead_end, fanout]);
        assert_eq!(removal_path.iter().filter(|&&n| n == fanout).count(), 1);

        remove_paths(&mut graph, removal_path.drain(..));
        assert!(graph.node_weight(dead_end).is_none());
        assert!(graph.node_weight(fanout).is_none());
        assert_eq!(graph.validate(), Ok(()));
    }

    #[test]
    fn a_fanout_dead_end_at_the_short_length_limit_is_not_removed_by_length_alone() {
        let (graph, dead_end, _, _) = dead_end_at_fanout(EdgeType::MinToMin, 3, 2, 1, 2, [40, 40]);
        let mut removal_path = Vec::new();
        let removed_by_coverage = check_dead_path(
            &graph,
            dead_end,
            &mut removal_path,
            4,
            4, // disable tier two at the strict boundary
            2.0,
            graph.first_outgoing_edge_type(dead_end).unwrap(),
        );

        assert!(
            removal_path.is_empty(),
            "represented length equal to limit is retained"
        );
        assert!(!removed_by_coverage);
    }

    #[test]
    fn a_long_low_coverage_dead_end_at_a_fanout_is_removed_with_its_endpoint() {
        let (graph, dead_end, fanout, _) =
            dead_end_at_fanout(EdgeType::MinToMin, 12, 2, 1, 2, [40, 40]);
        let (removal_path, removed_by_coverage) = walk_dead_end(&graph, dead_end, 2.0);

        assert!(removed_by_coverage);
        assert_eq!(removal_path, vec![dead_end, fanout]);
    }

    #[test]
    fn fanout_dead_end_coverage_must_be_strictly_above_the_cutoff() {
        // Equal path-node coverage makes the dead end's mean exactly 10; its two successors average 20.
        let (graph, dead_end, _, _) =
            dead_end_at_fanout(EdgeType::MinToMin, 12, 10, 1, 10, [20, 20]);
        let (removal_path, removed_by_coverage) = walk_dead_end(&graph, dead_end, 2.0);
        assert!(removal_path.is_empty());
        assert!(!removed_by_coverage);

        let (graph, dead_end, _, _) =
            dead_end_at_fanout(EdgeType::MinToMin, 12, 10, 1, 10, [19, 19]);
        let (removal_path, removed_by_coverage) = walk_dead_end(&graph, dead_end, 2.0);
        assert!(removal_path.is_empty());
        assert!(!removed_by_coverage);
    }

    #[test]
    fn fanout_endpoint_counts_towards_the_coverage_length_limit() {
        // The dead end alone is below 100 k-mers, but including the one-k-mer fan-out endpoint reaches it.
        let (graph, dead_end, _, _) = dead_end_at_fanout(EdgeType::MinToMin, 99, 2, 1, 2, [40, 40]);
        let (removal_path, removed_by_coverage) = walk_dead_end(&graph, dead_end, 2.0);
        assert!(removal_path.is_empty());
        assert!(!removed_by_coverage);
    }

    #[test]
    fn fanout_endpoint_removal_works_for_all_edge_orientations() {
        for edge in [
            EdgeType::MinToMin,
            EdgeType::MinToMax,
            EdgeType::MaxToMin,
            EdgeType::MaxToMax,
        ] {
            let (mut graph, dead_end, fanout, _) = dead_end_at_fanout(edge, 2, 2, 1, 2, [20, 20]);
            let mut removal_path = Vec::new();
            let removed_by_coverage = check_dead_path(
                &graph,
                dead_end,
                &mut removal_path,
                4,
                100,
                2.0,
                graph.first_outgoing_edge_type(dead_end).unwrap(),
            );

            assert!(
                !removed_by_coverage,
                "{edge:?} short dead end should be removed by length"
            );
            assert_eq!(
                removal_path,
                vec![dead_end, fanout],
                "edge orientation {edge:?}"
            );
            remove_paths(&mut graph, removal_path.drain(..));
            assert!(
                graph.node_weight(fanout).is_none(),
                "edge orientation {edge:?}"
            );
            assert_eq!(graph.validate(), Ok(()), "edge orientation {edge:?}");
        }
    }

    /// A dead end that itself fans out: `dead_end_at_fanout` with its leading dead end removed.
    fn fanning_dead_end(kmers: usize, counts: u32, branch_counts: [u32; 2]) -> (DbgGraph, NodeId) {
        let (mut graph, dead_end, fanout, _) =
            dead_end_at_fanout(EdgeType::MinToMin, 1, 1, kmers, counts, branch_counts);
        graph.remove_node(dead_end);
        (graph, fanout)
    }

    #[test]
    fn a_short_dead_end_that_fans_out_is_removed_by_length() {
        let (graph, dead_end) = fanning_dead_end(2, 999, [1, 1]);
        assert_eq!(
            walk_dead_end(&graph, dead_end, 2.0),
            (vec![dead_end], false)
        );
    }

    #[test]
    fn a_long_dead_end_that_fans_out_goes_only_when_its_successors_outcover_it() {
        let (graph, dead_end) = fanning_dead_end(12, 2, [40, 40]);
        assert_eq!(walk_dead_end(&graph, dead_end, 2.0), (vec![dead_end], true));
        let (graph, dead_end) = fanning_dead_end(12, 10, [20, 20]);
        assert_eq!(walk_dead_end(&graph, dead_end, 2.0), (vec![], false));
        let (graph, dead_end) = fanning_dead_end(100, 2, [40, 40]);
        assert_eq!(walk_dead_end(&graph, dead_end, 2.0), (vec![], false));
    }

    #[test]
    fn convergence_handling_keeps_precedence_over_fanout_handling() {
        let (mut graph, dead_end, fanout, _) =
            dead_end_at_fanout(EdgeType::MinToMin, 12, 2, 1, 2, [40, 40]);
        let other_incoming = graph.add_node(NodeStruct {
            counts: 40,
            abs_ind: vec![0],
            innerdir: None,
        });
        graph.add_bi_edge(other_incoming, fanout, EdgeType::MinToMin);

        let mut removal_path = Vec::new();
        let removed_by_coverage = check_dead_path(
            &graph,
            dead_end,
            &mut removal_path,
            10,
            100,
            2.0,
            graph.first_outgoing_edge_type(dead_end).unwrap(),
        );

        assert!(removed_by_coverage);
        assert_eq!(removal_path, vec![dead_end]);
        assert!(!removal_path.contains(&fanout));
    }

    /// The point of the tier: a dead end too long for the length rule still goes when the junction it
    /// hangs off is far better covered. Taken from Minia!
    #[test]
    fn a_long_dead_end_goes_when_its_junction_is_far_better_covered() {
        let (graph, dead_end, _) = dead_end_at_junction(20, 2, 40);
        let (removed, removed_by_coverage) =
            walk_dead_end(&graph, dead_end, DEFAULT_DEAD_END_COVERAGE_RATIO);
        assert_eq!(removed, vec![dead_end]);
        assert!(
            removed_by_coverage,
            "removal must be attributed to the coverage tier"
        );
    }

    /// The other half: a comparably covered neighbourhood is evidence the dead end is real sequence.
    #[test]
    fn a_long_dead_end_survives_a_comparably_covered_junction() {
        let (graph, dead_end, _) = dead_end_at_junction(20, 20, 30);
        let (removed, removed_by_coverage) =
            walk_dead_end(&graph, dead_end, DEFAULT_DEAD_END_COVERAGE_RATIO);
        assert!(removed.is_empty(), "1.5x is under the 2x cutoff");
        assert!(!removed_by_coverage);
    }

    /// Strict `>`, matching every other coverage rule in the tree: a neighbourhood sitting exactly on
    /// `cutoff * dead end mean` is not *more* than it, so the dead end stays.
    #[test]
    fn the_coverage_ratio_is_strict_at_the_boundary() {
        let (graph, dead_end, _) = dead_end_at_junction(20, 10, 20);
        let (removed, _) = walk_dead_end(&graph, dead_end, 2.0);
        assert!(removed.is_empty(), "20 is not > 2.0 * 10");
    }

    /// Below the topological threshold the coverage tier is never consulted, so the change is
    /// strictly additive: a short dead end goes on length alone however poor the neighbourhood.
    #[test]
    fn a_short_dead_end_ignores_coverage_entirely() {
        let (graph, dead_end, _) = dead_end_at_junction(2, 999, 1);
        let (removed, removed_by_coverage) =
            walk_dead_end(&graph, dead_end, DEFAULT_DEAD_END_COVERAGE_RATIO);
        assert_eq!(removed, vec![dead_end]);
        assert!(
            !removed_by_coverage,
            "tier one decided this, not the coverage rule"
        );
    }

    /// An empty band must reproduce the length rule exactly — this is what `--no-dead-end-coverage` and
    /// `--dead-end-coverage-kmult 0` rely on.
    #[test]
    fn an_empty_coverage_band_reproduces_the_length_rule() {
        let (graph, dead_end, _) = dead_end_at_junction(20, 2, 40);
        let mut out = Vec::new();
        let removed_by_coverage = check_dead_path(
            &graph,
            dead_end,
            &mut out,
            10,
            10, // `coverage_limit.max(limit)` collapses the band to nothing
            DEFAULT_DEAD_END_COVERAGE_RATIO,
            graph.first_outgoing_edge_type(dead_end).unwrap(),
        );
        assert!(
            out.is_empty(),
            "over the length threshold and the coverage tier is off, so nothing is removed"
        );
        assert!(!removed_by_coverage);
    }

    /// A whole isolated component has no junction to be compared against, so it stays tier one's
    /// business whatever the band is set to.
    #[test]
    fn an_isolated_component_is_never_judged_on_coverage() {
        let mut graph = DbgGraph::new(3);
        let head = graph.add_node(NodeStruct {
            counts: 1,
            abs_ind: vec![0; 20],
            innerdir: None,
        });
        let tail = graph.add_node(NodeStruct {
            counts: 1,
            abs_ind: vec![0; 20],
            innerdir: None,
        });
        graph.add_bi_edge(head, tail, EdgeType::MinToMin);

        let (removed, removed_by_coverage) =
            walk_dead_end(&graph, head, DEFAULT_DEAD_END_COVERAGE_RATIO);
        assert!(
            removed.is_empty(),
            "40 k-mers is over the tier-one limit of 10"
        );
        assert!(!removed_by_coverage);
    }

    // ── dead_end_length_nts ───────────────────────────────────────────────────────

    /// The dead-end threshold is the larger of the flat floor and the k multiple; a multiple of zero or below
    /// keeps the flat floor.
    #[test]
    fn dead_end_length_nts_cases() {
        // (flat bases, k multiple, k, threshold)
        for (flat, kmult, k, want) in [
            (100, 0.0, 81, 100),
            (100, -1.0, 81, 100),
            (100, 2.5, 31, 100),
            (100, 2.5, 81, 203),
            (100, 10.0, 81, 810),
        ] {
            assert_eq!(
                dead_end_length_nts(flat, kmult, k),
                want,
                "flat {flat}, kmult {kmult}, k {k}"
            );
        }
    }

    // ── prune_unpaired_edges ─────────────────────────────────────────────────

    #[test]
    fn prune_unpaired_edges_removes_exactly_the_phantom() {
        let mut g = DbgGraph::new(3);
        let a = g.add_node(make_node());
        let b = g.add_node(make_node());
        let c = g.add_node(make_node());
        g.add_bi_edge(a, b, EdgeType::MinToMin);
        g.add_edge(c, b, EdgeType::MinToMin); // phantom: no reverse partner

        assert_eq!(prune_unpaired_edges(&mut g, b), 1);
        assert_eq!(g.edge_count(), 2); // the paired a<->b couple is intact
        assert_eq!(prune_unpaired_edges(&mut g, b), 0);
    }

    #[test]
    fn prune_unpaired_edges_removes_balanced_unpaired_edges() {
        let mut g = DbgGraph::new(3);
        let left = g.add_node(make_node());
        let node = g.add_node(make_node());
        let right = g.add_node(make_node());
        let phantom_in = g.add_node(make_node());
        let phantom_out = g.add_node(make_node());

        g.add_bi_edge(left, node, EdgeType::MinToMin);
        g.add_bi_edge(node, right, EdgeType::MinToMin);

        // One unpaired incoming and one unpaired outgoing edge keep the
        // aggregate in-degree and out-degree equal.
        g.add_edge(phantom_in, node, EdgeType::MinToMin);
        g.add_edge(node, phantom_out, EdgeType::MinToMin);

        assert_eq!(prune_unpaired_edges(&mut g, node), 2);
        assert!(g.edges_between(phantom_in, node).is_empty());
        assert!(g.edges_between(node, phantom_out).is_empty());
    }

    #[test]
    fn prune_unpaired_edges_removes_excess_parallel_edge() {
        let mut g = DbgGraph::new(3);
        let a = g.add_node(make_node());
        let b = g.add_node(make_node());

        g.add_bi_edge(a, b, EdgeType::MinToMin);
        g.add_edge(a, b, EdgeType::MinToMin);

        assert_eq!(prune_unpaired_edges(&mut g, a), 1);
        assert_eq!(g.edge_count(), 2);
        assert_eq!(g.validate(), Ok(()));
    }

    #[test]
    fn prune_unpaired_edges_removes_balanced_duplicate_pair() {
        let mut g = DbgGraph::new(3);
        let a = g.add_node(make_node());
        let b = g.add_node(make_node());

        g.add_bi_edge(a, b, EdgeType::MinToMin);
        g.add_bi_edge(a, b, EdgeType::MinToMin);

        assert_eq!(prune_unpaired_edges(&mut g, a), 2);
        assert_eq!(g.edge_count(), 2);
        assert_eq!(g.validate(), Ok(()));
    }

    #[test]
    fn prune_unpaired_edges_removes_all_duplicate_incoming_phantoms() {
        let mut g = DbgGraph::new(3);
        let source = g.add_node(make_node());
        let node = g.add_node(make_node());

        g.add_edge(source, node, EdgeType::MinToMin);
        g.add_edge(source, node, EdgeType::MinToMin);

        assert_eq!(prune_unpaired_edges(&mut g, node), 2);
        assert_eq!(g.edge_count(), 0);
        assert_eq!(prune_unpaired_edges(&mut g, node), 0);
    }

    // ---- CoverageRef: the absolute rule, and how it degrades ----

    /// The error ceiling is a fraction of the peak (0.25 fitted; 0.15 for a fallback median, which repeats
    /// bias upward), floored at min_count + 1 so it never falls under a cutoff every node already clears.
    #[test]
    fn error_ceiling_cases() {
        // (peak, min_count, ceiling)
        for (peak, min_count, want) in [
            (PeakSource::Fitted(60), 20, Some(21)),
            (PeakSource::Fitted(60), 5, Some(15)),
            (PeakSource::Fitted(100), 2, Some(25)),
            (PeakSource::Fallback(100), 2, Some(15)),
        ] {
            assert_eq!(
                CoverageRef::new(peak, min_count).error_ceiling(),
                want,
                "{peak:?}, min_count {min_count}"
            );
        }
        assert_eq!(CoverageRef::unknown().error_ceiling(), None);
    }
}
