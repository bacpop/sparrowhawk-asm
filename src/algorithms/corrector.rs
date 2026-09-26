//! Corrects parts of the provided graph, if needed
use crate::logw;
use crate::preprocessing::PeakSource;
use sparrowhawk_graph::{
    BubbleStartEdge, CarryType, DbgGraph, EdgeId, EdgeType, NodeId, NodeStruct,
};

use crate::EdgeWeight;

use std::{
    collections::{BTreeMap, BTreeSet},
    vec::Drain,
};

/// Minimum number of k-mers a path must exceed to be worth keeping in dead-end removal.
pub(crate) fn short_path_limit(minnts: usize, k: usize) -> usize {
    (minnts + 1).saturating_sub(k) // sat_sub is compulsory, becase as these are usize, going negative my change the path to an absurd value!!!
}

/// Tip-removal threshold in bases: the larger of a flat floor and a multiple of k, as Minia sizes it
/// (`-tip-len-topo-kmult`). A flat 100 nt is generous at k=31 and catches almost nothing at k=81.
pub(crate) fn tip_length_nts(flat_nts: usize, kmult: f32, k: usize) -> usize {
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
    pub tip_nts: usize,
    /// Upper bound of the coverage-judged tip band, already resolved against k. Zero disables it.
    pub tip_rctc_nts: usize,
    /// How many times better covered a junction must be than the tip hanging off it.
    pub tip_rctc_cutoff: f64,
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

    /// Solve bubbles from the graph, popping only where the coverage difference is big.
    fn correct_bubbles(&mut self, pop_ratio: f32, coverage: &CoverageRef) -> bool;

    /// Remove dead paths shorter than `tip_nts` bases, and those up to `tip_rctc_nts` whose junction
    /// is `rctc_cutoff` times better covered.
    fn remove_dead_paths(&mut self, tip_nts: usize, tip_rctc_nts: usize, rctc_cutoff: f64) -> bool;
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

    fn correct_bubbles(&mut self, pop_ratio: f32, coverage: &CoverageRef) -> bool {
        // `path_correction` is native-only, so the browser keeps the diamond popper it used before
        // bulges landed.
        #[cfg(not(target_family = "wasm"))]
        {
            crate::algorithms::path_correction::remove_bulges(self, pop_ratio, coverage)
        }
        #[cfg(target_family = "wasm")]
        {
            pop_bubbles_by_coverage(self, pop_ratio, coverage)
        }
    }

    fn remove_dead_paths(&mut self, tip_nts: usize, tip_rctc_nts: usize, rctc_cutoff: f64) -> bool {
        logw(
            format!(
                "Before pruning: {} nodes and {} edges",
                self.node_count(),
                self.edge_count()
            )
            .as_str(),
            Some("info"),
        );

        // Hoisted out of the walk: it only depends on k and the threshold, both fixed for this call.
        let limit = short_path_limit(tip_nts, self.k());
        // An empty band disables tier two outright: the walk bound falls back to `limit` and not one
        // comparison is made, so `--tip-length-rctc-kmult 0` reproduces today's output exactly.
        let rctc_limit = short_path_limit(tip_rctc_nts, self.k()).max(limit);
        let (mut topo_tips, mut rctc_tips) = (0usize, 0usize);
        let mut dididoanything = false;
        logw(format!("Starting graph pruning. Graph has {} externals, {} alone nodes, the remaining are internal.",
            self.externals_bi().len(),
            self.node_indices().filter(|n| self.out_degree(*n) == 0 && self.in_degree(*n) == 0).count()).as_str(), Some("info"));

        let mut to_remove: Vec<NodeId> = vec![];
        loop {
            let mut path_check_vec = vec![];
            let externals: Vec<_> = self
                .externals_bi()
                .into_iter()
                .filter(|n| self.out_degree(*n) == 1)
                .collect();

            logw(
                format!("Detected {} externals", externals.len()).as_str(),
                Some("trace"),
            );

            for v in externals {
                let carryedge = self.first_outgoing_edge_type(v).unwrap();
                let by_rctc = check_dead_path(
                    self,
                    v,
                    &mut path_check_vec,
                    limit,
                    rctc_limit,
                    rctc_cutoff,
                    carryedge,
                );
                if !path_check_vec.is_empty() {
                    dididoanything = true;
                    if by_rctc {
                        rctc_tips += 1;
                    } else {
                        topo_tips += 1;
                    }
                    to_remove.append(&mut path_check_vec);
                }
            }

            // if there are no dead paths left pruning is done
            if to_remove.is_empty() {
                logw(
                    format!(
                        "Graph is pruned: {topo_tips} tips by length (threshold {tip_nts} nt, \
                         {limit} k-mers), {rctc_tips} by coverage (up to {tip_rctc_nts} nt, \
                         {rctc_limit} k-mers, cutoff {rctc_cutoff})"
                    )
                    .as_str(),
                    Some("info"),
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

    let mut incident: BTreeMap<Connection, BTreeSet<EdgeId>> = BTreeMap::new();

    for carry in [CarryType::Min, CarryType::Max] {
        for (edge_id, target, edge_type) in g.outgoing_edges_by_carry(n, carry) {
            incident
                .entry((n, target, edge_type))
                .or_default()
                .insert(edge_id);
        }
    }

    for (source, edge_type) in g.incoming_edges(n) {
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

/// Collapse every bubble whose two branches differ significantly enough in coverage; leave the rest alone.
pub fn pop_bubbles_by_coverage(g: &mut DbgGraph, pop_ratio: f32, coverage: &CoverageRef) -> bool {
    let mut dididoanything = false;

    logw("Starting resolution of standard bubbles", Some("info"));

    let bubbles = g
        .node_indices()
        .filter(|n| {
            // Two or more forward branches, any number of predecessors, and the join may keep its
            // own successors. `out_degree >= 2` is a necessary condition for two Min-carry branches,
            // so it excludes nothing and spares the edge walk on most nodes.
            g.out_degree(*n) >= 2
                && bubble_shape_ok(
                    g,
                    *n,
                    &g.bubble_start_edges_by_carry(*n, CarryType::Min),
                    true,
                )
        })
        .collect::<BTreeSet<NodeId>>();

    let examined = bubbles.len();
    let mut collapsed = 0usize;

    if bubbles.is_empty() {
        return false;
    } else {
        logw(
            format!("Found {examined:?} potential bubbles (they might be less). Starting to collapse them ")
                .as_str(),
            Some("info"),
        );
        for n in bubbles {
            if g.contains_node(n) {
                let tmpb = collapse_bubble(g, n, pop_ratio, coverage);
                if tmpb {
                    collapsed += 1;
                    dididoanything = true;
                }
            }
        }
    }

    // Examined vs collapsed, because the two diverge badly: the ratio rule alone leaves nearly every
    // candidate standing, and a candidate count on its own hides that.
    logw(
        format!(
            "Bubble correction ended: {collapsed}/{examined} candidates collapsed (ceiling {:?}). \
             Corrected graph has {} nodes and {} edges",
            coverage.error_ceiling(),
            g.node_count(),
            g.edge_count()
        )
        .as_str(),
        Some("info"),
    );

    dididoanything
}

/// The bubble shape test, generalised to N branches. `allow_multi_succ` lifts the requirement that
/// the join have exactly one successor, which is the stage-1.5 relaxation.
fn bubble_shape_ok(
    ptgraph: &DbgGraph,
    startn: NodeId,
    invec: &[BubbleStartEdge],
    allow_multi_succ: bool,
) -> bool {
    if invec.len() < 2 {
        return false;
    }
    let midnodes: Vec<NodeId> = invec.iter().map(|e| e.target).collect();
    let midcts: Vec<CarryType> = invec
        .iter()
        .map(|e| e.edge_type.get_from_and_to().1)
        .collect();

    // Every branch its own node, none of them the entrance.
    let uniq: BTreeSet<NodeId> = midnodes.iter().copied().collect();
    if uniq.len() != midnodes.len() || uniq.contains(&startn) {
        return false;
    }

    // Branch 0 fixes the join and the carry the others must agree on.
    let tmpv0 = ptgraph.out_neighbours_bi(midnodes[0], midcts[0]);
    if tmpv0.len() != 1 || ptgraph.in_neighbours_bi(midnodes[0], midcts[0]).len() != 1 {
        return false;
    }
    let outnode = tmpv0[0].0;
    let outct = tmpv0[0].1.get_from_and_to().1;
    if outnode == startn {
        return false;
    }

    for (mid, ct) in midnodes.iter().zip(&midcts).skip(1) {
        let tmpv = ptgraph.out_neighbours_bi(*mid, *ct);
        if tmpv.len() != 1
            || tmpv[0].0 != outnode
            || tmpv[0].1.get_from_and_to().1 != outct
            || ptgraph.in_neighbours_bi(*mid, *ct).len() != 1
        {
            return false;
        }
    }

    // The join must take our branches and nothing else, and none of its successors may lead back
    // into the bubble.
    let outs = ptgraph.out_neighbours_bi(outnode, outct);
    if outs.is_empty() || (!allow_multi_succ && outs.len() != 1) {
        return false;
    }
    outs.iter().all(|s| s.0 != startn && !uniq.contains(&s.0))
        && ptgraph.in_neighbours_bi(outnode, outct).len() == midnodes.len()
}

/// What the coverage heuristic decided to do with a bubble.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BubbleChoice {
    /// Collapse the bubble onto this branch, discarding the others.
    Keep(usize),
    /// Delete only these branches; the rest of the bubble stands for the next round.
    Prune(Vec<usize>),
    /// Do nothing at all. The bubble stays exactly as it is, contig break and all.
    Leave,
}

/// The coverage heuristic. It inspects the graph and decides, but changes nothing.
fn choose_branch_by_counts(
    ptgraph: &DbgGraph,
    midconns: &[(NodeId, EdgeType)],
    pop_ratio: f32,
    coverage: &CoverageRef,
) -> BubbleChoice {
    let counts: Vec<u32> = midconns
        .iter()
        .map(|c| ptgraph.node_weight(c.0).unwrap().counts)
        .collect();
    // `Reverse` on the index makes the lowest-indexed branch win a tie, so the choice does not
    // depend on edge insertion order.
    let Some((winner, _)) = counts
        .iter()
        .enumerate()
        .max_by_key(|(i, c)| (**c, std::cmp::Reverse(*i)))
    else {
        return BubbleChoice::Leave;
    };
    let hi = counts[winner];

    // The ratio test is nearly unsatisfiable on its own: every node already clears `min_count`, which
    // measures 0.16-0.30 of the peak, so the winner would have to stand at 1.6-3.0x single-copy
    // coverage. The absolute test asks what the ratio cannot — is the weak branch too far below
    // single-copy coverage to be a copy, while the strong one is one?
    let doomed: Vec<usize> = counts
        .iter()
        .enumerate()
        .filter(|(i, &lo)| {
            // Strict `<`, so equal coverages can never pop however the ratio is set.
            *i != winner
                && ((lo as f32) < pop_ratio * hi as f32
                    || (coverage.is_error_like(lo) && coverage.is_genomic(hi)))
        })
        .map(|(i, _)| i)
        .collect();

    if doomed.is_empty() {
        BubbleChoice::Leave
    } else if doomed.len() == counts.len() - 1 {
        BubbleChoice::Keep(winner)
    } else {
        // Some branches qualify and some do not. Drop those that do and leave the rest standing: the
        // correction loop runs to a fixed point, so next round this is a narrower bubble and goes
        // through the ordinary path. Partial deletion needs no rewiring.
        BubbleChoice::Prune(doomed)
    }
}

/// Get the final graph with the chosen branch (if so) in it..
pub fn apply_bubble_collapse(
    ptgraph: &mut DbgGraph,
    startn: NodeId,
    midconns: &[(NodeId, EdgeType)],
    winner: usize,
) -> bool {
    let savedmidw = ptgraph.node_weight(midconns[winner].0).unwrap().clone();

    let midnodect = midconns[winner].1.get_from_and_to().1;
    // Re-validate: earlier collapses in this pass (or a collision) may have changed the shape.
    let midouts = ptgraph.out_neighbours_bi(midconns[winner].0, midnodect);
    if midouts.len() != 1 {
        log::debug!(
            "Bubble at {:?} no longer matches its detected shape; skipping",
            startn
        );
        return false;
    }
    let midconn2 = midouts[0];
    let outct = midconn2.1.get_from_and_to().1;
    let endouts = ptgraph.out_neighbours_bi(midconn2.0, outct);
    if endouts.is_empty() {
        log::debug!(
            "Bubble at {:?} no longer matches its detected shape; skipping",
            startn
        );
        return false;
    }
    let savedoutw = ptgraph.node_weight(midconn2.0).unwrap().clone();

    if ptgraph.node_weight(startn).unwrap().abs_ind.len() > 1
        || ptgraph.node_weight(midconn2.0).unwrap().abs_ind.len() > 1
    {
        panic!("Trying to remove a bubble with a start or end node with more than one k-mer!");
    };

    for mc in midconns {
        ptgraph.remove_node(mc.0); // every branch, winner included: it is folded into `startn` below
    }
    ptgraph.remove_node(midconn2.0); // end node

    let mutrefw = ptgraph.node_weight_mut(startn).unwrap();

    mutrefw.merge(&savedmidw, midconns[winner].1);
    mutrefw.set_internal_edge(EdgeType::MinToMin);
    mutrefw.abs_ind.push(savedoutw.abs_ind[0]);
    // The guard above pins the start and end nodes at one k-mer each, so only the winning branch
    // can be a longer unitig. `merge` extends `abs_ind` but leaves `counts`, so this is still the
    // start node's own count.
    mutrefw.set_mean_counts(&[
        (mutrefw.counts, 1),
        (savedmidw.counts, savedmidw.abs_ind.len()),
        (savedoutw.counts, 1),
    ]);

    // The join's successors move onto the entrance. `bubble_shape_ok` has already excluded any that
    // point back into the bubble, so none of these targets has just been removed.
    for outconn in endouts {
        match outct {
            CarryType::Min => {
                ptgraph.add_bi_edge(startn, outconn.0, outconn.1);
            }
            CarryType::Max => {
                let tmptype =
                    EdgeType::from_carrytypes(CarryType::Min, outconn.1.get_from_and_to().1);
                ptgraph.add_bi_edge(startn, outconn.0, tmptype);
            }
        }
    }
    true
}

/// Collapse one standard bubble, if the coverage difference between its branches is big enough.
fn collapse_bubble(
    ptgraph: &mut DbgGraph,
    startn: NodeId,
    pop_ratio: f32,
    coverage: &CoverageRef,
) -> bool {
    let midconns = ptgraph.out_neighbours_min(startn);
    if midconns.len() < 2
        || ptgraph
            .out_neighbours_bi(midconns[0].0, midconns[0].1.get_from_and_to().1)
            .len()
            != 1
    {
        return false;
    }

    match choose_branch_by_counts(ptgraph, &midconns, pop_ratio, coverage) {
        BubbleChoice::Keep(w) => apply_bubble_collapse(ptgraph, startn, &midconns, w),
        BubbleChoice::Prune(losers) => {
            for i in losers {
                ptgraph.remove_node(midconns[i].0);
            }
            true
        }
        BubbleChoice::Leave => false,
    }
}

/// Remove dead input path.
#[inline]
fn remove_paths(ptgraph: &mut DbgGraph, to_remove: Drain<NodeId>) {
    log::trace!("Removing {} dead paths", to_remove.len());
    for n in to_remove {
        ptgraph.remove_node(n);
    }
}

/// Minia's relative-coverage tip condition (`Simplifications.cpp:319`, `satisfyRCTC`): a tip too long
/// to drop on length alone still goes when the junction it hangs off is much better covered.
fn neighbourhood_outcovers_tip(
    ptgraph: &DbgGraph,
    junction: NodeId,
    junction_ty: CarryType,
    tip_last: NodeId,
    tip_mean: f64,
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
        if nb == tip_last || !seen.insert(nb) {
            continue;
        }
        total += f64::from(ptgraph.node_weight(nb).unwrap().counts);
    }
    if seen.is_empty() {
        return false; // nothing to compare against: an isolated tip is tier one's business
    }
    // Strict `>`, and `f64` throughout — the cutoff is declared `f64` from the CLI down, as
    // `--ec-coverage-ratio` is, so there is no `f32` hop to shift the boundary the way
    // `f64::from(0.1f32)` did in `should_delete`.
    total / seen.len() as f64 > cutoff * tip_mean
}

/// Check if vertex initializes a dead path.
#[inline]
fn check_dead_path(
    ptgraph: &DbgGraph,
    vertex: NodeId,
    output_vec: &mut Vec<NodeId>,
    limit: usize,
    rctc_limit: usize,
    rctc_cutoff: f64,
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

    loop {
        if cnt >= rctc_limit {
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
            if neighbourhood_outcovers_tip(
                ptgraph,
                candidate_node,
                candidate_ty,
                *output_vec.last().unwrap(),
                path_mean_coverage(ptgraph, output_vec),
                rctc_cutoff,
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

            // Keep the existing strict short-tip boundary and remove by topology alone below it.
            if cnt < limit {
                return false;
            }

            // Longer tips retain the existing relative-coverage rule, now measuring the complete
            // path through its fan-out endpoint. I had this wrong in the past, partially intentional, but I was wrong because I was leaving lots of bad tips here.
            if cnt < rctc_limit
                && neighbourhood_outcovers_tip(
                    ptgraph,
                    candidate_node,
                    candidate_ty,
                    before,
                    path_mean_coverage(ptgraph, output_vec),
                    rctc_cutoff,
                )
            {
                return true;
            }
            output_vec.clear();
            return false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::DEFAULT_TIP_RCTC_CUTOFF;
    use sparrowhawk_graph::{DbgGraph, NodeStruct};
    use std::cmp::{max, min};

    fn make_node() -> NodeStruct {
        NodeStruct {
            counts: 10,
            abs_ind: vec![0u64],
            innerdir: None,
        }
    }

    // ── dead-path limit formula ──────────────────────────────────────────────

    // These used to re-implement the formula inline instead of calling it, which is precisely why an
    // integer underflow above k = 100 survived: every case they covered was under the boundary, and a
    // copy of the code cannot disagree with itself.

    #[test]
    fn dead_path_limit_k3() {
        assert_eq!(short_path_limit(100, 3), 98);
    }

    #[test]
    fn dead_path_limit_k31() {
        assert_eq!(short_path_limit(100, 31), 70);
    }

    #[test]
    fn dead_path_limit_k100() {
        assert_eq!(short_path_limit(100, 100), 1);
    }

    /// The first k that underflowed. `100 - 101 + 1` on `usize` wraps to ~1.8e19, after which every
    /// contig is dropped and every path is judged dead — a k > 100 assembly came out entirely empty.
    #[test]
    fn dead_path_limit_does_not_underflow_above_minnts() {
        assert_eq!(short_path_limit(100, 101), 0);
        assert_eq!(short_path_limit(100, 139), 0);
        assert_eq!(short_path_limit(100, 255), 0);
    }

    #[test]
    fn dead_path_arbitration_compares_kmer_totals() {
        let mut graph = DbgGraph::new(3);
        let tip_a = graph.add_node(NodeStruct {
            counts: 1,
            abs_ind: vec![0; 2],
            innerdir: None,
        });
        let middle_a = graph.add_node(NodeStruct {
            counts: 1,
            abs_ind: vec![0; 2],
            innerdir: None,
        });
        let tip_b = graph.add_node(NodeStruct {
            counts: 1,
            abs_ind: vec![0; 3],
            innerdir: None,
        });
        let junction = graph.add_node(make_node());

        graph.add_bi_edge(tip_a, middle_a, EdgeType::MinToMin);
        graph.add_bi_edge(middle_a, junction, EdgeType::MinToMin);
        graph.add_bi_edge(tip_b, junction, EdgeType::MinToMin);

        let limit = 98;
        let mut from_a = Vec::new();
        check_dead_path(
            &graph,
            tip_a,
            &mut from_a,
            limit,
            limit, // tier two disabled: an empty band, as `--no-tip-rctc` produces
            DEFAULT_TIP_RCTC_CUTOFF,
            graph.first_outgoing_edge_type(tip_a).unwrap(),
        );
        assert_eq!(from_a, vec![tip_b]);

        let mut from_b = Vec::new();
        check_dead_path(
            &graph,
            tip_b,
            &mut from_b,
            limit,
            limit, // tier two disabled: an empty band, as `--no-tip-rctc` produces
            DEFAULT_TIP_RCTC_CUTOFF,
            graph.first_outgoing_edge_type(tip_b).unwrap(),
        );
        assert_eq!(from_b, vec![tip_b]);
    }

    // ── the coverage tier (Minia's RCTC) ─────────────────────────────────────

    /// A tip hanging off a junction, with a genomic path running in and out through it.
    ///
    /// `tip_kmers` sets the tip's length so a test can place it either side of the topological
    /// threshold; `tip_counts` against `flank_counts` sets what the coverage tier sees.
    fn tip_at_junction(
        tip_kmers: usize,
        tip_counts: u32,
        flank_counts: u32,
    ) -> (DbgGraph, NodeId, NodeId) {
        let mut graph = DbgGraph::new(3);
        let tip = graph.add_node(NodeStruct {
            counts: tip_counts,
            abs_ind: vec![0; tip_kmers],
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
        graph.add_bi_edge(tip, junction, EdgeType::MinToMin);
        graph.add_bi_edge(junction, downstream, EdgeType::MinToMin);
        (graph, tip, junction)
    }

    /// Walks from `tip` with the tier-one threshold at 10 k-mers and tier two at 100.
    fn walk_tip(graph: &DbgGraph, tip: NodeId, cutoff: f64) -> (Vec<NodeId>, bool) {
        let mut out = Vec::new();
        let by_rctc = check_dead_path(
            graph,
            tip,
            &mut out,
            10,
            100,
            cutoff,
            graph.first_outgoing_edge_type(tip).unwrap(),
        );
        (out, by_rctc)
    }

    /// Build a tip ending at a two-way fan-out. The outgoing branches occupy opposite carry
    /// orientations, so the helper exercises both sides of the bidirected endpoint.
    fn tip_at_fanout(
        edge: EdgeType,
        tip_kmers: usize,
        tip_counts: u32,
        fanout_kmers: usize,
        fanout_counts: u32,
        branch_counts: [u32; 2],
    ) -> (DbgGraph, NodeId, NodeId, [NodeId; 2]) {
        let mut graph = DbgGraph::new(3);
        let tip = graph.add_node(NodeStruct {
            counts: tip_counts,
            abs_ind: vec![0; tip_kmers],
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

        graph.add_bi_edge(tip, fanout, edge);
        let fanout_carry = edge.get_from_and_to().1;
        for (branch, branch_carry) in branches.iter().zip([CarryType::Min, CarryType::Max]) {
            graph.add_bi_edge(
                fanout,
                *branch,
                EdgeType::from_carrytypes(fanout_carry, branch_carry),
            );
        }
        (graph, tip, fanout, branches)
    }

    #[test]
    fn a_short_tip_at_a_fanout_includes_and_removes_the_endpoint() {
        let (mut graph, tip, fanout, _) = tip_at_fanout(EdgeType::MinToMin, 2, 2, 1, 2, [20, 20]);
        let mut removal_path = Vec::new();
        let by_rctc = check_dead_path(
            &graph,
            tip,
            &mut removal_path,
            4,
            100,
            2.0,
            graph.first_outgoing_edge_type(tip).unwrap(),
        );

        assert!(!by_rctc, "a short fan-out tip is removed by length");
        assert_eq!(removal_path, vec![tip, fanout]);
        assert_eq!(removal_path.iter().filter(|&&n| n == fanout).count(), 1);

        remove_paths(&mut graph, removal_path.drain(..));
        assert!(graph.node_weight(tip).is_none());
        assert!(graph.node_weight(fanout).is_none());
        assert_eq!(graph.validate(), Ok(()));
    }

    #[test]
    fn a_fanout_tip_at_the_short_length_limit_is_not_removed_by_length_alone() {
        let (graph, tip, _, _) = tip_at_fanout(EdgeType::MinToMin, 3, 2, 1, 2, [40, 40]);
        let mut removal_path = Vec::new();
        let by_rctc = check_dead_path(
            &graph,
            tip,
            &mut removal_path,
            4,
            4, // disable tier two at the strict boundary
            2.0,
            graph.first_outgoing_edge_type(tip).unwrap(),
        );

        assert!(
            removal_path.is_empty(),
            "represented length equal to limit is retained"
        );
        assert!(!by_rctc);
    }

    #[test]
    fn a_long_low_coverage_tip_at_a_fanout_is_removed_with_its_endpoint() {
        let (graph, tip, fanout, _) = tip_at_fanout(EdgeType::MinToMin, 12, 2, 1, 2, [40, 40]);
        let (removal_path, by_rctc) = walk_tip(&graph, tip, 2.0);

        assert!(by_rctc);
        assert_eq!(removal_path, vec![tip, fanout]);
    }

    #[test]
    fn fanout_tip_coverage_must_be_strictly_above_the_cutoff() {
        // Equal path-node coverage makes the tip mean exactly 10; its two successors average 20.
        let (graph, tip, _, _) = tip_at_fanout(EdgeType::MinToMin, 12, 10, 1, 10, [20, 20]);
        let (removal_path, by_rctc) = walk_tip(&graph, tip, 2.0);
        assert!(removal_path.is_empty());
        assert!(!by_rctc);

        let (graph, tip, _, _) = tip_at_fanout(EdgeType::MinToMin, 12, 10, 1, 10, [19, 19]);
        let (removal_path, by_rctc) = walk_tip(&graph, tip, 2.0);
        assert!(removal_path.is_empty());
        assert!(!by_rctc);
    }

    #[test]
    fn fanout_endpoint_counts_towards_the_rctc_length_limit() {
        // The tip alone is below 100 k-mers, but including the one-k-mer fan-out endpoint reaches it.
        let (graph, tip, _, _) = tip_at_fanout(EdgeType::MinToMin, 99, 2, 1, 2, [40, 40]);
        let (removal_path, by_rctc) = walk_tip(&graph, tip, 2.0);
        assert!(removal_path.is_empty());
        assert!(!by_rctc);
    }

    #[test]
    fn fanout_endpoint_removal_works_for_all_edge_orientations() {
        for edge in [
            EdgeType::MinToMin,
            EdgeType::MinToMax,
            EdgeType::MaxToMin,
            EdgeType::MaxToMax,
        ] {
            let (mut graph, tip, fanout, _) = tip_at_fanout(edge, 2, 2, 1, 2, [20, 20]);
            let mut removal_path = Vec::new();
            let by_rctc = check_dead_path(
                &graph,
                tip,
                &mut removal_path,
                4,
                100,
                2.0,
                graph.first_outgoing_edge_type(tip).unwrap(),
            );

            assert!(!by_rctc, "{edge:?} short tip should be removed by length");
            assert_eq!(removal_path, vec![tip, fanout], "edge orientation {edge:?}");
            remove_paths(&mut graph, removal_path.drain(..));
            assert!(
                graph.node_weight(fanout).is_none(),
                "edge orientation {edge:?}"
            );
            assert_eq!(graph.validate(), Ok(()), "edge orientation {edge:?}");
        }
    }

    #[test]
    fn convergence_handling_keeps_precedence_over_fanout_handling() {
        let (mut graph, tip, fanout, _) = tip_at_fanout(EdgeType::MinToMin, 12, 2, 1, 2, [40, 40]);
        let other_incoming = graph.add_node(NodeStruct {
            counts: 40,
            abs_ind: vec![0],
            innerdir: None,
        });
        graph.add_bi_edge(other_incoming, fanout, EdgeType::MinToMin);

        let mut removal_path = Vec::new();
        let by_rctc = check_dead_path(
            &graph,
            tip,
            &mut removal_path,
            10,
            100,
            2.0,
            graph.first_outgoing_edge_type(tip).unwrap(),
        );

        assert!(by_rctc);
        assert_eq!(removal_path, vec![tip]);
        assert!(!removal_path.contains(&fanout));
    }

    /// The point of the tier: a tip too long for the length rule still goes when the junction it
    /// hangs off is far better covered. Taken from Minia!
    #[test]
    fn a_long_tip_goes_when_its_junction_is_far_better_covered() {
        let (graph, tip, _) = tip_at_junction(20, 2, 40);
        let (removed, by_rctc) = walk_tip(&graph, tip, DEFAULT_TIP_RCTC_CUTOFF);
        assert_eq!(removed, vec![tip]);
        assert!(by_rctc, "removal must be attributed to the coverage tier");
    }

    /// The other half: a comparably covered neighbourhood is evidence the tip is real sequence.
    #[test]
    fn a_long_tip_survives_a_comparably_covered_junction() {
        let (graph, tip, _) = tip_at_junction(20, 20, 30);
        let (removed, by_rctc) = walk_tip(&graph, tip, DEFAULT_TIP_RCTC_CUTOFF);
        assert!(removed.is_empty(), "1.5x is under the 2x cutoff");
        assert!(!by_rctc);
    }

    /// Strict `>`, matching every other coverage rule in the tree: a neighbourhood sitting exactly on
    /// `cutoff * tip` is not *more* than it, so the tip stays.
    #[test]
    fn the_rctc_cutoff_is_strict_at_the_boundary() {
        let (graph, tip, _) = tip_at_junction(20, 10, 20);
        let (removed, _) = walk_tip(&graph, tip, 2.0);
        assert!(removed.is_empty(), "20 is not > 2.0 * 10");
    }

    /// Below the topological threshold the coverage tier is never consulted, so the change is
    /// strictly additive: a short tip goes on length alone however poor the neighbourhood.
    #[test]
    fn a_short_tip_ignores_coverage_entirely() {
        let (graph, tip, _) = tip_at_junction(2, 999, 1);
        let (removed, by_rctc) = walk_tip(&graph, tip, DEFAULT_TIP_RCTC_CUTOFF);
        assert_eq!(removed, vec![tip]);
        assert!(!by_rctc, "tier one decided this, not the coverage rule");
    }

    /// An empty band must reproduce the length rule exactly — this is what `--no-tip-rctc` and
    /// `--tip-length-rctc-kmult 0` rely on.
    #[test]
    fn an_empty_rctc_band_reproduces_the_length_rule() {
        let (graph, tip, _) = tip_at_junction(20, 2, 40);
        let mut out = Vec::new();
        let by_rctc = check_dead_path(
            &graph,
            tip,
            &mut out,
            10,
            10, // `rctc_limit.max(limit)` collapses the band to nothing
            DEFAULT_TIP_RCTC_CUTOFF,
            graph.first_outgoing_edge_type(tip).unwrap(),
        );
        assert!(
            out.is_empty(),
            "over the length threshold and the coverage tier is off, so nothing is removed"
        );
        assert!(!by_rctc);
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

        let (removed, by_rctc) = walk_tip(&graph, head, DEFAULT_TIP_RCTC_CUTOFF);
        assert!(
            removed.is_empty(),
            "40 k-mers is over the tier-one limit of 10"
        );
        assert!(!by_rctc);
    }

    // ── tip_length_nts ───────────────────────────────────────────────────────

    /// Zero keeps the flat floor, which is how the k multiplier is switched off.
    #[test]
    fn tip_length_nts_zero_kmult_keeps_the_flat_floor() {
        assert_eq!(tip_length_nts(100, 0.0, 81), 100);
        assert_eq!(tip_length_nts(100, -1.0, 81), 100);
    }

    /// The multiple wins once it clears the floor, and the floor wins while it does not.
    #[test]
    fn tip_length_nts_takes_the_larger_of_floor_and_multiple() {
        assert_eq!(tip_length_nts(100, 2.5, 31), 100); // 77.5 rounds under the floor
        assert_eq!(tip_length_nts(100, 2.5, 81), 203); // 202.5 rounds up
        assert_eq!(tip_length_nts(100, 10.0, 81), 810);
    }

    /// A single k-mer already spans k bases, so above the threshold nothing may be filtered for length.
    #[test]
    fn a_single_kmer_contig_survives_when_k_exceeds_the_floor() {
        for k in [101usize, 139, 255] {
            assert!(
                1 > short_path_limit(100, k),
                "one k-mer spans {k} bases, which is over the 100 nt floor"
            );
        }
    }

    // ── bubble_shape_ok ───────────────────────────────────────────────

    /// Build a valid 5-node diamond: S → M1 → E → F
    ///                                S → M2 → E
    fn make_valid_bubble() -> (DbgGraph, NodeId, Vec<BubbleStartEdge>) {
        let mut g = DbgGraph::new(3);
        let s = g.add_node(make_node());
        let m1 = g.add_node(make_node());
        let m2 = g.add_node(make_node());
        let e = g.add_node(make_node());
        let f = g.add_node(make_node());
        g.add_bi_edge(s, m1, EdgeType::MinToMin);
        g.add_bi_edge(s, m2, EdgeType::MinToMin);
        g.add_bi_edge(m1, e, EdgeType::MinToMin);
        g.add_bi_edge(m2, e, EdgeType::MinToMin);
        g.add_bi_edge(e, f, EdgeType::MinToMin);
        let edges = g.bubble_start_edges_by_carry(s, CarryType::Min);
        (g, s, edges)
    }

    // ── the pop rule ─────────────────────────────────────────────────────────

    /// The same diamond, with the two branches given explicit coverages and an **upstream flank**.
    ///
    /// The flank matters: `pop_bubbles_by_coverage` filters candidates on `out_degree(start) == 3`,
    /// which decomposes as the two outgoing `Min` branches plus the one outgoing `Max` back-link that
    /// `add_bi_edge` installs for the incoming flank edge. Without `f0 -> s` the degree is 2 and the
    /// bubble is never even offered to the heuristic.
    fn bubble_with_counts(c0: u32, c1: u32) -> (DbgGraph, NodeId, Vec<(NodeId, EdgeType)>) {
        let mut g = DbgGraph::new(3);
        let f0 = g.add_node(make_node());
        let s = g.add_node(make_node());
        let m1 = g.add_node(NodeStruct {
            counts: c0,
            ..make_node()
        });
        let m2 = g.add_node(NodeStruct {
            counts: c1,
            ..make_node()
        });
        let e = g.add_node(make_node());
        let f = g.add_node(make_node());
        g.add_bi_edge(f0, s, EdgeType::MinToMin);
        g.add_bi_edge(s, m1, EdgeType::MinToMin);
        g.add_bi_edge(s, m2, EdgeType::MinToMin);
        g.add_bi_edge(m1, e, EdgeType::MinToMin);
        g.add_bi_edge(m2, e, EdgeType::MinToMin);
        g.add_bi_edge(e, f, EdgeType::MinToMin);
        assert_eq!(
            g.out_degree(s),
            3,
            "fixture must be a candidate for the popper"
        );
        let mids = g.out_neighbours_min(s);
        (g, s, mids)
    }

    /// Index in `mids` of the branch with the higher count.
    ///
    /// `out_neighbours_min` does not promise insertion order, so a test that assumed `mids[0]` is the
    /// first node it added would be asserting on an artefact of `petgraph`'s edge storage rather than
    /// on the rule. Ask the graph instead.
    fn stronger(g: &DbgGraph, mids: &[(NodeId, EdgeType)]) -> usize {
        let c0 = g.node_weight(mids[0].0).unwrap().counts;
        let c1 = g.node_weight(mids[1].0).unwrap().counts;
        if c0 >= c1 {
            0
        } else {
            1
        }
    }

    /// **The regression this whole rule exists to prevent.** A collapsed two-copy repeat has branches
    /// of equal length and equal coverage; the old heuristic fell through to an arbitrary `Keep(1)` and
    /// deleted one real copy. Nothing may be touched here, however the ratio is set.
    #[test]
    fn equal_coverage_bubble_is_left_alone() {
        let (g, _s, mids) = bubble_with_counts(40, 40);
        for ratio in [0.01_f32, 0.1, 0.5, 0.99] {
            assert_eq!(
                choose_branch_by_counts(&g, &mids, ratio, &CoverageRef::unknown()),
                BubbleChoice::Leave,
                "equal coverage must never pop, ratio {ratio}"
            );
        }

        let (mut g, _s, _) = bubble_with_counts(40, 40);
        let before = g.node_count();
        assert!(!pop_bubbles_by_coverage(
            &mut g,
            DEFAULT_POP_RATIO,
            &CoverageRef::unknown()
        ));
        assert_eq!(g.node_count(), before, "the graph must be untouched");
    }

    /// A branch carrying a twentieth of the other is noise, and is popped — whichever way round the
    /// two are stored.
    #[test]
    fn a_noise_branch_is_popped() {
        for (c0, c1) in [(100u32, 5u32), (5, 100)] {
            let (g, _s, mids) = bubble_with_counts(c0, c1);
            assert_eq!(
                choose_branch_by_counts(&g, &mids, 0.1, &CoverageRef::unknown()),
                BubbleChoice::Keep(stronger(&g, &mids)),
                "must keep the stronger branch for ({c0}, {c1})"
            );
        }

        let (mut g, _s, _) = bubble_with_counts(100, 5);
        let before = g.node_count();
        assert!(pop_bubbles_by_coverage(
            &mut g,
            DEFAULT_POP_RATIO,
            &CoverageRef::unknown()
        ));
        assert!(
            g.node_count() < before,
            "the popped branch and end node are gone"
        );
    }

    /// Pins the boundary: the comparison is a strict `<`, so a branch sitting exactly on the ratio
    /// survives. Getting this backwards would pop bubbles at exactly 10 %, which is the side of the
    /// line where two real things start to look alike.
    #[test]
    fn a_branch_exactly_on_the_ratio_survives() {
        let (g, _s, mids) = bubble_with_counts(100, 10);
        assert_eq!(
            choose_branch_by_counts(&g, &mids, 0.1, &CoverageRef::unknown()),
            BubbleChoice::Leave
        );

        let (g, _s, mids) = bubble_with_counts(100, 9);
        assert_eq!(
            choose_branch_by_counts(&g, &mids, 0.1, &CoverageRef::unknown()),
            BubbleChoice::Keep(stronger(&g, &mids))
        );
    }

    /// A `0/0` bubble decides nothing and is left alone — the old code had a floor of 1 on the
    /// threshold specifically to force a drop here, and that floor is gone.
    #[test]
    fn a_zero_coverage_bubble_is_left_alone() {
        let (g, _s, mids) = bubble_with_counts(0, 0);
        assert_eq!(
            choose_branch_by_counts(&g, &mids, DEFAULT_POP_RATIO, &CoverageRef::unknown()),
            BubbleChoice::Leave
        );

        // But zero beside anything real is still noise.
        let (g, _s, mids) = bubble_with_counts(50, 0);
        assert_eq!(
            choose_branch_by_counts(&g, &mids, DEFAULT_POP_RATIO, &CoverageRef::unknown()),
            BubbleChoice::Keep(stronger(&g, &mids))
        );
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

    /// The apply-time re-validation: a bubble corrupted between detection and collapse
    /// must be skipped instead of panicking on `[0]`.
    #[test]
    fn a_bubble_that_lost_its_shape_is_skipped() {
        let (mut g, s, mids) = bubble_with_counts(100, 5);
        let w = stronger(&g, &mids);
        let midct = mids[w].1.get_from_and_to().1;
        let (e, ety) = {
            let v = g.out_neighbours_bi(mids[w].0, midct);
            (v[0].0, v[0].1)
        };
        let outct = ety.get_from_and_to().1;
        let f = g.out_neighbours_bi(e, outct)[0].0;
        for eid in g.edges_between(e, f) {
            g.remove_edge(eid);
        }
        for eid in g.edges_between(f, e) {
            g.remove_edge(eid);
        }

        let before = g.node_count();
        assert!(!apply_bubble_collapse(&mut g, s, &mids, w));
        assert_eq!(g.node_count(), before);
    }

    #[test]
    fn valid_bubble_returns_true() {
        let (g, s, edges) = make_valid_bubble();
        assert!(bubble_shape_ok(&g, s, &edges, true));
    }

    #[test]
    fn invalid_midnodes_equal() {
        // Two edges from S to the same node M1 → midnodes[0] == midnodes[1]
        let mut g = DbgGraph::new(3);
        let s = g.add_node(make_node());
        let m1 = g.add_node(make_node());
        g.add_bi_edge(s, m1, EdgeType::MinToMin);
        g.add_edge(s, m1, EdgeType::MinToMax); // second edge to same node
        let edges = g.bubble_start_edges_by_carry(s, CarryType::Min);
        assert_eq!(edges.len(), 2);
        assert!(!bubble_shape_ok(&g, s, &edges, true));
    }

    #[test]
    fn invalid_midnode_is_startn() {
        // One invec edge is a self-loop on S → midnodes[0] == startn
        let mut g = DbgGraph::new(3);
        let s = g.add_node(make_node());
        let m1 = g.add_node(make_node());
        g.add_edge(s, s, EdgeType::MinToMin); // self-loop
        g.add_bi_edge(s, m1, EdgeType::MinToMin);
        let edges = g.bubble_start_edges_by_carry(s, CarryType::Min);
        assert_eq!(edges.len(), 2);
        assert!(!bubble_shape_ok(&g, s, &edges, true));
    }

    #[test]
    fn invalid_wrong_in_degree_of_middle() {
        // Add an extra incoming Min edge to M1 → in_degree check fails
        let (mut g, s, edges) = make_valid_bubble();
        let extra = g.add_node(make_node());
        let m1 = edges[0].target;
        g.add_bi_edge(extra, m1, EdgeType::MinToMin);
        // Now in_neighbours_bi(M1, Min).len() == 2 != 1, so the structure is invalid.
        assert!(!bubble_shape_ok(&g, s, &edges, true));
    }

    #[test]
    fn invalid_paths_diverge() {
        // M1 → E1, M2 → E2 (different end nodes) → false
        let mut g = DbgGraph::new(3);
        let s = g.add_node(make_node());
        let m1 = g.add_node(make_node());
        let m2 = g.add_node(make_node());
        let e1 = g.add_node(make_node());
        let e2 = g.add_node(make_node());
        let f = g.add_node(make_node());
        g.add_bi_edge(s, m1, EdgeType::MinToMin);
        g.add_bi_edge(s, m2, EdgeType::MinToMin);
        g.add_bi_edge(m1, e1, EdgeType::MinToMin);
        g.add_bi_edge(m2, e2, EdgeType::MinToMin); // different end
        g.add_bi_edge(e1, f, EdgeType::MinToMin);
        let edges = g.bubble_start_edges_by_carry(s, CarryType::Min);
        assert!(!bubble_shape_ok(&g, s, &edges, true));
    }

    // ── stage 1.5: N branches, and ends that carry their own traffic ─────────

    /// An N-way bubble: parallel single-node branches from one entrance to one join, with an
    /// upstream flank so the entrance looks like a real branch point.
    fn nway_bubble(counts: &[u32]) -> (DbgGraph, NodeId, Vec<(NodeId, EdgeType)>) {
        let mut g = DbgGraph::new(3);
        let f0 = g.add_node(make_node());
        let s = g.add_node(make_node());
        let e = g.add_node(make_node());
        let f1 = g.add_node(make_node());
        g.add_bi_edge(f0, s, EdgeType::MinToMin);
        for &c in counts {
            let m = g.add_node(NodeStruct {
                counts: c,
                abs_ind: vec![0u64],
                innerdir: None,
            });
            g.add_bi_edge(s, m, EdgeType::MinToMin);
            g.add_bi_edge(m, e, EdgeType::MinToMin);
        }
        g.add_bi_edge(e, f1, EdgeType::MinToMin);
        let mids = g.out_neighbours_min(s);
        (g, s, mids)
    }

    #[test]
    fn a_three_way_bubble_is_a_valid_shape() {
        let (g, s, _) = nway_bubble(&[10, 10, 10]);
        let edges = g.bubble_start_edges_by_carry(s, CarryType::Min);
        assert_eq!(edges.len(), 3);
        assert!(bubble_shape_ok(&g, s, &edges, true));
    }

    /// The join keeping several successors is exactly what 1.5.2 lifts, so it must be rejected under
    /// the strict rule and accepted under the relaxed one.
    #[test]
    fn a_join_with_several_successors_is_only_valid_when_relaxed() {
        let (mut g, s, _) = nway_bubble(&[10, 10]);
        let edges = g.bubble_start_edges_by_carry(s, CarryType::Min);
        let e = g.out_neighbours_bi(edges[0].target, edges[0].edge_type.get_from_and_to().1)[0].0;
        let extra = g.add_node(make_node());
        g.add_bi_edge(e, extra, EdgeType::MinToMin);
        assert!(!bubble_shape_ok(&g, s, &edges, false));
        assert!(bubble_shape_ok(&g, s, &edges, true));
    }

    /// A join whose successor re-enters the bubble would leave a dangling edge after the surgery.
    #[test]
    fn a_join_leading_back_into_the_bubble_is_rejected() {
        let (mut g, s, mids) = nway_bubble(&[10, 10]);
        let edges = g.bubble_start_edges_by_carry(s, CarryType::Min);
        let e = g.out_neighbours_bi(edges[0].target, edges[0].edge_type.get_from_and_to().1)[0].0;
        g.add_bi_edge(e, mids[1].0, EdgeType::MinToMin);
        assert!(!bubble_shape_ok(&g, s, &edges, true));
    }

    /// `out_neighbours_min` does not return branches in insertion order, so tests assert on the
    /// coverages a decision picked out, never on positions.
    fn counts_at(g: &DbgGraph, mids: &[(NodeId, EdgeType)], idx: &[usize]) -> Vec<u32> {
        let mut v: Vec<u32> = idx
            .iter()
            .map(|i| g.node_weight(mids[*i].0).unwrap().counts)
            .collect();
        v.sort_unstable();
        v
    }

    /// Three branches, only one of them weak: the weak one goes and the other two stay, so the next
    /// round sees an ordinary two-way bubble.
    #[test]
    fn a_three_way_bubble_prunes_only_the_qualifying_branch() {
        let (g, _, mids) = nway_bubble(&[100, 5, 90]);
        match choose_branch_by_counts(&g, &mids, DEFAULT_POP_RATIO, &CoverageRef::unknown()) {
            BubbleChoice::Prune(losers) => assert_eq!(counts_at(&g, &mids, &losers), vec![5]),
            other => panic!("expected a partial prune, got {other:?}"),
        }
    }

    /// With every loser weak it collapses outright, as the two-way case always has.
    #[test]
    fn a_three_way_bubble_with_all_losers_weak_collapses() {
        let (g, _, mids) = nway_bubble(&[100, 5, 4]);
        match choose_branch_by_counts(&g, &mids, DEFAULT_POP_RATIO, &CoverageRef::unknown()) {
            BubbleChoice::Keep(w) => assert_eq!(counts_at(&g, &mids, &[w]), vec![100]),
            other => panic!("expected an outright collapse, got {other:?}"),
        }
    }

    /// The N=2 path must be untouched by all of the above: one loser can never be a partial prune.
    #[test]
    fn two_branches_never_produce_a_partial_prune() {
        for (a, b) in [(100u32, 5u32), (100, 90), (10, 10)] {
            let (g, _, mids) = nway_bubble(&[a, b]);
            let choice =
                choose_branch_by_counts(&g, &mids, DEFAULT_POP_RATIO, &CoverageRef::unknown());
            assert!(
                !matches!(choice, BubbleChoice::Prune(_)),
                "({a}, {b}) produced {choice:?}"
            );
        }
    }

    // ---- CoverageRef: the absolute rule, and how it degrades ----

    /// With no peak the new rule must be invisible: every pair decides exactly as the ratio alone did.
    #[test]
    fn an_unknown_peak_reproduces_the_ratio_rule() {
        let unknown = CoverageRef::unknown();
        for (c0, c1) in [(10, 10), (100, 5), (100, 11), (50, 4), (7, 7), (1, 100)] {
            let (g, _s, mids) = bubble_with_counts(c0, c1);
            let expected = if (min(c0, c1) as f32) < DEFAULT_POP_RATIO * max(c0, c1) as f32 {
                BubbleChoice::Keep(stronger(&g, &mids))
            } else {
                BubbleChoice::Leave
            };
            assert_eq!(
                choose_branch_by_counts(&g, &mids, DEFAULT_POP_RATIO, &unknown),
                expected,
                "counts {c0}/{c1}"
            );
        }
    }

    /// The case the ratio cannot reach. At peak 60 the ceiling is 15, so a branch at 12 beside one at
    /// 60 is an error — but 12 is well above `0.1 * 60`, so the ratio leaves it alone.
    #[test]
    fn a_branch_far_under_the_genomic_peak_is_popped() {
        let coverage = CoverageRef::new(PeakSource::Fitted(60), 5);
        let (g, _s, mids) = bubble_with_counts(60, 12);
        assert_eq!(
            choose_branch_by_counts(&g, &mids, DEFAULT_POP_RATIO, &CoverageRef::unknown()),
            BubbleChoice::Leave,
            "the ratio alone must not reach this"
        );
        assert_eq!(
            choose_branch_by_counts(&g, &mids, DEFAULT_POP_RATIO, &coverage),
            BubbleChoice::Keep(stronger(&g, &mids))
        );
    }

    /// The collapsed-repeat regression, restated with a peak set: `is_error_like` and `is_genomic` are
    /// exact complements, so equal counts can never satisfy both however the peak is chosen.
    #[test]
    fn equal_coverage_is_left_alone_even_under_a_peak() {
        for peak in [1u32, 10, 40, 60, 1000] {
            for counts in [3u32, 20, 60] {
                let coverage = CoverageRef::new(PeakSource::Fitted(peak), 2);
                let (g, _s, mids) = bubble_with_counts(counts, counts);
                assert_eq!(
                    choose_branch_by_counts(&g, &mids, DEFAULT_POP_RATIO, &coverage),
                    BubbleChoice::Leave,
                    "peak {peak}, counts {counts}"
                );
            }
        }
    }

    /// Two branches of a real two-copy repeat both sit at genomic depth, so neither is error-like.
    #[test]
    fn two_genomic_branches_are_left_alone() {
        let coverage = CoverageRef::new(PeakSource::Fitted(60), 5);
        let (g, _s, mids) = bubble_with_counts(55, 48);
        assert_eq!(
            choose_branch_by_counts(&g, &mids, DEFAULT_POP_RATIO, &coverage),
            BubbleChoice::Leave
        );
    }

    /// Where `min_count` is a large share of the peak, a bare fraction would land under a cutoff every
    /// surviving node already clears, leaving the rule unsatisfiable. The floor is what prevents that.
    #[test]
    fn the_ceiling_never_falls_to_the_cutoff() {
        // 0.25 * 60 = 15, below the cutoff of 20, so the floor binds at 21.
        assert_eq!(
            CoverageRef::new(PeakSource::Fitted(60), 20).error_ceiling(),
            Some(21)
        );
        // With room to spare the fraction binds instead.
        assert_eq!(
            CoverageRef::new(PeakSource::Fitted(60), 5).error_ceiling(),
            Some(15)
        );
        assert_eq!(CoverageRef::unknown().error_ceiling(), None);
    }

    /// A fallback peak is an occurrence-weighted median, biased upward by repeats, so it must buy a
    /// lower ceiling than the same number fitted — overstating the peak is what deletes real sequence.
    #[test]
    fn a_fallback_peak_is_used_more_conservatively() {
        let fitted = CoverageRef::new(PeakSource::Fitted(100), 2);
        let fallback = CoverageRef::new(PeakSource::Fallback(100), 2);
        assert!(fallback.error_ceiling() < fitted.error_ceiling());
        assert_eq!(fitted.error_ceiling(), Some(25));
        assert_eq!(fallback.error_ceiling(), Some(15));
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
