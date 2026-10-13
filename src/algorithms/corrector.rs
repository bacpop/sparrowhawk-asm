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
                // A dead end with several neighbours is its own fan-out; a self-loop has no dead side.
                .filter(|n| match self.out_degree(*n) {
                    0 => false,
                    1 => true,
                    _ => !self.has_self_loop(*n),
                })
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

    // A dead end that fans out is its own junction, Minia's single-node tip: its successors are the
    // whole neighbourhood.
    if ptgraph.out_neighbours_bi(current_vertex, ty).len() > 1 {
        if cnt < limit {
            return false;
        }
        if cnt < rctc_limit
            && neighbourhood_outcovers_tip(
                ptgraph,
                current_vertex,
                ty,
                current_vertex,
                path_mean_coverage(ptgraph, output_vec),
                rctc_cutoff,
            )
        {
            return true;
        }
        output_vec.clear();
        return false;
    }

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
            // path through its fan-out endpoint.
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
    use crate::cli::DEFAULT_TIP_RCTC_CUTOFF;
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

    /// A dead end that itself fans out: `tip_at_fanout` with its tip removed.
    fn fanning_dead_end(kmers: usize, counts: u32, branch_counts: [u32; 2]) -> (DbgGraph, NodeId) {
        let (mut graph, tip, fanout, _) =
            tip_at_fanout(EdgeType::MinToMin, 1, 1, kmers, counts, branch_counts);
        graph.remove_node(tip);
        (graph, fanout)
    }

    #[test]
    fn a_short_dead_end_that_fans_out_is_removed_by_length() {
        let (graph, dead_end) = fanning_dead_end(2, 999, [1, 1]);
        assert_eq!(walk_tip(&graph, dead_end, 2.0), (vec![dead_end], false));
    }

    #[test]
    fn a_long_dead_end_that_fans_out_goes_only_when_its_successors_outcover_it() {
        let (graph, dead_end) = fanning_dead_end(12, 2, [40, 40]);
        assert_eq!(walk_tip(&graph, dead_end, 2.0), (vec![dead_end], true));
        let (graph, dead_end) = fanning_dead_end(12, 10, [20, 20]);
        assert_eq!(walk_tip(&graph, dead_end, 2.0), (vec![], false));
        let (graph, dead_end) = fanning_dead_end(100, 2, [40, 40]);
        assert_eq!(walk_tip(&graph, dead_end, 2.0), (vec![], false));
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

    /// The tip threshold is the larger of the flat floor and the k multiple; a multiple of zero or below
    /// keeps the flat floor.
    #[test]
    fn tip_length_nts_cases() {
        // (flat bases, k multiple, k, threshold)
        for (flat, kmult, k, want) in [
            (100, 0.0, 81, 100),
            (100, -1.0, 81, 100),
            (100, 2.5, 31, 100),
            (100, 2.5, 81, 203),
            (100, 10.0, 81, 810),
        ] {
            assert_eq!(
                tip_length_nts(flat, kmult, k),
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
