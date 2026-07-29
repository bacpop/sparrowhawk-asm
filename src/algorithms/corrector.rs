//! Corrects parts of the provided graph, if needed
use crate::logw;
use sparrowhawk_graph::{
    BubbleStartEdge, CarryType, DbgGraph, EdgeId, EdgeType, NodeId,
};

use crate::EdgeWeight;

use std::{
    cmp::{max, min},
    collections::BTreeSet,
    vec::Drain,
};

/// Minimum number of k-mers a path must exceed to be worth keeping in dead-end removal.
pub(crate) fn short_path_limit(minnts: usize, k: usize) -> usize {
    (minnts + 1).saturating_sub(k) // sat_sub is compulsory, becase as these are usize, going negative my change the path to an absurd value!!!
}

/// A branch carrying less than this fraction of the stronger branch's coverage is noise. SKESA-inspired.
pub const DEFAULT_POP_RATIO: f32 = 0.1;

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
    fn correct_bubbles(&mut self, pop_ratio: f32) -> bool;

    /// Remove all input and output dead paths
    fn remove_dead_paths(&mut self) -> bool;
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

    fn correct_bubbles(&mut self, pop_ratio: f32) -> bool {
        pop_bubbles_by_coverage(self, pop_ratio)
    }

    fn remove_dead_paths(&mut self) -> bool {
        logw(
            format!(
                "Before pruning: {} nodes and {} edges",
                self.node_count(),
                self.edge_count()
            )
            .as_str(),
            Some("info"),
        );

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
                check_dead_path(self, v, &mut path_check_vec, self.k(), carryedge);
                if !path_check_vec.is_empty() {
                    dididoanything = true;
                    to_remove.append(&mut path_check_vec);
                }
            }

            // if there are no dead paths left pruning is done
            if to_remove.is_empty() {
                logw("Graph is pruned.", Some("info"));
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

/// Collapse every bubble whose two branches differ significantly enough in coverage; leave the rest alone.
pub fn pop_bubbles_by_coverage(g: &mut DbgGraph, pop_ratio: f32) -> bool {
    let mut dididoanything = false;

    logw("Starting resolution of standard bubbles", Some("info"));
    let bubbles = g
        .node_indices()
        .filter(|n| g.out_degree(*n) == 3)
        .filter(|n| {
            let vmin = g.bubble_start_edges_by_carry(*n, CarryType::Min);
            if vmin.len() > 2 {
                return false;
            }

            if vmin.len() == 2 {
                check_bubble_structure(g, *n, vmin)
            } else {
                false
            }
        })
        .collect::<BTreeSet<NodeId>>();

    if bubbles.is_empty() {
        return false;
    } else {
        logw(
            format!(
                "Found {:?} potential bubbles (they might be less). Starting to collapse them ",
                bubbles.len()
            )
            .as_str(),
            Some("info"),
        );
        for n in bubbles {
            if g.contains_node(n) {
                let tmpb = collapse_bubble(g, n, pop_ratio);
                if tmpb {
                    dididoanything = true;
                }
            }
        }
    }

    logw(
        format!(
            "Bubble correction ended. Corrected graph has {} nodes and {} edges",
            g.node_count(),
            g.edge_count()
        )
        .as_str(),
        Some("info"),
    );

    dididoanything
}

/// Checks whether the candidate area can be a good bubble for error correction.
fn check_bubble_structure(ptgraph: &DbgGraph, startn: NodeId, invec: Vec<BubbleStartEdge>) -> bool {
    let mut midnodes = Vec::with_capacity(2);
    let mut midcts = Vec::with_capacity(2);

    for e in invec {
        midnodes.push(e.target);
        midcts.push(e.edge_type.get_from_and_to().1);
    }

    // We need to check that the two intermediate nodes are different
    if midnodes[0] == midnodes[1] || midnodes[0] == startn || midnodes[1] == startn {
        return false;
    }

    // Now, how many neighbours do we have from the middle nodes?
    let tmpv0 = ptgraph.out_neighbours_bi(midnodes[0], midcts[0]);
    if tmpv0.len() != 1 || ptgraph.in_neighbours_bi(midnodes[0], midcts[0]).len() != 1 {
        return false;
    }
    let outnode = tmpv0[0].0;
    let tmpv1 = ptgraph.out_neighbours_bi(midnodes[1], midcts[1]);
    let outct = tmpv0[0].1.get_from_and_to().1;

    if (tmpv1.len() != 1)
        || (tmpv1[0].0 != outnode)
        || (outct != tmpv1[0].1.get_from_and_to().1)
        || (ptgraph.in_neighbours_bi(midnodes[1], midcts[1]).len() != 1)
    {
        return false;
    }

    let tmpv3 = ptgraph.out_neighbours_bi(outnode, outct);
    tmpv3.len() == 1 && tmpv3[0].0 != startn && ptgraph.in_neighbours_bi(outnode, outct).len() == 2
}

/// What the coverage heuristic decided to do with a bubble.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BubbleChoice {
    /// Collapse the bubble onto this branch, discarding the other.
    Keep(usize),
    /// Do nothing at all. The bubble stays exactly as it is, contig break and all.
    Leave,
}

/// The coverage heuristic. It inspects the graph and decides, but changes nothing.
fn choose_branch_by_counts(
    ptgraph: &DbgGraph,
    midconns: &[(NodeId, EdgeType)],
    pop_ratio: f32,
) -> BubbleChoice {
    let c0 = ptgraph.node_weight(midconns[0].0).unwrap().counts;
    let c1 = ptgraph.node_weight(midconns[1].0).unwrap().counts;
    let hi = max(c0, c1);
    let lo = min(c0, c1);

    // Strict `<`, so equal coverages can never pop however the ratio is set.
    if (lo as f32) < pop_ratio * hi as f32 {
        BubbleChoice::Keep(if c0 > c1 { 0 } else { 1 })
    } else {
        BubbleChoice::Leave
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
    let midconn2 = ptgraph.out_neighbours_bi(midconns[winner].0, midnodect)[0];
    let outct = midconn2.1.get_from_and_to().1;
    let outconn = ptgraph.out_neighbours_bi(midconn2.0, outct)[0];
    let savedoutw = ptgraph.node_weight(midconn2.0).unwrap().clone();

    if ptgraph.node_weight(startn).unwrap().abs_ind.len() > 1
        || ptgraph.node_weight(midconn2.0).unwrap().abs_ind.len() > 1
    {
        panic!("Trying to remove a bubble with a start or end node with more than one k-mer!");
    };

    ptgraph.remove_node(midconns[0].0); // intermediate node 0
    ptgraph.remove_node(midconns[1].0); // intermediate node 1
    ptgraph.remove_node(midconn2.0); // end node

    let mutrefw = ptgraph.node_weight_mut(startn).unwrap();

    mutrefw.merge(&savedmidw, midconns[winner].1);
    mutrefw.set_internal_edge(EdgeType::MinToMin);
    mutrefw.abs_ind.push(savedoutw.abs_ind[0]);
    mutrefw.set_mean_counts(&[mutrefw.counts, savedmidw.counts, savedoutw.counts]);

    match outct {
        CarryType::Min => {
            ptgraph.add_bi_edge(startn, outconn.0, outconn.1);
        }
        CarryType::Max => {
            let tmptype = EdgeType::from_carrytypes(CarryType::Min, outconn.1.get_from_and_to().1);
            ptgraph.add_bi_edge(startn, outconn.0, tmptype);
        }
    }
    true
}

/// Collapse one standard bubble, if the coverage difference between its branches is big enough.
fn collapse_bubble(ptgraph: &mut DbgGraph, startn: NodeId, pop_ratio: f32) -> bool {
    let midconns = ptgraph.out_neighbours_min(startn);
    if midconns.len() != 2
        || ptgraph
            .out_neighbours_bi(midconns[0].0, midconns[0].1.get_from_and_to().1)
            .len()
            != 1
    {
        return false;
    }

    match choose_branch_by_counts(ptgraph, &midconns, pop_ratio) {
        BubbleChoice::Keep(w) => apply_bubble_collapse(ptgraph, startn, &midconns, w),
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

/// Check if vertex initializes a dead path.
#[inline]
fn check_dead_path(
    ptgraph: &DbgGraph,
    vertex: NodeId,
    output_vec: &mut Vec<NodeId>,
    k: usize,
    carryedge: EdgeType,
) {
    let mut current_vertex = vertex;
    output_vec.push(current_vertex);
    let cntopt = ptgraph.node_weight(current_vertex);
    let mut cnt: usize;
    if let Some(thecntopt) = cntopt {
        cnt = thecntopt.abs_ind.len();
    } else {
        return;
    }

    let (mut ty, _) = carryedge.get_from_and_to();
    let limit = short_path_limit(100, k);

    loop {
        if cnt >= limit {
            output_vec.clear();
            return;
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
            let mut altpath: Vec<Vec<NodeId>> = Vec::with_capacity(nbkgn_c - 1);
            let mut maxlen = 0;
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
                    let tmplen = tmppath.len();
                    if tmplen != 0 {
                        altpath.push(tmppath);
                        if tmplen > maxlen {
                            maxlen = tmplen;
                        }
                    }
                }
            }

            if maxlen != 0 && cnt > maxlen {
                output_vec.clear();
                for iv in altpath.iter_mut() {
                    output_vec.append(iv);
                }
            }
            return;
        }

        if nfwdn_c == 0 {
            output_vec.push(candidate_node);
            cnt += ptgraph.node_weight(candidate_node).unwrap().abs_ind.len();

            if cnt >= limit {
                output_vec.clear();
            }
            return;
        } else if nfwdn_c == 1 {
            current_vertex = candidate_node;
            ty = candidate_ty;
            output_vec.push(current_vertex);
            cnt += ptgraph.node_weight(current_vertex).unwrap().abs_ind.len();
        } else {
            output_vec.clear();
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparrowhawk_graph::{DbgGraph, NodeStruct};

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

    // ── check_bubble_structure ───────────────────────────────────────────────

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
        let m1 = g.add_node(NodeStruct { counts: c0, ..make_node() });
        let m2 = g.add_node(NodeStruct { counts: c1, ..make_node() });
        let e = g.add_node(make_node());
        let f = g.add_node(make_node());
        g.add_bi_edge(f0, s, EdgeType::MinToMin);
        g.add_bi_edge(s, m1, EdgeType::MinToMin);
        g.add_bi_edge(s, m2, EdgeType::MinToMin);
        g.add_bi_edge(m1, e, EdgeType::MinToMin);
        g.add_bi_edge(m2, e, EdgeType::MinToMin);
        g.add_bi_edge(e, f, EdgeType::MinToMin);
        assert_eq!(g.out_degree(s), 3, "fixture must be a candidate for the popper");
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
                choose_branch_by_counts(&g, &mids, ratio),
                BubbleChoice::Leave,
                "equal coverage must never pop, ratio {ratio}"
            );
        }

        let (mut g, _s, _) = bubble_with_counts(40, 40);
        let before = g.node_count();
        assert!(!pop_bubbles_by_coverage(&mut g, DEFAULT_POP_RATIO));
        assert_eq!(g.node_count(), before, "the graph must be untouched");
    }

    /// A branch carrying a twentieth of the other is noise, and is popped — whichever way round the
    /// two are stored.
    #[test]
    fn a_noise_branch_is_popped() {
        for (c0, c1) in [(100u32, 5u32), (5, 100)] {
            let (g, _s, mids) = bubble_with_counts(c0, c1);
            assert_eq!(
                choose_branch_by_counts(&g, &mids, 0.1),
                BubbleChoice::Keep(stronger(&g, &mids)),
                "must keep the stronger branch for ({c0}, {c1})"
            );
        }

        let (mut g, _s, _) = bubble_with_counts(100, 5);
        let before = g.node_count();
        assert!(pop_bubbles_by_coverage(&mut g, DEFAULT_POP_RATIO));
        assert!(g.node_count() < before, "the popped branch and end node are gone");
    }

    /// Pins the boundary: the comparison is a strict `<`, so a branch sitting exactly on the ratio
    /// survives. Getting this backwards would pop bubbles at exactly 10 %, which is the side of the
    /// line where two real things start to look alike.
    #[test]
    fn a_branch_exactly_on_the_ratio_survives() {
        let (g, _s, mids) = bubble_with_counts(100, 10);
        assert_eq!(choose_branch_by_counts(&g, &mids, 0.1), BubbleChoice::Leave);

        let (g, _s, mids) = bubble_with_counts(100, 9);
        assert_eq!(
            choose_branch_by_counts(&g, &mids, 0.1),
            BubbleChoice::Keep(stronger(&g, &mids))
        );
    }

    /// A `0/0` bubble decides nothing and is left alone — the old code had a floor of 1 on the
    /// threshold specifically to force a drop here, and that floor is gone.
    #[test]
    fn a_zero_coverage_bubble_is_left_alone() {
        let (g, _s, mids) = bubble_with_counts(0, 0);
        assert_eq!(choose_branch_by_counts(&g, &mids, DEFAULT_POP_RATIO), BubbleChoice::Leave);

        // But zero beside anything real is still noise.
        let (g, _s, mids) = bubble_with_counts(50, 0);
        assert_eq!(
            choose_branch_by_counts(&g, &mids, DEFAULT_POP_RATIO),
            BubbleChoice::Keep(stronger(&g, &mids))
        );
    }

    #[test]
    fn valid_bubble_returns_true() {
        let (g, s, edges) = make_valid_bubble();
        assert!(check_bubble_structure(&g, s, edges));
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
        assert!(!check_bubble_structure(&g, s, edges));
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
        assert!(!check_bubble_structure(&g, s, edges));
    }

    #[test]
    fn invalid_wrong_in_degree_of_middle() {
        // Add an extra incoming Min edge to M1 → in_degree check fails
        let (mut g, s, edges) = make_valid_bubble();
        let extra = g.add_node(make_node());
        let m1 = edges[0].target;
        g.add_bi_edge(extra, m1, EdgeType::MinToMin);
        // Now in_neighbours_bi(M1, Min).len() == 2 != 1, so the structure is invalid.
        assert!(!check_bubble_structure(&g, s, edges));
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
        assert!(!check_bubble_structure(&g, s, edges));
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
