//! Corrects parts of the provided graph, if needed
use crate::logw;
use sparrowhawk_graph::{
    BubbleStartEdge, CarryType, DbgGraph, EdgeId, EdgeType, NodeId,
};

use crate::EdgeWeight;

use std::{
    cmp::max,
    collections::BTreeSet,
    vec::Drain,
};

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

    /// Solve bubbles from the graph
    fn correct_bubbles(&mut self) -> bool;

    /// Remove all input and output dead paths
    fn remove_dead_paths(&mut self) -> bool;

    /// Find and remove all links that are impossible in bi-directed de Bruijn graphs derived from DNA sequences.
    fn remove_conflictive_links(&mut self) -> bool;
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

    fn correct_bubbles(&mut self) -> bool {
        correct_bubbles_skipping(self, &BTreeSet::new())
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

    fn remove_conflictive_links(&mut self) -> bool {
        false
    }
}

/// Collapse every bubble in the graph, except those in `protected`.
///
/// `protected` is how the multi-k evidence path vetoes the coverage heuristic: a bubble whose branches
/// are *both* corroborated by the reads at the larger k has no wrong branch to pop, so whichever the
/// heuristic chose it would be deleting real sequence.
pub fn correct_bubbles_skipping(g: &mut DbgGraph, protected: &BTreeSet<NodeId>) -> bool {
    let mut dididoanything = false;

    logw("Starting resolution of standard bubbles", Some("info"));
    let bubbles = g
        .node_indices()
        .filter(|n| g.out_degree(*n) == 3)
        .filter(|n| !protected.contains(n))
        .filter(|n| {
            let vmin = g.bubble_start_edges_by_carry(*n, CarryType::Min);
            if vmin.len() > 2 {
                return false;
            }

            if vmin.len() == 2 {
                check_bubble_structure(g, *n, vmin).is_some()
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
        if !protected.is_empty() {
            logw(
                format!(
                    "Multi-k vetoed {} bubble(s): both branches are corroborated at the evidence k, \
                     so there is no wrong branch to pop.",
                    protected.len()
                )
                .as_str(),
                Some("info"),
            );
        }
        for n in bubbles {
            if g.contains_node(n) {
                let tmpb = collapse_bubble(g, n);
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

/// Every part of a bubble, as `check_bubble_structure` derives it.
///
/// It used to compute all of this and then throw it away to return a `bool`, leaving `collapse_bubble`
/// to rebuild it from scratch. The multi-k oracle needs the same parts — plus the two flanking unitigs,
/// which are what supply the context a larger k needs to judge the branches — so hand them back.
#[derive(Debug, Clone)]
pub struct BubbleParts {
    /// The fork. Exactly one k-mer, traversed on `CarryType::Min`.
    pub start: NodeId,
    /// The two branches, with the edge that reaches each from `start`.
    pub mid: [(NodeId, EdgeType); 2],
    /// Traversal carry of each branch.
    pub midct: [CarryType; 2],
    /// The join. Exactly one k-mer.
    pub end: NodeId,
    /// Traversal carry of `end`.
    pub endct: CarryType,
    /// Downstream flanking unitig, and the edge reaching it from `end`.
    pub right: (NodeId, EdgeType),
    /// Upstream flanking unitig, and the edge reaching `start` from it.
    ///
    /// `out_degree(start) == 3` decomposes as two outgoing `Min` branches plus one outgoing `Max`
    /// back-link — the mate `add_bi_edge` installs for the incoming flank edge — so a well-formed
    /// bubble start has exactly one of these. `None` if it does not, which the oracle treats as
    /// "no context" rather than trusting it.
    pub left: Option<(NodeId, EdgeType)>,
}

/// Checks whether the candidate area can be a good bubble for error correction, returning its parts.
fn check_bubble_structure(
    ptgraph: &DbgGraph,
    startn: NodeId,
    invec: Vec<BubbleStartEdge>,
) -> Option<BubbleParts> {
    let mut midnodes = Vec::with_capacity(2);
    let mut midcts = Vec::with_capacity(2);
    let mut midedges = Vec::with_capacity(2);

    for e in invec {
        midnodes.push(e.target);
        midcts.push(e.edge_type.get_from_and_to().1);
        midedges.push(e.edge_type);
    }

    // We need to check that the two intermediate nodes are different
    if midnodes[0] == midnodes[1] || midnodes[0] == startn || midnodes[1] == startn {
        return None;
    }

    // Now, how many neighbours do we have from the middle nodes?
    let tmpv0 = ptgraph.out_neighbours_bi(midnodes[0], midcts[0]);
    if tmpv0.len() != 1 || ptgraph.in_neighbours_bi(midnodes[0], midcts[0]).len() != 1 {
        return None;
    }
    let outnode = tmpv0[0].0;
    let tmpv1 = ptgraph.out_neighbours_bi(midnodes[1], midcts[1]);
    let outct = tmpv0[0].1.get_from_and_to().1;

    if (tmpv1.len() != 1)
        || (tmpv1[0].0 != outnode)
        || (outct != tmpv1[0].1.get_from_and_to().1)
        || (ptgraph.in_neighbours_bi(midnodes[1], midcts[1]).len() != 1)
    {
        return None;
    }

    let tmpv3 = ptgraph.out_neighbours_bi(outnode, outct);
    if tmpv3.len() != 1
        || tmpv3[0].0 == startn
        || ptgraph.in_neighbours_bi(outnode, outct).len() != 2
    {
        return None;
    }

    // The upstream flank. Not part of the structural test — a bubble is still a bubble without one —
    // so it is optional, and only the oracle cares.
    let inmin = ptgraph.in_neighbours_bi(startn, CarryType::Min);
    let left = if inmin.len() == 1 { Some(inmin[0]) } else { None };

    Some(BubbleParts {
        start: startn,
        mid: [(midnodes[0], midedges[0]), (midnodes[1], midedges[1])],
        midct: [midcts[0], midcts[1]],
        end: outnode,
        endct: outct,
        right: tmpv3[0],
        left,
    })
}

/// Derive a bubble's parts from its start node alone. The multi-k oracle's entry point.
pub fn bubble_parts(ptgraph: &DbgGraph, startn: NodeId) -> Option<BubbleParts> {
    if ptgraph.out_degree(startn) != 3 {
        return None;
    }
    let vmin = ptgraph.bubble_start_edges_by_carry(startn, CarryType::Min);
    if vmin.len() != 2 {
        return None;
    }
    check_bubble_structure(ptgraph, startn, vmin)
}

/// What the coverage heuristic decided to do with a bubble.
///
/// Split out of `collapse_bubble` so the multi-k path can substitute its own decision and still reuse
/// the surgery verbatim — the bidirected bookkeeping in `apply_bubble_collapse` is the hardest code
/// here and must not be reimplemented.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BubbleChoice {
    /// Collapse the bubble onto this branch, discarding the other.
    Keep(usize),
    /// Cut this branch loose. Leaves a contig break rather than guessing.
    SeverOne(usize),
    /// Cut both branches loose: the evidence does not favour either. Deliberate — an ambiguous bubble
    /// becomes a contig break, as SKESA does, rather than a coin flip.
    SeverBoth,
}

/// The coverage heuristic. Pure: it inspects the graph and decides, but changes nothing.
fn choose_branch_by_counts(
    ptgraph: &DbgGraph,
    startn: NodeId,
    midconns: &[(NodeId, EdgeType)],
) -> BubbleChoice {
    let node0w = ptgraph.node_weight(midconns[0].0).unwrap();
    let node1w = ptgraph.node_weight(midconns[1].0).unwrap();

    // Inspired by Skesa: a branch carrying less than 10% of the stronger branch's coverage is noise.
    // The floor of 1 is there so a `counts == 0` branch is always dropped; it must be a floor, not a
    // ceiling. This was `min`, which clamped the threshold to at most 1 — and since every surviving
    // k-mer has `counts >= min_count`, the two arms below could then never fire, so the whole filter
    // was dead code.
    let count_threshold = max(
        (0.1_f32 * max(node0w.counts, node1w.counts) as f32).round() as u16,
        1,
    );

    if node0w.counts < count_threshold {
        return BubbleChoice::Keep(1);
    }
    if node1w.counts < count_threshold {
        return BubbleChoice::Keep(0);
    }

    // Cast before subtracting: these are usizes, so `a - b` underflows whenever branch 0 is the shorter
    // one. Release wrapped and the `as i32` truncation recovered the right negative value by accident;
    // debug panicked with "attempt to subtract with overflow".
    let lendiff = (node0w.abs_ind.len() as i32 - node1w.abs_ind.len() as i32).abs() as f32
        / (max(node0w.abs_ind.len(), node1w.abs_ind.len()) as f32);

    if lendiff > 0.025 {
        // The branches differ in length, so coverage alone cannot say which is right. Compare each
        // against the coverage of the sequence flanking the bubble instead, and cut loose whichever
        // deviates. If both deviate, or neither does, cut both: an ambiguous bubble becomes a contig
        // break rather than a guess.
        let startn_counts = ptgraph.node_weight(startn).unwrap().counts;
        let endn_counts = ptgraph
            .node_weight(
                ptgraph.out_neighbours_bi(midconns[0].0, midconns[0].1.get_from_and_to().1)[0].0,
            )
            .unwrap()
            .counts;
        let average_surrounding_counts = ((startn_counts + endn_counts) as f32 / 2.0).round() as u16;

        let rel_diff = |c: u16| {
            ((c as i32) - (average_surrounding_counts as i32)).abs() as f32
                / (average_surrounding_counts as f32)
        };
        let (rel_diff_0, rel_diff_1) = (rel_diff(node0w.counts), rel_diff(node1w.counts));

        if rel_diff_0 > 0.2 && rel_diff_1 <= 0.2 {
            BubbleChoice::SeverOne(0)
        } else if rel_diff_0 <= 0.2 && rel_diff_1 > 0.2 {
            BubbleChoice::SeverOne(1)
        } else {
            BubbleChoice::SeverBoth
        }
    } else if node0w.counts > node1w.counts {
        BubbleChoice::Keep(0)
    } else if node0w.counts < node1w.counts {
        BubbleChoice::Keep(1)
    } else if node0w.abs_ind.len() > node1w.abs_ind.len() {
        BubbleChoice::Keep(0)
    } else {
        BubbleChoice::Keep(1)
    }
}

/// The surgery: fuse the bubble onto `winner`, discarding the other branch.
///
/// Removes both branches and the end node, merges the winner into `start`, appends the end node's
/// single k-mer, and reattaches to the downstream flank. The precondition — `start` and `end` each hold
/// exactly one k-mer — is what makes the hardcoded `set_internal_edge(MinToMin)` sound: `start` was
/// reached on the `Min` strand and its lone k-mer sits at index 0, so the fused `abs_ind` reads
/// front-to-back in `Min`. `shrink` never merges a junction node, so it holds in practice.
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

/// This function collapses standard bubbles depending on the number of counts (very naive)
fn collapse_bubble(ptgraph: &mut DbgGraph, startn: NodeId) -> bool {
    // Deliberately weaker than `check_bubble_structure`: earlier collapses in the same pass may have
    // reshaped the graph, and this is the guard the heuristic has always used. Tightening it here would
    // change which bubbles get collapsed.
    let midconns = ptgraph.out_neighbours_min(startn);
    if midconns.len() != 2
        || ptgraph
            .out_neighbours_bi(midconns[0].0, midconns[0].1.get_from_and_to().1)
            .len()
            != 1
    {
        return false;
    }

    match choose_branch_by_counts(ptgraph, startn, &midconns) {
        BubbleChoice::Keep(w) => apply_bubble_collapse(ptgraph, startn, &midconns, w),
        BubbleChoice::SeverOne(b) => {
            ptgraph.remove_all_edges_of(midconns[b].0);
            true
        }
        BubbleChoice::SeverBoth => {
            ptgraph.remove_all_edges_of(midconns[0].0);
            ptgraph.remove_all_edges_of(midconns[1].0);
            true
        }
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
    let minnts = 100;
    let limit = max(0, minnts - k + 1);

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

    #[test]
    fn dead_path_limit_k3() {
        let (minnts, k) = (100usize, 3usize);
        let limit = std::cmp::max(0, minnts - k + 1);
        assert_eq!(limit, 98);
    }

    #[test]
    fn dead_path_limit_k100() {
        let (minnts, k) = (100usize, 100usize);
        let limit = std::cmp::max(0, minnts - k + 1);
        assert_eq!(limit, 1);
    }

    #[test]
    fn dead_path_limit_k31() {
        let (minnts, k) = (100usize, 31usize);
        let limit = std::cmp::max(0, minnts - k + 1);
        assert_eq!(limit, 70);
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

    #[test]
    fn valid_bubble_returns_true() {
        let (g, s, edges) = make_valid_bubble();
        assert!(check_bubble_structure(&g, s, edges).is_some());
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
        assert!(check_bubble_structure(&g, s, edges).is_none());
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
        assert!(check_bubble_structure(&g, s, edges).is_none());
    }

    #[test]
    fn invalid_wrong_in_degree_of_middle() {
        // Add an extra incoming Min edge to M1 → in_degree check fails
        let (mut g, s, edges) = make_valid_bubble();
        let extra = g.add_node(make_node());
        let m1 = edges[0].target;
        g.add_bi_edge(extra, m1, EdgeType::MinToMin);
        // Now in_neighbours_bi(M1, Min).len() == 2 != 1, so the structure is invalid.
        assert!(check_bubble_structure(&g, s, edges).is_none());
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
        assert!(check_bubble_structure(&g, s, edges).is_none());
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
