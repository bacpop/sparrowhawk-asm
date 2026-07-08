//! Corrects parts of the provided graph, if needed
use crate::logw;
use sparrowhawk_graph::{CarryType, DbgGraph, EdgeIndex, EdgeType, NodeIndex, NodeStruct};

use crate::EdgeWeight;

use std::{
    cmp::{max, min},
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
    type EdgeIdx = EdgeIndex;

    type NodeIdx = NodeIndex;

    fn remove_weak_nodes(&mut self, threshold: EdgeWeight) {
        self.retain_nodes_by_count(threshold);
    }

    fn remove_self_loops(&mut self) {
        DbgGraph::remove_self_loops(self);
    }

    fn correct_bubbles(&mut self) -> bool {
        let mut dididoanything = false;

        logw("Starting resolution of standard bubbles", Some("info"));
        let bubbles = self
            .node_indices()
            .filter(|n| self.out_degree(*n) == 3)
            .filter(|n| {
                let vmin = self.outgoing_edges_by_carry(*n, CarryType::Min);
                if vmin.len() > 2 {
                    return false;
                }

                if vmin.len() == 2 {
                    check_bubble_structure(
                        self,
                        *n,
                        vmin.into_iter().map(|(edge, _, _)| edge).collect(),
                    )
                } else {
                    false
                }
            })
            .collect::<BTreeSet<NodeIndex>>();

        if bubbles.is_empty() {
            return false;
        } else {
            logw(
                format!(
                    "Found {:?} potential bubbles (they might be less). Starting to collapse them ",
                    bubbles.len()
                )
                .as_str(),
                Some("trace"),
            );
            for n in bubbles {
                if self.contains_node(n) {
                    let tmpb = collapse_bubble(self, n);
                    if tmpb {
                        dididoanything = true;
                    }
                }
            }
        }

        logw(
            format!(
                "Bubble correction ended. Corrected graph has {} nodes and {} edges",
                self.node_count(),
                self.edge_count()
            )
            .as_str(),
            Some("info"),
        );

        dididoanything
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

        let mut to_remove: Vec<NodeIndex> = vec![];
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

/// Checks whether the candidate area can be a good bubble for error correction.
fn check_bubble_structure(ptgraph: &DbgGraph, startn: NodeIndex, invec: Vec<EdgeIndex>) -> bool {
    let mut midnodes = Vec::with_capacity(2);
    let mut midcts = Vec::with_capacity(2);

    for e in invec {
        midnodes.push(ptgraph.edge_endpoints(e).unwrap().1);
        midcts.push(ptgraph.edge_weight(e).unwrap().t.get_from_and_to().1);
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
    if tmpv3.len() != 1
        || tmpv3[0].0 == startn
        || ptgraph.in_neighbours_bi(outnode, outct).len() != 2
    {
        return false;
    }

    true
}

/// This function collapses standard bubbles depending on the number of counts (very naive)
fn collapse_bubble(ptgraph: &mut DbgGraph, startn: NodeIndex) -> bool {
    let midconns = ptgraph.out_neighbours_min(startn);
    if midconns.len() != 2
        || ptgraph
            .out_neighbours_bi(midconns[0].0, midconns[0].1.get_from_and_to().1)
            .len()
            != 1
    {
        return false;
    }
    let chosennode: usize;
    let node0w = ptgraph.node_weight(midconns[0].0).unwrap();
    let node1w = ptgraph.node_weight(midconns[1].0).unwrap();
    let savedmidw: NodeStruct;

    let count_threshold = min(
        (0.1_f32 * max(node0w.counts, node1w.counts) as f32).round() as u16,
        1,
    ); // Inspired by Skesa

    if node0w.counts < count_threshold {
        chosennode = 1;
        savedmidw = node1w.clone();
    } else if node1w.counts < count_threshold {
        chosennode = 0;
        savedmidw = node0w.clone();
    } else {
        if ((node0w.abs_ind.len() - node1w.abs_ind.len()) as i32).abs() as f32
            / (max(node0w.abs_ind.len(), node1w.abs_ind.len()) as f32)
            > 0.025
        {
            let startn_counts = ptgraph.node_weight(startn).unwrap().counts;

            let endn_counts = ptgraph
                .node_weight(
                    ptgraph.out_neighbours_bi(midconns[0].0, midconns[0].1.get_from_and_to().1)[0]
                        .0,
                )
                .unwrap()
                .counts;
            let average_surrounding_counts =
                ((startn_counts + endn_counts) as f32 / 2.0).round() as u16;

            let rel_diff_0 = ((node0w.counts as i32) - (average_surrounding_counts as i32)).abs()
                as f32
                / (average_surrounding_counts as f32);
            let rel_diff_1 = ((node1w.counts as i32) - (average_surrounding_counts as i32)).abs()
                as f32
                / (average_surrounding_counts as f32);

            if rel_diff_0 > 0.2 && rel_diff_1 <= 0.2 {
                ptgraph.remove_all_edges_of(midconns[0].0);
            } else if rel_diff_0 <= 0.2 && rel_diff_1 > 0.2 {
                ptgraph.remove_all_edges_of(midconns[1].0);
            } else {
                ptgraph.remove_all_edges_of(midconns[0].0);
                ptgraph.remove_all_edges_of(midconns[1].0);
            }
            return true;
        } else {
            if node0w.counts > node1w.counts {
                chosennode = 0;
                savedmidw = node0w.clone();
            } else if node0w.counts < node1w.counts {
                chosennode = 1;
                savedmidw = node1w.clone();
            } else if node0w.abs_ind.len() > node1w.abs_ind.len() {
                chosennode = 0;
                savedmidw = node0w.clone();
            } else {
                chosennode = 1;
                savedmidw = node1w.clone();
            }
        }
    }

    let midnodect = midconns[chosennode].1.get_from_and_to().1;
    let midconn2 = ptgraph.out_neighbours_bi(midconns[chosennode].0, midnodect)[0];
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

    mutrefw.merge(&savedmidw, midconns[chosennode].1);
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

/// Remove dead input path.
#[inline]
fn remove_paths(ptgraph: &mut DbgGraph, to_remove: Drain<NodeIndex>) {
    log::trace!("Removing {} dead paths", to_remove.len());
    for n in to_remove {
        ptgraph.remove_node(n);
    }
}

/// Check if vertex initializes a dead path.
#[inline]
fn check_dead_path(
    ptgraph: &DbgGraph,
    vertex: NodeIndex,
    output_vec: &mut Vec<NodeIndex>,
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
            let mut altpath: Vec<Vec<NodeIndex>> = Vec::with_capacity(nbkgn_c - 1);
            let mut maxlen = 0;
            for n in bkgneigh_c.iter() {
                if n.0 == *output_vec.last().unwrap() {
                    continue;
                } else {
                    let mut tmppath: Vec<NodeIndex> = Vec::new();
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
    fn make_valid_bubble() -> (DbgGraph, NodeIndex, EdgeIndex, EdgeIndex) {
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
        let e_s_m1 = g.edges_between(s, m1)[0];
        let e_s_m2 = g.edges_between(s, m2)[0];
        (g, s, e_s_m1, e_s_m2)
    }

    #[test]
    fn valid_bubble_returns_true() {
        let (g, s, e1, e2) = make_valid_bubble();
        assert!(check_bubble_structure(&g, s, vec![e1, e2]));
    }

    #[test]
    fn invalid_midnodes_equal() {
        // Two edges from S to the same node M1 → midnodes[0] == midnodes[1]
        let mut g = DbgGraph::new(3);
        let s = g.add_node(make_node());
        let m1 = g.add_node(make_node());
        g.add_bi_edge(s, m1, EdgeType::MinToMin);
        g.add_edge(s, m1, EdgeType::MinToMax); // second edge to same node
        let edges = g.edges_between(s, m1);
        assert_eq!(edges.len(), 2);
        assert!(!check_bubble_structure(&g, s, vec![edges[0], edges[1]]));
    }

    #[test]
    fn invalid_midnode_is_startn() {
        // One invec edge is a self-loop on S → midnodes[0] == startn
        let mut g = DbgGraph::new(3);
        let s = g.add_node(make_node());
        let m1 = g.add_node(make_node());
        g.add_edge(s, s, EdgeType::MinToMin); // self-loop
        g.add_bi_edge(s, m1, EdgeType::MinToMin);
        let e_self = g.edges_between(s, s)[0];
        let e_s_m1 = g.edges_between(s, m1)[0];
        assert!(!check_bubble_structure(&g, s, vec![e_self, e_s_m1]));
    }

    #[test]
    fn invalid_wrong_in_degree_of_middle() {
        // Add an extra incoming Min edge to M1 → in_degree check fails
        let (mut g, s, e1, e2) = make_valid_bubble();
        let extra = g.add_node(make_node());
        // Find M1 (target of e1)
        let m1 = g.edge_endpoints(e1).unwrap().1;
        g.add_bi_edge(extra, m1, EdgeType::MinToMin);
        // Now in_neighbours_bi(M1, Min).len() == 2 ≠ 1 → false
        assert!(!check_bubble_structure(&g, s, vec![e1, e2]));
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
        let es_m1 = g.edges_between(s, m1)[0];
        let es_m2 = g.edges_between(s, m2)[0];
        assert!(!check_bubble_structure(&g, s, vec![es_m1, es_m2]));
    }
}

fn check_backwards_path(
    ptgraph: &DbgGraph,
    vertex: NodeIndex,
    mut ty: CarryType,
    output_vec: &mut Vec<NodeIndex>,
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
