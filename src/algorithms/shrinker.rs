//! Shrink the given graph
use crate::algorithms::corrector::prune_unpaired_edges;
use crate::logw;
use sparrowhawk_graph::{CarryType, DbgGraph, EdgeId, EdgeType, NodeId};

use std::collections::BTreeSet;

/// Mark graph as shrinkable.
pub trait Shrinkable {
    /// Edge index associated with collection.
    type EdgeIdx;
    /// Node index associated with collection.
    type NodeIdx;
    /// Shrink graph.
    ///
    /// This operation should shrink all straight paths
    /// It is assumed that after shrinking graph will not have any nodes
    /// connected in this way: s -> x -> ... -> t
    fn shrink(&mut self) -> bool;

    /// Shrink one single path. This method assumes that `base_edge` argument points
    /// to a valid edge, which target has a single outgoing edge.
    /// Returns whether anything was actually shrunk.
    fn shrink_single_path(
        &mut self,
        start_node: Self::NodeIdx,
        mid_node: Self::NodeIdx,
        ambnodes: &BTreeSet<NodeId>,
        currtype: EdgeType,
    ) -> bool;
}

impl Shrinkable for DbgGraph {
    type EdgeIdx = EdgeId;
    type NodeIdx = NodeId;

    fn shrink(&mut self) -> bool {
        // Shrinkage here means to only find consecutive nodes, w/o bifurcations

        let mut dididoanything = false;
        loop {
            let mut dididoanythingnow = false;
            let ambnodes = self.get_ambiguous_nodes_bi(); // Just in case we hadn't got them yet
            logw(format!("Starting shrinking the graph with {} nodes and {} edges, beginning from {} ambiguous nodes",
                  self.node_count(),
                  self.edge_count(),
                  ambnodes.len()).as_str(), Some("info"));
            for an in ambnodes.iter() {
                if !self.contains_node(*an) {
                    continue;
                }

                let neigh = self.get_good_neighbours_bi(*an);
                let conns = neigh.len();

                if conns == 1 {
                    // Self-loop nodes are ambiguity boundaries. Their loop edges are excluded
                    // from ordinary neighbours, so do not start a contraction from this node.
                    if self.node_has_self_loops(*an) {
                        continue;
                    }

                    let tmpty = neigh[0].1.get_from_and_to().1;
                    let outn = self.out_neighbours_bi(neigh[0].0, tmpty);
                    if outn.len() == 1 && (outn[0].0 == *an || ambnodes.contains(&outn[0].0)) {
                        continue;
                    } else if outn.len() <= 1 && self.in_neighbours_bi(neigh[0].0, tmpty).len() == 1
                    {
                        if self.shrink_single_path(*an, neigh[0].0, &ambnodes, neigh[0].1) {
                            dididoanything = true;
                            dididoanythingnow = true;
                        }
                    }
                } else {
                    for n in neigh {
                        if ambnodes.contains(&n.0) || n.0 == *an {
                            continue;
                        } else {
                            let tmpty = n.1.get_from_and_to().1;
                            let outn = self.out_neighbours_bi(n.0, tmpty);

                            if outn.is_empty() {
                                continue;
                            }

                            if outn.len() == 1
                                && outn[0].0 != n.0
                                && outn[0].0 != *an
                                && (!ambnodes.contains(&outn[0].0)
                                    || self.get_good_neighbours_bi(*an).len() == 1)
                            {
                                let incn = self.in_neighbours_bi(n.0, tmpty);
                                if incn.len() == 1 {
                                    if self.shrink_single_path(n.0, outn[0].0, &ambnodes, outn[0].1)
                                    {
                                        dididoanything = true;
                                        dididoanythingnow = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if !dididoanythingnow {
                break;
            }
        }

        log::info!(
            "Shrinking ended. Shrunk graph has {} nodes and {} edges",
            self.node_count(),
            self.edge_count()
        );

        dididoanything
    }

    #[inline]
    fn shrink_single_path(
        &mut self,
        base_node: NodeId,
        mut next_node: NodeId,
        ambnodes: &BTreeSet<NodeId>,
        mut curredge: EdgeType,
    ) -> bool {
        let (initty, mut currtype) = curredge.get_from_and_to();

        // Pairing is checked by reciprocal edge type, not aggregate degree.
        let mut pruned = prune_unpaired_edges(self, base_node) > 0;
        pruned |= prune_unpaired_edges(self, next_node) > 0;
        // If the connecting edge was itself the phantom, there is nothing left to shrink here.
        if pruned
            && !self
                .out_neighbours_bi(base_node, initty)
                .iter()
                .any(|&(m, t)| m == next_node && t == curredge)
        {
            return false;
        }

        let mut countsformean: Vec<u32> = vec![self.node_weight(base_node).unwrap().counts];

        log::trace!("Starting shrinkage!");
        if self.out_degree_bi(next_node, currtype) == 0 {
            // Only the base_node + next_node shrinkage is possible
            log::trace!("No one to continue already from the next_node, before looping");

            // After repair a terminal next_node must have its single paired adjacency to base_node.
            if self.in_degree(next_node) != 1 || self.out_degree(next_node) != 1 {
                log::warn!(
                    "Not shrinking into {:?} (in/out {}/{}, expected 1/1)",
                    next_node,
                    self.in_degree(next_node),
                    self.out_degree(next_node)
                );
                return false;
            }

            countsformean.push(self.node_weight(next_node).unwrap().counts);
            let next_base_weight = self.remove_node(next_node).unwrap();
            let ind = self.in_neighbours_bi(base_node, initty);

            self.node_weight_mut(base_node)
                .unwrap()
                .merge(&next_base_weight, curredge);
            self.node_weight_mut(base_node)
                .unwrap()
                .set_mean_counts(&countsformean);

            log::trace!(
                "base_node {:?} next_node {:?} curredge {:?} initty {:?} currtype {:?} ind {:?}",
                base_node,
                next_node,
                curredge,
                initty,
                currtype,
                ind
            );

            // ======================================= CHANGING INTERNAL EDGES IF NEEDED BEGIN

            self.node_weight_mut(base_node)
                .unwrap()
                .invert_if_needed(curredge);

            if curredge.is_direct() {
                self.node_weight_mut(base_node)
                    .unwrap()
                    .set_internal_edge(curredge);
            } else if ind.is_empty() {
                // NOTE: now, this is set to MintoMin by default. IT IS A LIE
                self.node_weight_mut(base_node)
                    .unwrap()
                    .set_internal_edge(EdgeType::MinToMin);
            } else {
                log::trace!("Modifying edges with a non-direct edge at the beginning.");
                self.modify_edges_when_shrinking_between(base_node, ind[0].0, curredge, ind[0].1);
            }
            // ======================================= CHANGING INTERNAL EDGES IF NEEDED END

            return true;
        }

        // As next_node has exactly one outgoing neighbour, we can start the main loop.
        // A clean mid-path next_node has one paired adjacency on each side.
        if self.in_degree(next_node) != 2 || self.out_degree(next_node) != 2 {
            log::warn!(
                "Not shrinking into {:?} (in/out {}/{}, expected 2/2)",
                next_node,
                self.in_degree(next_node),
                self.out_degree(next_node)
            );
            return false;
        }

        loop {
            let prospective_node = self.out_neighbours_bi(next_node, currtype)[0];

            // First, avoid self-loops.
            if prospective_node.0 == base_node {
                panic!("This should not happen");
            }

            countsformean.push(self.node_weight(next_node).unwrap().counts);
            let next_base_weight = self.remove_node(next_node).unwrap();
            let prospfromandto = prospective_node.1.get_from_and_to();
            currtype = prospfromandto.0;

            self.node_weight_mut(base_node)
                .unwrap()
                .merge(&next_base_weight, curredge);

            // Remove any incident edge whose reciprocal edge has the wrong type or is absent.
            prune_unpaired_edges(self, prospective_node.0);

            if self.out_degree_bi(prospective_node.0, prospfromandto.1) == 0
                && self.in_degree_bi(prospective_node.0, prospfromandto.1) == 0
            {
                // Prospective_node is an ambiguous node, but only because it is an external where we can finish.
                countsformean.push(self.node_weight(prospective_node.0).unwrap().counts);
                let next_base_weight = self.remove_node(prospective_node.0).unwrap();
                let nw = self.node_weight_mut(base_node).unwrap();

                curredge = EdgeType::from_carrytypes(initty, prospfromandto.1);
                nw.merge(&next_base_weight, curredge);
                nw.set_mean_counts(&countsformean);

                // ======================================= CHANGING INTERNAL EDGES IF NEEDED BEGIN
                let ind = self.in_neighbours_bi(base_node, initty);

                self.node_weight_mut(base_node)
                    .unwrap()
                    .invert_if_needed(curredge);

                if curredge.is_direct() {
                    self.node_weight_mut(base_node)
                        .unwrap()
                        .set_internal_edge(curredge);
                } else if ind.is_empty() {
                    self.node_weight_mut(base_node)
                        .unwrap()
                        .set_internal_edge(EdgeType::MinToMin);
                } else {
                    log::trace!("Modifying edges with a non-direct edge in the loop to an ambiguous node that is an external");
                    self.modify_edges_when_shrinking_between(
                        base_node, ind[0].0, curredge, ind[0].1,
                    );
                }
                // ======================================= CHANGING INTERNAL EDGES IF NEEDED END

                return true;
            } else if ambnodes.contains(&prospective_node.0)
                || self.in_degree(prospective_node.0) != 1
                || self.out_degree(prospective_node.0) != 1
            {
                // We cannot add prospective_node: it is ambiguous, or it stayed anomalous
                // after the repair (e.g. a genuine junction reached through stale caller data).
                self.node_weight_mut(base_node)
                    .unwrap()
                    .set_mean_counts(&countsformean);

                // ======================================= CHANGING INTERNAL EDGES IF NEEDED BEGIN
                let ind = self.in_neighbours_bi(base_node, initty);
                curredge = EdgeType::from_carrytypes(initty, currtype);

                let newoutedge: EdgeType;

                self.node_weight_mut(base_node)
                    .unwrap()
                    .invert_if_needed(curredge);

                if curredge.is_direct() {
                    self.node_weight_mut(base_node)
                        .unwrap()
                        .set_internal_edge(curredge);
                    newoutedge = prospective_node.1;
                } else if ind.is_empty() {
                    match curredge {
                        EdgeType::MinToMax => {
                            newoutedge = EdgeType::from_carrytypes(
                                CarryType::Min,
                                prospective_node.1.get_from_and_to().1,
                            );
                            self.node_weight_mut(base_node)
                                .unwrap()
                                .set_internal_edge(EdgeType::MinToMin);
                        }
                        EdgeType::MaxToMin => {
                            newoutedge = EdgeType::from_carrytypes(
                                CarryType::Max,
                                prospective_node.1.get_from_and_to().1,
                            );
                            self.node_weight_mut(base_node)
                                .unwrap()
                                .set_internal_edge(EdgeType::MaxToMax);
                        }
                        _ => panic!("Value not expected"),
                    }
                } else {
                    newoutedge = prospective_node.1;
                    log::trace!("Modifying edges with a non-direct edge in the loop when reaching another ambiguous node that is not an external.");
                    self.modify_edges_when_shrinking_between(
                        base_node, ind[0].0, curredge, ind[0].1,
                    );
                }

                self.add_bi_edge(base_node, prospective_node.0, newoutedge);
                // ======================================= CHANGING INTERNAL EDGES IF NEEDED END

                return true;
            } else {
                // If we are here, that means we can continue shrinking!
                // Updating next_node and currtype
                next_node = prospective_node.0;
                currtype = prospective_node.1.get_from_and_to().1;
                curredge = EdgeType::from_carrytypes(initty, currtype);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparrowhawk_graph::NodeStruct;

    fn node(h: u64) -> NodeStruct {
        NodeStruct {
            counts: 10,
            abs_ind: vec![h],
            innerdir: None,
        }
    }

    /// A phantom (one-sided) incoming edge mid-path must be pruned, and the shrink must
    /// continue through the healed node instead of panicking.
    #[test]
    fn shrink_prunes_unpaired_edges_instead_of_panicking() {
        let mut g = DbgGraph::new(31);
        let an = g.add_node(node(0));
        let n1 = g.add_node(node(1));
        let n2 = g.add_node(node(2));
        let n3 = g.add_node(node(3));
        let extra = g.add_node(node(4));
        g.add_bi_edge(an, n1, EdgeType::MinToMin);
        g.add_bi_edge(n1, n2, EdgeType::MinToMin);
        g.add_bi_edge(n2, n3, EdgeType::MinToMin);
        g.add_edge(extra, n2, EdgeType::MinToMin); // phantom: no reverse partner

        assert!(g.shrink());

        assert_eq!(g.node_count(), 2); // the merged path plus the isolated `extra`
        assert_eq!(g.edge_count(), 0); // the phantom is gone too
        assert_eq!(g.node_weight(an).unwrap().abs_ind.len(), 4);
        assert!(g.contains_node(extra));
    }
}
