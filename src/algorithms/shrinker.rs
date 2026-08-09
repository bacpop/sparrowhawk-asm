//! Shrink the given graph
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
    /// Returns index of the shrinked path represented by edge.
    ///
    /// `diag` is `(pass, shrinks_this_pass)`, diagnostic-only context printed on the
    /// invariant panics: a panic with `shrinks_this_pass == 0` on pass 1 means the graph
    /// was already corrupt before any shrink mutated it.
    fn shrink_single_path(
        &mut self,
        start_node: Self::NodeIdx,
        mid_node: Self::NodeIdx,
        ambnodes: &BTreeSet<NodeId>,
        currtype: EdgeType,
        diag: (usize, usize),
    );
}

impl Shrinkable for DbgGraph {
    type EdgeIdx = EdgeId;
    type NodeIdx = NodeId;

    fn shrink(&mut self) -> bool {
        // Shrinkage here means to only find consecutive nodes, w/o bifurcations

        let mut dididoanything = false;
        let mut pass = 0usize;
        loop {
            let mut dididoanythingnow = false;
            pass += 1;
            let mut shrinks_this_pass = 0usize;
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
                    // This means that this is the outermost k-mer of a dead-end, that might also have self-loops.
                    // Let's check for self-loops first:
                    if self.node_has_self_loops(*an) {
                        continue;
                    }

                    let tmpty = neigh[0].1.get_from_and_to().1;
                    let outn = self.out_neighbours_bi(neigh[0].0, tmpty);
                    if outn.len() == 1 && (outn[0].0 == *an || ambnodes.contains(&outn[0].0)) {
                        continue;
                    } else if outn.len() <= 1 && self.in_neighbours_bi(neigh[0].0, tmpty).len() == 1
                    {
                        self.shrink_single_path(
                            *an,
                            neigh[0].0,
                            &ambnodes,
                            neigh[0].1,
                            (pass, shrinks_this_pass),
                        );
                        shrinks_this_pass += 1;
                        dididoanything = true;
                        dididoanythingnow = true;
                    }
                } else {
                    for n in neigh {
                        // =================== DEBUG: passive staleness detection. `neigh` was
                        // computed before this loop, but earlier shrinks in the same iteration
                        // mutate the graph; log (without changing the flow) when a cached entry
                        // no longer matches reality.
                        if !self.contains_node(n.0) {
                            log::warn!(
                                "[audit] stale neighbour: {:?} of ambiguous {:?} no longer exists (pass {pass}, shrinks {shrinks_this_pass})",
                                n.0,
                                an
                            );
                        } else if !self
                            .outgoing_edges(*an)
                            .iter()
                            .any(|&(m, t)| m == n.0 && t == n.1)
                        {
                            log::warn!(
                                "[audit] stale edge: cached {:?} -{:?}-> {:?} is gone or retyped (pass {pass}, shrinks {shrinks_this_pass})",
                                an,
                                n.1,
                                n.0
                            );
                        }
                        // =================== DEBUG

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
                                    // =================== DEBUG
                                    if ambnodes.contains(&outn[0].0) {
                                        log::warn!(
                                            "[audit] escape clause: shrinking into stale-ambiguous {:?} (an {:?} now has 1 good neighbour; pass {pass}, shrinks {shrinks_this_pass})",
                                            outn[0].0,
                                            an
                                        );
                                    }
                                    // =================== DEBUG
                                    self.shrink_single_path(
                                        n.0,
                                        outn[0].0,
                                        &ambnodes,
                                        outn[0].1,
                                        (pass, shrinks_this_pass),
                                    );
                                    shrinks_this_pass += 1;
                                    dididoanything = true;
                                    dididoanythingnow = true;
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
        diag: (usize, usize),
    ) {
        let mut countsformean: Vec<u32> = vec![self.node_weight(base_node).unwrap().counts];

        let (initty, mut currtype) = curredge.get_from_and_to();

        log::trace!("Starting shrinkage!");
        if self.out_degree_bi(next_node, currtype) == 0 {
            // Only the base_node + next_node shrinkage is possible
            log::trace!("No one to continue already from the next_node, before looping");

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

            // =================== DEBUG
            if self.in_degree(base_node) != 0 || self.out_degree(base_node) != 0 {
                panic!(
                    "Invariant broken after dead-end merge (pass {}, shrinks so far {}): \
                     base_node {:?} should be isolated but has in/out {}/{} \
                     (min-out {}, max-out {}, min-in {}, max-in {}). \
                     incoming {:?}, outgoing {:?}, kmers {:?}. \
                     removed next_node {:?} (in stale ambnodes: {}), \
                     curredge {:?}, initty {:?}, currtype {:?}, ind {:?}",
                    diag.0,
                    diag.1,
                    base_node,
                    self.in_degree(base_node),
                    self.out_degree(base_node),
                    self.out_degree_min(base_node),
                    self.out_degree_max(base_node),
                    self.in_degree_min(base_node),
                    self.in_degree_max(base_node),
                    self.incoming_edges(base_node),
                    self.outgoing_edges(base_node),
                    self.node_weight(base_node).map(|w| &w.abs_ind),
                    next_node,
                    ambnodes.contains(&next_node),
                    curredge,
                    initty,
                    currtype,
                    ind,
                );
            }
            // =================== DEBUG

            return;
        }

        // As next_node has exactly one outgoing neighbour, we can start the main loop
        // =================== DEBUG
        if self.in_degree(next_node) != 2 || self.out_degree(next_node) != 2 {
            panic!(
                "Pairing invariant broken entering main shrink loop (pass {}, shrinks so far {}): \
                 next_node {:?} in/out {}/{} (expected 2/2; min-out {}, max-out {}, min-in {}, max-in {}), \
                 in stale ambnodes: {}. incoming {:?}, outgoing {:?}, kmers {:?}. \
                 base_node {:?} in/out {}/{}, in stale ambnodes: {}, kmers {:?}. \
                 curredge {:?}, initty {:?}, currtype {:?}",
                diag.0,
                diag.1,
                next_node,
                self.in_degree(next_node),
                self.out_degree(next_node),
                self.out_degree_min(next_node),
                self.out_degree_max(next_node),
                self.in_degree_min(next_node),
                self.in_degree_max(next_node),
                ambnodes.contains(&next_node),
                self.incoming_edges(next_node),
                self.outgoing_edges(next_node),
                self.node_weight(next_node).map(|w| &w.abs_ind),
                base_node,
                self.in_degree(base_node),
                self.out_degree(base_node),
                ambnodes.contains(&base_node),
                self.node_weight(base_node).map(|w| &w.abs_ind),
                curredge,
                initty,
                currtype,
            );
        }
        // =================== DEBUG

        loop {
            let prospective_node = self.out_neighbours_bi(next_node, currtype)[0];

            // First, avoid self-loops.
            if prospective_node.0 == base_node {
                panic!(
                    "Shrink walk looped back to its base (pass {}, shrinks so far {}): \
                     prospective {:?} ({:?}) == base_node. base in/out {}/{}, \
                     incoming {:?}, outgoing {:?}, kmers {:?}. \
                     next_node {:?} in/out {}/{} (in stale ambnodes: {}), kmers {:?}. \
                     curredge {:?}, initty {:?}, currtype {:?}",
                    diag.0,
                    diag.1,
                    prospective_node.0,
                    prospective_node.1,
                    self.in_degree(base_node),
                    self.out_degree(base_node),
                    self.incoming_edges(base_node),
                    self.outgoing_edges(base_node),
                    self.node_weight(base_node).map(|w| &w.abs_ind),
                    next_node,
                    self.in_degree(next_node),
                    self.out_degree(next_node),
                    ambnodes.contains(&next_node),
                    self.node_weight(next_node).map(|w| &w.abs_ind),
                    curredge,
                    initty,
                    currtype,
                );
            }

            countsformean.push(self.node_weight(next_node).unwrap().counts);
            let next_base_weight = self.remove_node(next_node).unwrap();
            let prospfromandto = prospective_node.1.get_from_and_to();
            currtype = prospfromandto.0;

            self.node_weight_mut(base_node)
                .unwrap()
                .merge(&next_base_weight, curredge);

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

                // =================== DEBUG
                if self.out_degree(base_node) != 1 && !ind.is_empty() {
                    panic!(
                        "Invariant broken after merging an external terminus (EY2; pass {}, shrinks so far {}): \
                         base_node {:?} out_degree {} != 1 (in_degree {}; min-out {}, max-out {}, min-in {}, max-in {}). \
                         incoming {:?}, outgoing {:?}, kmers {:?}, internal edge {:?}. \
                         merged external {:?} (in stale ambnodes: {}), \
                         curredge {:?}, initty {:?}, currtype {:?}, ind {:?}",
                        diag.0,
                        diag.1,
                        base_node,
                        self.out_degree(base_node),
                        self.in_degree(base_node),
                        self.out_degree_min(base_node),
                        self.out_degree_max(base_node),
                        self.in_degree_min(base_node),
                        self.in_degree_max(base_node),
                        self.incoming_edges(base_node),
                        self.outgoing_edges(base_node),
                        self.node_weight(base_node).map(|w| &w.abs_ind),
                        self.node_weight(base_node).and_then(|w| w.innerdir),
                        prospective_node.0,
                        ambnodes.contains(&prospective_node.0),
                        curredge,
                        initty,
                        currtype,
                        ind,
                    );
                }
                // =================== DEBUG

                return;
            } else if ambnodes.contains(&prospective_node.0) {
                // We cannot add prospective_node, because it is an ambiguous node.
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

                // =================== DEBUG
                if self.out_degree_min(base_node) != 1 && !ind.is_empty()
                    || self.out_degree(base_node) != 1 && ind.is_empty()
                {
                    panic!(
                        "Invariant broken after reconnecting to an ambiguous node (EY1; pass {}, shrinks so far {}): \
                         base_node {:?} in/out {}/{} (min-out {}, max-out {}, min-in {}, max-in {}), \
                         internal edge {:?}, kmers {:?}. incoming {:?}, outgoing {:?}. \
                         incoming node from before: {:?} (ind {:?}). \
                         prospective_node {:?} ({:?}) in/out {}/{} (in stale ambnodes: {}), kmers {:?}. \
                         newoutedge {:?}, curredge {:?}, initty {:?}, currtype {:?}",
                        diag.0,
                        diag.1,
                        base_node,
                        self.in_degree(base_node),
                        self.out_degree(base_node),
                        self.out_degree_min(base_node),
                        self.out_degree_max(base_node),
                        self.in_degree_min(base_node),
                        self.in_degree_max(base_node),
                        self.node_weight(base_node).and_then(|w| w.innerdir),
                        self.node_weight(base_node).map(|w| &w.abs_ind),
                        self.incoming_edges(base_node),
                        self.outgoing_edges(base_node),
                        ind.first().map(|e| e.0),
                        ind,
                        prospective_node.0,
                        prospective_node.1,
                        self.in_degree(prospective_node.0),
                        self.out_degree(prospective_node.0),
                        ambnodes.contains(&prospective_node.0),
                        self.node_weight(prospective_node.0).map(|w| &w.abs_ind),
                        newoutedge,
                        curredge,
                        initty,
                        currtype,
                    );
                }
                // =================== DEBUG

                return;
            } else {
                // If we are here, that means we can continue shrinking!
                // =================== DEBUG
                if self.in_degree(prospective_node.0) != 1
                    || self.out_degree(prospective_node.0) != 1
                {
                    panic!(
                        "Pairing invariant broken mid-walk (pass {}, shrinks so far {}): \
                         prospective_node {:?} ({:?}) in/out {}/{} (expected 1/1 after removing its \
                         predecessor; min-out {}, max-out {}, min-in {}, max-in {}), \
                         in stale ambnodes: {}. incoming {:?}, outgoing {:?}, kmers {:?}. \
                         base_node {:?} in/out {}/{}, kmers {:?}. removed next_node {:?}. \
                         curredge {:?}, initty {:?}, currtype {:?}",
                        diag.0,
                        diag.1,
                        prospective_node.0,
                        prospective_node.1,
                        self.in_degree(prospective_node.0),
                        self.out_degree(prospective_node.0),
                        self.out_degree_min(prospective_node.0),
                        self.out_degree_max(prospective_node.0),
                        self.in_degree_min(prospective_node.0),
                        self.in_degree_max(prospective_node.0),
                        ambnodes.contains(&prospective_node.0),
                        self.incoming_edges(prospective_node.0),
                        self.outgoing_edges(prospective_node.0),
                        self.node_weight(prospective_node.0).map(|w| &w.abs_ind),
                        base_node,
                        self.in_degree(base_node),
                        self.out_degree(base_node),
                        self.node_weight(base_node).map(|w| &w.abs_ind),
                        next_node,
                        curredge,
                        initty,
                        currtype,
                    );
                }
                // =================== DEBUG

                // Updating next_node and currtype
                next_node = prospective_node.0;
                currtype = prospective_node.1.get_from_and_to().1;
                curredge = EdgeType::from_carrytypes(initty, currtype);
            }
        }
    }
}
