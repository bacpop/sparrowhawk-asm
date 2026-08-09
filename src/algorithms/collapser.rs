//! Create string representation of contigs out of `DbgGraph`.

use super::corrector::prune_unpaired_edges;
use super::shrinker::Shrinkable;
use sparrowhawk_graph::{
    get_nodelist_kmer_length, CarryType, DbgGraph, NodeId, NodeStruct, SerializedContigs,
};

/// Collapse `DbgGraph` into `SerializedContigs`.
pub trait Collapsable: Shrinkable {
    /// Collapses into `SerializedContigs`.
    fn collapse(self) -> SerializedContigs;
}

impl Collapsable for DbgGraph {
    fn collapse(mut self) -> SerializedContigs {
        let mut contigs: SerializedContigs = vec![];

        log::info!("Removing self-loops (temporal restriction)");
        self.remove_self_loops();

        log::info!(
            "Graph has {} weakly connected component(s), among which {} are single nodes.",
            self.connected_components(),
            self.isolated_node_count()
        );

        log::info!("Starting collapse loop.");
        // 100 nt, though independent of this value the minimum is always at least k.
        let limit = crate::algorithms::corrector::short_path_limit(100, self.k());

        loop {
            loop {
                // get all starting nodes, i.e. nodes with in_degree == 0
                let externals = self.externals_bi();
                log::debug!("\t- Loop over {} external nodes.", externals.len());
                if externals.is_empty() {
                    break;
                }
                // create contigs from each starting node
                for n in externals {
                    // We need first to take care of perfect contigs, almost-already provided as such. These
                    // are seen as nodes with no incoming/outcoming edges.
                    if self.contains_node(n) {
                        if self.get_good_connections_degree(n) == 0 {
                            log::debug!("\t\t# Isolated node.");
                            let thecont = vec![self.node_weight(n).unwrap().clone()];
                            if get_nodelist_kmer_length(&thecont) > limit {
                                contigs.push(thecont);
                            }
                            self.remove_node(n);
                        } else {
                            stacker::maybe_grow(32 * 1024, 1024 * 1024, || {
                                let contigs_ = contigs_from_vertex(&mut self, n);
                                contigs.extend(
                                    contigs_
                                        .into_iter()
                                        .filter(|c| get_nodelist_kmer_length(c) > limit)
                                        .collect::<Vec<_>>(),
                                );
                            });
                        }
                    }
                }
            }
            log::debug!("\t- Loops over external nodes finished.");

            let tmpc = self.node_count();
            if tmpc != 0 {
                // we guarantee that there's at least one node to unwrap here
                log::debug!(
                    "\t\t# {} nodes remain. Starting to build from middle node.",
                    tmpc
                );

                // This call to stacker::grow here is needed because of the algorithm that is run to obtain the
                // strongly-connected components. It is recursive, so in very entangled graphs (and/or when k is
                // low, i.e. k ~< 15), it might lead to a stack overflow.
                stacker::grow(100 * 1024 * 1024, || {
                    let sccvec: Vec<Vec<NodeId>> = self.strongly_connected_components();
                    let node_in_cycle = sccvec[0].last().unwrap();

                    log::debug!(
                        "\t\t# Remaining nodes {}, remaining SCCs {}, starting with {} neighbours",
                        tmpc,
                        sccvec.len(),
                        self.get_good_connections_degree(*node_in_cycle)
                    );

                    let thecontigs = contigs_from_intermediate_vertex(&mut self, *node_in_cycle);

                    contigs.extend(
                        thecontigs
                            .into_iter()
                            .filter(|c| get_nodelist_kmer_length(c) > limit)
                            .collect::<Vec<_>>(),
                    );
                });
                log::debug!("\t\t# Finished creating one contig from starting circle.");
            } else {
                break;
            }
        }

        log::trace!(
            "{} nodes left in the graph after collapse",
            self.node_count()
        );
        log::info!("Collapse ended. Created {} contigs", contigs.len());

        contigs
    }
}

// Main collapse function/method
#[inline]
fn contigs_from_vertex(ptgraph: &mut DbgGraph, v: NodeId) -> SerializedContigs {
    let mut contigs: SerializedContigs = vec![];
    let mut contig: Vec<NodeStruct> = vec![];
    let mut current_vertex = v;
    let mut target;
    let mut current_type = ptgraph
        .first_outgoing_edge_type(v)
        .unwrap()
        .get_from_and_to()
        .0;
    let mut outneighs = ptgraph.out_neighbours_bi(v, current_type);
    let mut num_following = outneighs.len();
    let mut num_preceding = 0;

    loop {
        if num_following == 1 && num_preceding == 0 {
            // Ok, so we can continue, let's go!
        } else if num_following == 0 && num_preceding == 0 {
            // We're finishing!!
            let mut nwtocopy = ptgraph.node_weight(current_vertex).unwrap().clone();
            if let Some(innvtx) = nwtocopy.innerdir {
                if current_type != innvtx.get_from_and_to().0 {
                    nwtocopy.abs_ind.reverse();
                }
            }

            contig.push(nwtocopy);
            contigs.push(contig.clone());
            contig.clear();
            ptgraph.remove_node(current_vertex);
            return contigs;
        } else {
            // We've found an ambiguous node/bifurcation, thus we need to stop the current contig and clear the vector
            contigs.push(contig.clone());
            contig.clear();

            // And now what we do depends on the neighbours from this new vertex. OR NOT: LET'S FINISH FOR NOW!
            if num_following == 0 {
                // We cannot continue.
                ptgraph.remove_node(current_vertex);
                return contigs;
            }

            ptgraph.remove_node(current_vertex);
            return contigs;
        }

        // If we arrived here, current_vertex is either considered good to be added to the current
        // contig, or we have created a contig break and we are starting from this ambiguous node
        // and also current_edge_index is the vertex through which we should continue our
        // journey, or we have either a simple loop or a circumference to deal with

        // We add the current_vertex to the contig
        let mut nwtocopy = ptgraph.node_weight(current_vertex).unwrap().clone();
        if let Some(innvtx) = nwtocopy.innerdir {
            if current_type != innvtx.get_from_and_to().0 {
                nwtocopy.abs_ind.reverse();
            }
        }

        contig.push(nwtocopy);

        // We get our next soon-to-be current_vertex (now named "target")
        target = outneighs[0].0;

        if current_vertex == target {
            panic!("FATAL: continuing to the same vertex!")
        };

        // We update the variables and get ready to do another iteration!
        current_type = outneighs[0].1.get_from_and_to().1;
        ptgraph.remove_node(current_vertex);
        current_vertex = target;
        num_preceding = ptgraph.in_degree_bi(current_vertex, current_type);
        outneighs = ptgraph.out_neighbours_bi(current_vertex, current_type);
        num_following = outneighs.len();
    }
}

#[inline]
fn contigs_from_intermediate_vertex(ptgraph: &mut DbgGraph, v: NodeId) -> SerializedContigs {
    let mut contigs: SerializedContigs = vec![];
    let mut contig: Vec<NodeStruct> = vec![];
    let mut current_vertex = v;
    let mut target;

    // We need to get the carrytype, the edges, and so on before we can begin. We'll try to set them to get a forward
    // direction with only one neighbour, if possible.
    // Pairing forbids one-sided patterns here: prune collision leftovers and re-read.
    let mut outmin = ptgraph.outgoing_edges_by_carry(v, CarryType::Min);
    let mut outmax = ptgraph.outgoing_edges_by_carry(v, CarryType::Max);
    if outmin.is_empty() || outmax.is_empty() {
        prune_unpaired_edges(ptgraph, v);
        outmin = ptgraph.outgoing_edges_by_carry(v, CarryType::Min);
        outmax = ptgraph.outgoing_edges_by_carry(v, CarryType::Max);
    }
    let outminlen = outmin.len();
    let outmaxlen = outmax.len();
    let outeds = match (outminlen, outmaxlen) {
        (0, 0) => {
            // Nothing left after the repair: emit the node as its own contig.
            contig.push(ptgraph.node_weight(v).unwrap().clone());
            contigs.push(contig);
            ptgraph.remove_node(v);
            return contigs;
        }
        (_, 0) => outmin,
        (0, _) => outmax,
        (1, _) => outmin, // We select the minimum outgoing edges
        (_, 1) => outmax, // We select the maximum outgoing edges
        (_, _) => {
            // We check whether they are the same and, if not, we select the first id from the minimum (this is clearly improvable)
            if outminlen <= outmaxlen {
                outmin
            } else {
                outmax
            }
        }
    };

    let mut current_type = outeds[0].2.get_from_and_to().0;
    let mut outneighs = ptgraph.out_neighbours_bi(v, current_type);
    let mut num_following = outneighs.len();
    let mut num_preceding = 0; /////// This is strictly speaking always false here, but it is only for the first iteration.
                               // Afterwards, we respect its true value to decide whether we stop or not the contig formation.

    loop {
        if num_following == 1 && num_preceding == 0 {
            // Ok, so we can continue, let's go!
        } else if num_following == 0 && num_preceding == 0 {
            let mut nwtocopy = ptgraph.node_weight(current_vertex).unwrap().clone();
            if let Some(innvtx) = nwtocopy.innerdir {
                if current_type != innvtx.get_from_and_to().0 {
                    nwtocopy.abs_ind.reverse();
                }
            }

            contig.push(nwtocopy);
            contigs.push(contig.clone());
            contig.clear();
            ptgraph.remove_node(current_vertex);
            return contigs;
        } else {
            // We've found an ambiguous node/bifurcation, thus we need to stop the current contig and clear the vector
            contigs.push(contig.clone());
            contig.clear();

            // And now what we do depends on the neighbours from this new vertex. OR NOT: LET'S FINISH FOR NOW!
            if num_following == 0 {
                // We cannot continue.
                return contigs;
            }
        }

        // If we arrived here, current_vertex is either considered good to be added to the current
        // contig, or we have created a contig break and we are starting from this ambiguous node
        // and also current_edge_index is the vertex through which we should continue our
        // journey, or we have either a simple loop or a circumference to deal with

        // We add the current_vertex to the contig
        let mut nwtocopy = ptgraph.node_weight(current_vertex).unwrap().clone();
        if let Some(innvtx) = nwtocopy.innerdir {
            if current_type != innvtx.get_from_and_to().0 {
                nwtocopy.abs_ind.reverse();
            }
        }

        contig.push(nwtocopy);

        // We get our next soon-to-be current_vertex (now named "target")
        target = outneighs[0].0;

        if current_vertex == target {
            panic!("FATAL: continuing to the same vertex!")
        };

        // We update the variables and get ready to do another iteration!
        current_type = outneighs[0].1.get_from_and_to().1;
        ptgraph.remove_node(current_vertex);
        current_vertex = target;
        num_preceding = ptgraph.in_degree_bi(current_vertex, current_type);
        outneighs = ptgraph.out_neighbours_bi(current_vertex, current_type);
        num_following = outneighs.len();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparrowhawk_graph::EdgeType;

    fn node(h: u64) -> NodeStruct {
        NodeStruct {
            counts: 10,
            abs_ind: vec![h],
            innerdir: None,
        }
    }

    /// A one-sided start (which used to panic as "External node!!!") is repaired and
    /// walks its available side.
    #[test]
    fn collapse_walks_a_one_sided_start_instead_of_panicking() {
        let mut g = DbgGraph::new(3);
        let v = g.add_node(node(0));
        let w = g.add_node(node(1));
        let u = g.add_node(node(2));
        g.add_bi_edge(v, w, EdgeType::MinToMin);
        g.add_edge(u, v, EdgeType::MinToMax); // phantom: no reverse partner

        let contigs = contigs_from_intermediate_vertex(&mut g, v);

        assert_eq!(contigs.len(), 1);
        assert_eq!(contigs[0].len(), 2);
        assert_eq!(g.node_count(), 1); // only `u` is left...
        assert_eq!(g.out_degree(u), 0); // ...and its phantom edge was pruned
    }

    /// A start left with no edges at all becomes its own single-node contig.
    #[test]
    fn a_disconnected_start_becomes_its_own_contig() {
        let mut g = DbgGraph::new(3);
        let v = g.add_node(node(0));

        let contigs = contigs_from_intermediate_vertex(&mut g, v);

        assert_eq!(contigs.len(), 1);
        assert_eq!(contigs[0].len(), 1);
        assert_eq!(g.node_count(), 0);
    }
}
