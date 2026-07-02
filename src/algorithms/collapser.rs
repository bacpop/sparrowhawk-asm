//! Create string representation of contigs out of `DbgGraph`.

use super::shrinker::Shrinkable;
use sparrowhawk_graph::{CarryType, DbgGraph, NodeIndex, NodeStruct, SerializedContigs};
use std::cmp::max;

use petgraph;
use petgraph::algo::{connected_components, tarjan_scc};
use petgraph::visit::EdgeRef;
use petgraph::EdgeDirection;

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
            connected_components(&petgraph::graph::Graph::from(self.inner_graph().clone())),
            self.node_indices()
                .filter(|n| self
                    .inner_graph()
                    .neighbors_directed(*n, EdgeDirection::Outgoing)
                    .count()
                    == 0
                    && self
                        .inner_graph()
                        .neighbors_directed(*n, EdgeDirection::Incoming)
                        .count()
                        == 0)
                .count()
        );

        log::info!("Starting collapse loop.");
        let minnts = 100;
        let limit = max(0, minnts - self.k() + 1);

        loop {
            loop {
                let externals = self.externals_bi();
                log::debug!("\t- Loop over {} external nodes.", externals.len());
                if externals.is_empty() {
                    break;
                }
                for n in externals {
                    if self.contains_node(n) {
                        if self.get_good_connections_degree(n) == 0 {
                            log::debug!("\t\t# Isolated node.");
                            let thecont = vec![self.node_weight(n).unwrap().clone()];
                            if get_contig_length(&thecont) > limit {
                                contigs.push(thecont);
                            }
                            self.remove_node(n);
                        } else {
                            stacker::maybe_grow(32 * 1024, 1024 * 1024, || {
                                let contigs_ = contigs_from_vertex(&mut self, n);
                                contigs.extend(
                                    contigs_
                                        .into_iter()
                                        .filter(|c| get_contig_length(c) > limit)
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
                log::debug!(
                    "\t\t# {} nodes remain. Starting to build from middle node.",
                    tmpc
                );

                stacker::grow(100 * 1024 * 1024, || {
                    let sccvec: Vec<Vec<NodeIndex>> = tarjan_scc(self.inner_graph());
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
                            .filter(|c| get_contig_length(c) > limit)
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

fn get_contig_length(vec: &[NodeStruct]) -> usize {
    vec.iter().map(|ns| ns.abs_ind.len()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparrowhawk_graph::NodeStruct;

    fn make_node(len: usize) -> NodeStruct {
        NodeStruct {
            counts: 1,
            abs_ind: vec![0u64; len],
            innerdir: None,
        }
    }

    #[test]
    fn get_contig_length_empty() {
        assert_eq!(get_contig_length(&[]), 0);
    }

    #[test]
    fn get_contig_length_single_node_no_kmers() {
        assert_eq!(get_contig_length(&[make_node(0)]), 0);
    }

    #[test]
    fn get_contig_length_single_node_five() {
        assert_eq!(get_contig_length(&[make_node(5)]), 5);
    }

    #[test]
    fn get_contig_length_multiple_nodes() {
        let nodes = vec![make_node(3), make_node(0), make_node(7)];
        assert_eq!(get_contig_length(&nodes), 10);
    }

    #[test]
    fn get_contig_length_large() {
        let nodes: Vec<_> = (0..100).map(|_| make_node(50)).collect();
        assert_eq!(get_contig_length(&nodes), 5000);
    }
}

#[inline]
fn contigs_from_vertex(ptgraph: &mut DbgGraph, v: NodeIndex) -> SerializedContigs {
    let mut contigs: SerializedContigs = vec![];
    let mut contig: Vec<NodeStruct> = vec![];
    let mut current_vertex = v;
    let mut target;
    let outeds: Vec<_> = ptgraph
        .inner_graph()
        .edges_directed(v, EdgeDirection::Outgoing)
        .map(|e| e.id())
        .collect();
    let mut current_type = ptgraph
        .inner_graph()
        .edge_weight(outeds[0])
        .unwrap()
        .t
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
            contigs.push(contig.clone());
            contig.clear();

            if num_following == 0 {
                ptgraph.remove_node(current_vertex);
                return contigs;
            }

            ptgraph.remove_node(current_vertex);
            return contigs;
        }

        let mut nwtocopy = ptgraph.node_weight(current_vertex).unwrap().clone();
        if let Some(innvtx) = nwtocopy.innerdir {
            if current_type != innvtx.get_from_and_to().0 {
                nwtocopy.abs_ind.reverse();
            }
        }

        contig.push(nwtocopy);

        target = outneighs[0].0;

        if current_vertex == target {
            panic!("FATAL: continuing to the same vertex!")
        };

        current_type = outneighs[0].1.get_from_and_to().1;
        ptgraph.remove_node(current_vertex);
        current_vertex = target;
        num_preceding = ptgraph.in_degree_bi(current_vertex, current_type);
        outneighs = ptgraph.out_neighbours_bi(current_vertex, current_type);
        num_following = outneighs.len();
    }
}

#[inline]
fn contigs_from_intermediate_vertex(ptgraph: &mut DbgGraph, v: NodeIndex) -> SerializedContigs {
    let mut contigs: SerializedContigs = vec![];
    let mut contig: Vec<NodeStruct> = vec![];
    let mut current_vertex = v;
    let mut target;

    let outeds;
    let mut outmin = Vec::new();
    let mut outmax = Vec::new();

    for e in ptgraph
        .inner_graph()
        .edges_directed(v, EdgeDirection::Outgoing)
    {
        if e.weight().t.get_from_and_to().0 == CarryType::Min {
            outmin.push(e.id());
        } else {
            outmax.push(e.id());
        }
    }
    let outminlen = outmin.len();
    let outmaxlen = outmax.len();
    match (outminlen, outmaxlen) {
        (0, 0) | (0, 1) | (1, 0) => panic!("External node!!!"),
        (1, 1) | (1, _) => outeds = outmin,
        (_, 1) => outeds = outmax,
        (_, _) => {
            if outminlen <= outmaxlen {
                outeds = outmin;
            } else {
                outeds = outmax;
            }
        }
    }

    let mut current_type = ptgraph
        .inner_graph()
        .edge_weight(outeds[0])
        .unwrap()
        .t
        .get_from_and_to()
        .0;
    let mut outneighs = ptgraph.out_neighbours_bi(v, current_type);
    let mut num_following = outneighs.len();
    let mut num_preceding = 0;

    loop {
        if num_following == 1 && num_preceding == 0 {
            // Continue
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
            contigs.push(contig.clone());
            contig.clear();

            if num_following == 0 {
                return contigs;
            }
        }

        let mut nwtocopy = ptgraph.node_weight(current_vertex).unwrap().clone();
        if let Some(innvtx) = nwtocopy.innerdir {
            if current_type != innvtx.get_from_and_to().0 {
                nwtocopy.abs_ind.reverse();
            }
        }

        contig.push(nwtocopy);

        target = outneighs[0].0;

        if current_vertex == target {
            panic!("FATAL: continuing to the same vertex!")
        };

        current_type = outneighs[0].1.get_from_and_to().1;
        ptgraph.remove_node(current_vertex);
        current_vertex = target;
        num_preceding = ptgraph.in_degree_bi(current_vertex, current_type);
        outneighs = ptgraph.out_neighbours_bi(current_vertex, current_type);
        num_following = outneighs.len();
    }
}
