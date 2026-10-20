//! Create string representation of contigs out of `DbgGraph`.

use super::shrinker::Shrinkable;
use sparrowhawk_graph::{CarryType, DbgGraph, NodeId, NodeStruct, SerializedContigs};

/// Collapse `DbgGraph` into `SerializedContigs`.
pub trait Collapsable: Shrinkable {
    /// Collapses into `SerializedContigs`.
    fn collapse(self) -> SerializedContigs;
}

impl Collapsable for DbgGraph {
    fn collapse(mut self) -> SerializedContigs {
        log::info!("Removing self-loops (temporal restriction)");
        self.remove_self_loops();

        log::info!(
            "Graph has {} weakly connected component(s), among which {} are single nodes.",
            self.connected_components(),
            self.isolated_node_count()
        );

        // Every contig is kept, however short, and each junction is one of its own: a multi-k step
        // carries them all to the next k. Output length filters come later.
        let mut contigs = erase_junction_nodes(&mut self);
        log::info!(
            "Kept {} branching junction nodes as their own contigs",
            contigs.len()
        );

        log::info!("Starting collapse loop.");

        // Nodes walked from each cycle left after the path walks.
        let mut cycles: Vec<usize> = Vec::new();
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
                            contigs.push(vec![self.node_weight(n).unwrap().clone()]);
                            self.remove_node(n);
                        } else {
                            let contigs_ = contigs_from_vertex(&mut self, n);
                            contigs.extend(contigs_.into_iter().filter(|c| !c.is_empty()));
                        }
                    }
                }
            }
            log::debug!("\t- Loops over external nodes finished.");

            // The walks from path ends consume every path, as the junctions are already gone, so only
            // cycles without a junction remain; any node on one is a valid start, and the walk goes
            // all the way round. No strongly-connected-components search: petgraph's is recursive,
            // and a plasmid the shrinker never reached is a cycle of one node per k-mer.
            let Some(start) = self.node_indices().next() else {
                break;
            };
            let before = self.node_count();
            contigs.extend(
                contigs_from_intermediate_vertex(&mut self, start)
                    .into_iter()
                    .filter(|c| !c.is_empty()),
            );
            cycles.push(before - self.node_count());
        }

        if !cycles.is_empty() {
            log::info!(
                "Walked {} cycle(s) without a junction; the largest had {} nodes",
                cycles.len(),
                cycles.iter().max().unwrap()
            );
        }

        log::trace!(
            "{} nodes left in the graph after collapse",
            self.node_count()
        );
        log::info!("Collapse ended. Created {} contigs", contigs.len());

        contigs
    }
}

/// Remove branching junction nodes before collapse walks begin, returning each as its own contig.
/// This is important for performance, speeds things up a lot
#[inline]
fn erase_junction_nodes(ptgraph: &mut DbgGraph) -> SerializedContigs {
    let junctions: Vec<NodeId> = ptgraph
        .ambiguous_nodes()
        .into_iter()
        .filter(|&node| ptgraph.nonself_degree(node) > 1)
        .collect();

    junctions
        .into_iter()
        .filter_map(|node| ptgraph.remove_node(node))
        .map(|weight| vec![weight])
        .collect()
}

/// Walk a path from its end `v`, emitting it as one contig. With the junctions already erased, this
/// turns every remaining path into a contig.
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
            // A junction ends the contig and becomes a contig of its own.
            contigs.push(contig.clone());
            contig.clear();
            contigs.extend(
                ptgraph
                    .remove_node(current_vertex)
                    .map(|weight| vec![weight]),
            );
            return contigs;
        }

        // If we arrive here, current_vertex has exactly one outgoing neighbour and no preceding
        // neighbour, so it can be added to the current contig.

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
    let outmin = ptgraph.outgoing_edges_by_carry(v, CarryType::Min);
    let outmax = ptgraph.outgoing_edges_by_carry(v, CarryType::Max);
    let outminlen = outmin.len();
    let outmaxlen = outmax.len();
    let outeds = match (outminlen, outmaxlen) {
        (0, 0) => {
            // An isolated node becomes its own contig.
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
            // A junction ends the contig and becomes a contig of its own.
            contigs.push(contig.clone());
            contig.clear();
            contigs.extend(
                ptgraph
                    .remove_node(current_vertex)
                    .map(|weight| vec![weight]),
            );
            return contigs;
        }

        // If we arrive here, current_vertex has exactly one outgoing neighbour and no preceding
        // neighbour, so it can be added to the current contig.

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
    use sparrowhawk_graph::EdgeType;

    fn node(h: u64) -> NodeStruct {
        NodeStruct {
            counts: 10,
            abs_ind: vec![h],
            innerdir: None,
        }
    }

    /// A legitimate one-carry start walks its available side.
    #[test]
    fn collapse_walks_a_legitimate_one_carry_start() {
        let mut g = DbgGraph::new(3);
        let v = g.add_node(node(0));
        let w = g.add_node(node(1));
        g.add_bi_edge(v, w, EdgeType::MinToMin);

        let contigs = contigs_from_intermediate_vertex(&mut g, v);

        assert_eq!(contigs.len(), 1);
        let walked: Vec<_> = contigs[0].iter().map(|n| n.abs_ind.clone()).collect();
        assert_eq!(walked, vec![vec![0], vec![1]], "the walk runs from v to w");
        assert_eq!(g.node_count(), 0);
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

    /// External and intermediate walks alike end at a branching junction and emit it as its own
    /// contig, leaving the branches beyond it in the graph.
    #[test]
    fn both_walks_keep_a_branching_junction_as_its_own_contig() {
        type Walker = fn(&mut DbgGraph, NodeId) -> SerializedContigs;
        let walkers: [(&str, Walker); 2] = [
            ("external", contigs_from_vertex),
            ("intermediate", contigs_from_intermediate_vertex),
        ];
        for (name, walk) in walkers {
            let mut g = DbgGraph::new(3);
            let start = g.add_node(node(0));
            let junction = g.add_node(node(1));
            let branch_a = g.add_node(node(2));
            let branch_b = g.add_node(node(3));

            g.add_bi_edge(start, junction, EdgeType::MinToMin);
            g.add_bi_edge(junction, branch_a, EdgeType::MinToMin);
            g.add_bi_edge(junction, branch_b, EdgeType::MinToMin);

            let contigs = walk(&mut g, start);

            assert_eq!(contigs.len(), 2, "{name} walk");
            assert_eq!(contigs[0].len(), 1, "{name} walk");
            assert_eq!(contigs[0][0].abs_ind, vec![0], "{name} walk");
            assert_eq!(contigs[1].len(), 1, "{name} walk");
            assert_eq!(contigs[1][0].abs_ind, vec![1], "{name} walk");
            assert!(!g.contains_node(junction), "{name} walk");
            assert!(
                g.contains_node(branch_a) && g.contains_node(branch_b),
                "{name} walk"
            );
        }
    }

    /// Pre-collapse junction erasure removes the junction and its incident edges only, and returns it.
    #[test]
    fn erase_junction_nodes_removes_branching_node_and_edges() {
        let mut g = DbgGraph::new(3);
        let junction = g.add_node(node(0));
        let branch_a = g.add_node(node(1));
        let branch_b = g.add_node(node(2));

        g.add_bi_edge(junction, branch_a, EdgeType::MinToMin);
        g.add_bi_edge(junction, branch_b, EdgeType::MinToMin);

        let kept = erase_junction_nodes(&mut g);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0][0].abs_ind, vec![0]);
        assert!(!g.contains_node(junction));
        assert!(g.contains_node(branch_a));
        assert!(g.contains_node(branch_b));
        assert_eq!(g.out_degree(branch_a), 0);
        assert_eq!(g.in_degree(branch_a), 0);
        assert_eq!(g.out_degree(branch_b), 0);
        assert_eq!(g.in_degree(branch_b), 0);
    }

    /// Adds `len` nodes chained by straight edges, closing them into a cycle when asked, and
    /// returns their hashes, which start at `first`.
    fn add_chain(g: &mut DbgGraph, first: u64, len: u64, circular: bool) -> Vec<u64> {
        let ids: Vec<NodeId> = (first..first + len).map(|h| g.add_node(node(h))).collect();
        for pair in ids.windows(2) {
            g.add_bi_edge(pair[0], pair[1], EdgeType::MinToMin);
        }
        if circular {
            g.add_bi_edge(ids[ids.len() - 1], ids[0], EdgeType::MinToMin);
        }
        (first..first + len).collect()
    }

    /// A long cycle without junctions, as a plasmid the shrinker never reaches leaves it, is walked
    /// whole without any stack-depth-dependent step.
    #[test]
    fn a_long_cycle_without_junctions_collapses_on_a_small_stack() {
        const LEN: u64 = 100_000;
        let mut g = DbgGraph::new(3);
        add_chain(&mut g, 0, LEN, true);
        assert!(g.ambiguous_nodes().is_empty(), "every node is straight");

        let contigs = std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(move || g.collapse())
            .unwrap()
            .join()
            .expect("collapse overflowed a small stack");

        assert_eq!(contigs.len(), 1);
        assert_eq!(contigs[0].len(), LEN as usize);
    }

    /// Paths and junction-free cycles together collapse into one contig each, with every node
    /// appearing exactly once.
    #[test]
    fn paths_and_cycles_collapse_together() {
        let mut g = DbgGraph::new(3);
        let mut expected = Vec::new();
        expected.extend(add_chain(&mut g, 0, 5, false));
        expected.extend(add_chain(&mut g, 100, 7, true));
        expected.extend(add_chain(&mut g, 200, 3, false));
        expected.extend(add_chain(&mut g, 300, 4, true));

        let contigs = g.collapse();

        let mut sizes: Vec<usize> = contigs.iter().map(Vec::len).collect();
        sizes.sort_unstable();
        assert_eq!(sizes, vec![3, 4, 5, 7]);
        let mut seen: Vec<u64> = contigs
            .iter()
            .flatten()
            .flat_map(|n| n.abs_ind.clone())
            .collect();
        seen.sort_unstable();
        expected.sort_unstable();
        assert_eq!(seen, expected);
    }
}
