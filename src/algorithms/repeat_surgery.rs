//! Convert validated repeat routes to the assembler's existing contig representation.

use std::collections::BTreeSet;

use sparrowhawk_graph::{DbgGraph, NodeId, SerializedContigs};

use crate::algorithms::superbubble::OrientedNode;

use super::repeat_coverage::CountSnapshot;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::superbubble::OrientedNode;
    use crate::indexed_kmers::IndexedKmers;
    use sparrowhawk_graph::{CarryType, EdgeType, NodeStruct};

    fn fixture() -> (DbgGraph, IndexedKmers<u64>, Vec<u32>, Vec<OrientedNode>) {
        let mut graph = DbgGraph::new(3);
        let values = [20, 10, 10, 20];
        let mut kmers = IndexedKmers::<u64>::with_capacity(values.len());
        let mut nodes = Vec::new();
        let mut states = Vec::new();
        for (index, count) in values.into_iter().enumerate() {
            let hash = 50 + index as u64;
            nodes.push(graph.add_node(NodeStruct {
                counts: count,
                abs_ind: vec![hash],
                innerdir: None,
            }));
            kmers.push(hash, hash + 100, 0, count, hash);
            states.push(OrientedNode::new(nodes[index], CarryType::Min));
        }
        for (from, to) in [(0, 1), (1, 3), (0, 2), (2, 3)] {
            graph.add_bi_edge(nodes[from], nodes[to], EdgeType::MinToMin);
        }
        (graph, kmers, values.to_vec(), states)
    }

    #[test]
    fn rejected_motif_does_not_reserve_a_route_before_later_valid_motif() {
        let (mut graph, kmers, counts, state) = fixture();
        let route_a = vec![state[0], state[1], state[3]];
        let route_b = vec![state[0], state[2], state[3]];
        let snapshot = CountSnapshot::new(&kmers, &counts);

        let output = extract_and_remove(
            &mut graph,
            &snapshot,
            &[vec![route_a.clone(), Vec::new()], vec![route_a, route_b]],
        );
        assert_eq!(output.len(), 2);
        assert_eq!(graph.node_count(), 0);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn duplicate_routes_are_localised_and_new_routes_still_emit() {
        let (mut graph, kmers, counts, state) = fixture();
        let route_a = vec![state[0], state[1], state[3]];
        let route_b = vec![state[0], state[2], state[3]];
        let reverse_a = route_a.iter().rev().copied().collect::<Vec<_>>();
        let reverse_b = route_b.iter().rev().copied().collect::<Vec<_>>();
        let snapshot = CountSnapshot::new(&kmers, &counts);

        // The second motif has one route already emitted and one novel route; it remains valid.
        let output = extract_and_remove(
            &mut graph,
            &snapshot,
            &[
                vec![route_a.clone(), route_b.clone()],
                vec![reverse_a, reverse_b],
                vec![route_a, route_b],
            ],
        );
        assert_eq!(output.len(), 2, "reverse and forward duplicates emit once");
        assert_eq!(graph.node_count(), 0);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn one_new_route_in_a_valid_motif_is_emitted_despite_global_duplicate() {
        let (mut graph, kmers, counts, state) = fixture();
        let route_a = vec![state[0], state[1], state[3]];
        let route_b = vec![state[0], state[2], state[3]];
        let new_route = vec![state[1], state[2]];
        let snapshot = CountSnapshot::new(&kmers, &counts);

        let output = extract_and_remove(
            &mut graph,
            &snapshot,
            &[vec![route_a.clone(), route_b], vec![route_a, new_route]],
        );
        assert_eq!(output.len(), 3);
        assert_eq!(graph.node_count(), 0);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn motif_with_only_local_duplicates_is_rejected_without_graph_mutation() {
        let (mut graph, kmers, counts, state) = fixture();
        let route = vec![state[0], state[1], state[3]];
        let snapshot = CountSnapshot::new(&kmers, &counts);
        let before = graph.node_indices().collect::<Vec<_>>();

        let output = extract_and_remove(&mut graph, &snapshot, &[vec![route.clone(), route]]);
        assert!(output.is_empty());
        assert_eq!(graph.node_indices().collect::<Vec<_>>(), before);
        assert!(graph.validate().is_ok());
    }
}

/// Serialize complete routes, then remove their source graph nodes.
pub(crate) fn extract_and_remove<IntT>(
    graph: &mut DbgGraph,
    counts: &CountSnapshot<'_, IntT>,
    accepted_routes: &[Vec<Vec<OrientedNode>>],
) -> SerializedContigs {
    let debug = log::log_enabled!(log::Level::Debug);
    let mut output = Vec::new();
    let mut removed = BTreeSet::<NodeId>::new();
    let mut seen_routes = BTreeSet::<Vec<u64>>::new();

    for (motif_index, motif_routes) in accepted_routes.iter().enumerate() {
        if debug {
            log::debug!(
                "Repeat extraction candidate #{motif_index}: preparing {} route(s), route_node_counts={:?}",
                motif_routes.len(),
                motif_routes.iter().map(Vec::len).collect::<Vec<_>>()
            );
        }
        let mut prepared = Vec::new();
        let mut local_routes = BTreeSet::<Vec<u64>>::new();
        let mut motif_valid = true;
        for (route_index, route) in motif_routes.iter().enumerate() {
            let Some(nodes) = counts.route_nodes(graph, route) else {
                if debug {
                    log::debug!(
                        "Repeat extraction candidate #{motif_index} route {} rejected: one or more source nodes or k-mer counts could not be materialised",
                        route_index + 1
                    );
                }
                motif_valid = false;
                break;
            };
            let hashes = nodes
                .iter()
                .flat_map(|node| node.abs_ind.iter().copied())
                .collect::<Vec<_>>();
            if hashes.is_empty() {
                if debug {
                    log::debug!(
                        "Repeat extraction candidate #{motif_index} route {} rejected: serialised route contains no k-mers",
                        route_index + 1
                    );
                }
                motif_valid = false;
                break;
            }
            let hash_count = hashes.len();
            let reverse = hashes.iter().rev().copied().collect::<Vec<_>>();
            let key = hashes.min(reverse);
            let is_new_local_route = local_routes.insert(key.clone());
            if debug {
                log::debug!(
                    "Repeat extraction candidate #{motif_index} route {}: oriented_nodes={:?}, kmer_count={}, distinct_within_candidate={}, mean_node_coverages={:?}",
                    route_index + 1,
                    route.iter().map(|state| (state.node, state.reverse)).collect::<Vec<_>>(),
                    hash_count,
                    is_new_local_route,
                    nodes.iter().map(|node| node.counts).collect::<Vec<_>>()
                );
            }
            prepared.push((
                key,
                nodes,
                route.iter().map(|state| state.node).collect::<Vec<_>>(),
            ));
        }
        // Do not let a malformed or degenerate motif reserve routes globally. Validate distinct
        // local routes before changing output, the seen set, or the graph-removal set.
        if !motif_valid || local_routes.len() < 2 {
            if debug {
                log::debug!(
                    "Repeat extraction candidate #{motif_index} rejected: motif_valid={motif_valid}, distinct_local_routes={}; at least two valid distinct routes are required",
                    local_routes.len()
                );
            }
            continue;
        }
        for (key, nodes, route_nodes) in prepared {
            let graph_node_count = route_nodes.len();
            removed.extend(route_nodes);
            if seen_routes.insert(key) {
                if debug {
                    log::debug!(
                        "Repeat extraction candidate #{motif_index}: emitted route contig #{}, kmers={}, graph_nodes={graph_node_count}, node_coverages={:?}",
                        output.len() + 1,
                        nodes.iter().map(|node| node.abs_ind.len()).sum::<usize>(),
                        nodes.iter().map(|node| node.counts).collect::<Vec<_>>()
                    );
                }
                output.push(nodes);
            } else if debug {
                log::debug!(
                    "Repeat extraction candidate #{motif_index}: route is a forward/reverse duplicate of an earlier extracted route; no duplicate contig emitted, but its graph nodes are still marked for removal"
                );
            }
        }
    }

    let removed_count = removed.len();
    for node in removed {
        graph.remove_node(node);
    }
    if debug {
        log::debug!(
            "Repeat extraction cleanup: removed {removed_count} graph node(s), emitted {} route contig(s), graph validation follows",
            output.len()
        );
    }
    if let Err(error) = graph.validate() {
        log::error!("Repeat extraction left an invalid graph: {error}");
    }
    output
}
