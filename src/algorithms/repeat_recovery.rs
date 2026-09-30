//! Native repeat motif discovery, coverage validation, and extraction.

use std::collections::{BTreeMap, BTreeSet};

use sparrowhawk_graph::{DbgGraph, SerializedContigs};

use crate::indexed_kmers::IndexedKmers;

use super::repeat_coverage::{
    explain_flow_diamond, explain_simple_repeat, explain_theta, homogeneous_node_means,
    is_parallel_bundle, supports_flow_diamond, supports_simple_repeat, supports_theta,
    CountSnapshot,
};
use super::repeat_surgery::extract_and_remove;
use super::superbubble::{find_all, OrientedNode, Superbubble};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MotifKind {
    SimpleRepeat,
    Theta,
    FlowDiamond,
}

#[derive(Clone)]
struct Candidate {
    id: usize,
    bubble: Superbubble,
    kind: MotifKind,
    interior: BTreeSet<OrientedNode>,
}

/// Find supported repeat structures, append their paths to the normal collapsed contigs, and remove
/// the recovered source nodes so collapse cannot emit those graph pieces a second time.
pub(crate) fn recover<IntT>(
    graph: &mut DbgGraph,
    kmers: &IndexedKmers<IntT>,
    count_snapshot: &[u32],
) -> SerializedContigs {
    let counts = CountSnapshot::new(kmers, count_snapshot);
    let bubbles = find_all(graph);
    let examined = bubbles.len();
    let debug = log::log_enabled!(log::Level::Debug);
    let mut candidates = Vec::new();
    for (id, bubble) in bubbles.into_iter().enumerate() {
        if debug {
            log_bubble_detail(graph, &counts, id, &bubble);
        }
        if let Some(candidate) = classify(graph, &counts, id, bubble, debug) {
            candidates.push(candidate);
        }
    }

    // A larger superbubble containing a smaller supported bubble is a nested structure. Recover its
    // local child routes separately, unless it is the three-route coverage staircase (#11).
    let candidate_ids = candidates
        .iter()
        .map(|candidate| candidate.id)
        .collect::<Vec<_>>();
    let nested_children = nested_children(&candidates);
    let nested = nested_children
        .iter()
        .map(Option::is_some)
        .collect::<Vec<_>>();
    let nested_count = nested.iter().filter(|&&value| value).count();
    let mut retained_candidates = Vec::with_capacity(candidates.len());
    for (candidate, nested_child) in candidates.into_iter().zip(nested_children) {
        let keep_nested_flow = nested_child.is_some() && candidate.kind == MotifKind::FlowDiamond;
        let retained = nested_child.is_none() || keep_nested_flow;
        if debug {
            log::debug!(
                "Repeat candidate #{} nested handling: kind={:?}, contains_smaller_candidate={:?}, decision={}",
                candidate.id,
                candidate.kind,
                nested_child.map(|index| candidate_ids[index]),
                if retained {
                    if keep_nested_flow { "retain (flow-diamond nesting rule)" } else { "retain (not nested)" }
                } else {
                    "reject (contained by a larger supported motif)"
                }
            );
        }
        if retained {
            retained_candidates.push(candidate);
        }
    }
    let mut candidates = retained_candidates;

    // Prefer a validated #11 over its contained local bubbles. Serial bubbles share only a boundary,
    // so they remain separate local route sets and are never expanded into cross-products.
    candidates.sort_by_key(|candidate| {
        (
            match candidate.kind {
                MotifKind::FlowDiamond => 0,
                MotifKind::Theta => 1,
                MotifKind::SimpleRepeat => 2,
            },
            candidate.interior.len(),
            candidate.bubble.entrance,
        )
    });
    let mut selected = Vec::<Candidate>::new();
    let mut claimed = BTreeMap::<OrientedNode, usize>::new();
    for candidate in candidates {
        let conflicts = candidate
            .interior
            .iter()
            .filter_map(|node| claimed.get(node).map(|owner| (*node, *owner)))
            .collect::<Vec<_>>();
        if !conflicts.is_empty() {
            if debug {
                log::debug!(
                    "Repeat candidate #{} overlap handling: reject because interior nodes are already claimed: {:?}",
                    candidate.id,
                    conflicts
                );
            }
            continue;
        }
        if debug {
            log::debug!(
                "Repeat candidate #{} overlap handling: selected kind={:?}, entrance={:?}, exit={:?}, routes={}",
                candidate.id,
                candidate.kind,
                candidate.bubble.entrance,
                candidate.bubble.exit,
                candidate.bubble.paths.len()
            );
        }
        claimed.extend(
            candidate
                .interior
                .iter()
                .copied()
                .map(|node| (node, candidate.id)),
        );
        selected.push(candidate);
    }

    let (serial_pairs, _) = count_serial_pairs(graph, &selected);
    let simple_count = selected
        .iter()
        .filter(|candidate| candidate.kind == MotifKind::SimpleRepeat)
        .count();
    let theta_count = selected
        .iter()
        .filter(|candidate| candidate.kind == MotifKind::Theta)
        .count();
    let flow_count = selected
        .iter()
        .filter(|candidate| candidate.kind == MotifKind::FlowDiamond)
        .count();

    let route_sets = selected
        .iter()
        .map(|candidate| candidate.bubble.paths.clone())
        .collect::<Vec<_>>();
    if debug {
        for candidate in &selected {
            log::debug!(
                "Repeat extraction selection #{}: kind={:?}, entrance={:?}, exit={:?}, interior_nodes={}, routes={:?}",
                candidate.id,
                candidate.kind,
                candidate.bubble.entrance,
                candidate.bubble.exit,
                candidate.interior.len(),
                candidate.bubble.paths.iter().map(|path| path.len()).collect::<Vec<_>>()
            );
        }
    }
    let recovered = extract_and_remove(graph, &counts, &route_sets);
    log::info!(
        "Repeat recovery: {examined} superbubbles examined; supported simple={simple_count}, theta={theta_count}, flow-diamond={flow_count}, nested={nested_count}, serial-pairs={serial_pairs}; extracted {} route contigs",
        recovered.len()
    );
    recovered
}

/// For each candidate enclosing a smaller candidate, return one contained child index. Candidate
/// entrance indexing bounds the search to motifs sharing a node with the outer region.
fn nested_children(candidates: &[Candidate]) -> Vec<Option<usize>> {
    let mut entrances = BTreeMap::<OrientedNode, Vec<usize>>::new();
    for (index, candidate) in candidates.iter().enumerate() {
        entrances
            .entry(candidate.bubble.entrance)
            .or_default()
            .push(index);
        entrances
            .entry(candidate.bubble.exit.reverse_complement())
            .or_default()
            .push(index);
    }

    let mut nested = vec![None; candidates.len()];
    for (outer_index, outer) in candidates.iter().enumerate() {
        let outer_states = candidate_states(outer);
        let mut possible_inner = BTreeSet::new();
        for state in &outer_states {
            if let Some(indices) = entrances.get(state) {
                possible_inner.extend(indices.iter().copied());
            }
        }
        for inner_index in possible_inner {
            if inner_index == outer_index
                || candidates[inner_index].interior.len() >= outer.interior.len()
            {
                continue;
            }
            let inner_states = candidate_states(&candidates[inner_index]);
            let reverse_inner = inner_states
                .iter()
                .map(|state| state.reverse_complement())
                .collect::<BTreeSet<_>>();
            if inner_states.is_subset(&outer_states) || reverse_inner.is_subset(&outer_states) {
                nested[outer_index] = Some(inner_index);
                break;
            }
        }
    }
    nested
}

/// Boolean compatibility helper retained for the focused nesting tests.
#[cfg(test)]
fn nested_flags(candidates: &[Candidate]) -> Vec<bool> {
    nested_children(candidates)
        .into_iter()
        .map(|parent| parent.is_some())
        .collect()
}

fn candidate_states(candidate: &Candidate) -> BTreeSet<OrientedNode> {
    candidate
        .interior
        .iter()
        .copied()
        .chain([candidate.bubble.entrance, candidate.bubble.exit])
        .collect()
}

/// Count serial relationships by walking once from each oriented exit and looking up entrances.
/// Returns the pair count and the number of graph states inspected (used by complexity tests).
fn count_serial_pairs(graph: &DbgGraph, selected: &[Candidate]) -> (usize, usize) {
    const MAX_CONNECTOR_STATES: usize = 60;
    let mut entrances = BTreeMap::<OrientedNode, Vec<usize>>::new();
    for (index, candidate) in selected.iter().enumerate() {
        entrances
            .entry(candidate.bubble.entrance)
            .or_default()
            .push(index);
        entrances
            .entry(candidate.bubble.exit.reverse_complement())
            .or_default()
            .push(index);
    }

    let mut pairs = BTreeSet::new();
    let mut inspected = 0;
    for (source_index, source) in selected.iter().enumerate() {
        for start in [
            source.bubble.exit,
            source.bubble.entrance.reverse_complement(),
        ] {
            let mut current = start;
            let mut visited = BTreeSet::from([start]);
            for _ in 0..=MAX_CONNECTOR_STATES {
                inspected += 1;
                // A shared exit/entrance is a valid direct serial boundary. Once the walk has
                // entered a connector, however, it must arrive at the next entrance uniquely.
                if current != start
                    && graph.in_neighbours_bi(current.node, current.carry()).len() != 1
                {
                    break;
                }
                if let Some(targets) = entrances.get(&current) {
                    for &target_index in targets {
                        if target_index != source_index {
                            pairs.insert((
                                source_index.min(target_index),
                                source_index.max(target_index),
                            ));
                        }
                    }
                    if targets.iter().any(|&target| target != source_index) {
                        break;
                    }
                }

                // The starting node is the source bubble's exit, so only its outgoing degree
                // constrains the connector. Every subsequent connector state must be one-in/one-out.
                let next = graph.out_neighbours_bi(current.node, current.carry());
                if next.len() != 1 {
                    break;
                }
                let (node, edge) = next[0];
                let (_, carry) = edge.get_from_and_to();
                current = OrientedNode::new(node, carry);
                if !visited.insert(current) {
                    break;
                }
            }
        }
    }
    (pairs.len(), inspected)
}

fn log_bubble_detail<IntT>(
    graph: &DbgGraph,
    counts: &CountSnapshot<'_, IntT>,
    id: usize,
    bubble: &Superbubble,
) {
    log::debug!(
        "Repeat candidate #{id} detected: entrance={:?}, exit={:?}, interior_nodes={}, route_count={}, routes={:?}",
        bubble.entrance,
        bubble.exit,
        bubble.interior.len(),
        bubble.paths.len(),
        bubble.paths.iter().map(|path| describe_route(path)).collect::<Vec<_>>()
    );

    let states = bubble
        .paths
        .iter()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>();
    for state in states {
        let detail = counts.node_coverage_diagnostic(graph, state);
        log::debug!(
            "Repeat candidate #{id} node {state:?}: graph_node_average_count={:?}, per_kmer_mean={:?}, per_kmer_min={:?}, per_kmer_max={:?}, kmer_count={}, missing_counts={}, zero_counts={}, depth_steps={:?}, homogeneous_mean={:?}, issue={:?}",
            detail.graph_node_average,
            detail.kmer_mean,
            detail.kmer_min,
            detail.kmer_max,
            detail.kmer_count,
            detail.missing_counts,
            detail.zero_counts,
            detail.depth_steps,
            detail.homogeneous_mean,
            detail.issue
        );
    }
}

fn describe_route(path: &[OrientedNode]) -> String {
    path.iter()
        .map(|state| format!("{:?}{}", state.node, if state.reverse { "-" } else { "+" }))
        .collect::<Vec<_>>()
        .join(" -> ")
}

fn classify<IntT>(
    graph: &DbgGraph,
    counts: &CountSnapshot<'_, IntT>,
    id: usize,
    bubble: Superbubble,
    debug: bool,
) -> Option<Candidate> {
    let interior = bubble.interior.iter().copied().collect::<BTreeSet<_>>();
    let route_interior = bubble
        .paths
        .iter()
        .flat_map(|path| {
            path.iter()
                .skip(1)
                .take(path.len().saturating_sub(2))
                .copied()
        })
        .collect::<BTreeSet<_>>();
    if interior != route_interior {
        if debug {
            log::debug!(
                "Repeat candidate #{id} rejected: superbubble interior does not match the union of route interiors (detector_interior={interior:?}, route_interior={route_interior:?})"
            );
        }
        return None;
    }
    if let Some((route_index, route_len)) = bubble
        .paths
        .iter()
        .enumerate()
        .find_map(|(index, path)| (path.len() < 3).then_some((index, path.len())))
    {
        if debug {
            log::debug!(
                "Repeat candidate #{id} rejected: route {} has {route_len} node(s), but motif validation requires at least 3",
                route_index + 1
            );
        }
        return None;
    }
    let Some(node_means) = homogeneous_node_means(graph, &bubble.paths, counts) else {
        if debug {
            let states = bubble
                .paths
                .iter()
                .flatten()
                .copied()
                .collect::<BTreeSet<_>>();
            for state in states {
                let detail = counts.node_coverage_diagnostic(graph, state);
                if detail.homogeneous_mean.is_none() {
                    log::debug!(
                        "Repeat candidate #{id} rejected at node {state:?}: cannot establish homogeneous per-k-mer coverage; graph_node_average_count={:?}, kmer_count={}, per_kmer_mean={:?}, missing_counts={}, zero_counts={}, depth_steps={:?}, reason={:?}",
                        detail.graph_node_average,
                        detail.kmer_count,
                        detail.kmer_mean,
                        detail.missing_counts,
                        detail.zero_counts,
                        detail.depth_steps,
                        detail.issue
                    );
                }
            }
            log::debug!(
                "Repeat candidate #{id} rejected: one or more unitigs had missing, zero, or non-homogeneous per-k-mer coverage"
            );
        }
        return None;
    };

    let parallel = is_parallel_bundle(&bubble.paths, &bubble.interior);

    if bubble.paths.len() == 2 && parallel {
        if supports_simple_repeat(graph, &bubble.paths, counts) {
            if debug {
                log::debug!(
                    "Repeat candidate #{id} accepted as SimpleRepeat: {}",
                    explain_simple_repeat(graph, &bubble.paths, counts)
                );
            }
            return Some(Candidate {
                id,
                bubble,
                kind: MotifKind::SimpleRepeat,
                interior,
            });
        }
        if debug {
            log::debug!(
                "Repeat candidate #{id} rejected as SimpleRepeat: {}",
                explain_simple_repeat(graph, &bubble.paths, counts)
            );
        }
        return None;
    }

    if bubble.paths.len() >= 3 && parallel {
        if supports_theta(graph, &bubble.paths, counts) {
            if debug {
                log::debug!(
                    "Repeat candidate #{id} accepted as Theta: {}",
                    explain_theta(graph, &bubble.paths, counts)
                );
            }
            return Some(Candidate {
                id,
                bubble,
                kind: MotifKind::Theta,
                interior,
            });
        }
        if debug {
            log::debug!(
                "Repeat candidate #{id} rejected as Theta: {}",
                explain_theta(graph, &bubble.paths, counts)
            );
        }
        return None;
    }

    if bubble.paths.len() == 3 && !parallel {
        if supports_flow_diamond(&bubble.paths, &node_means) {
            if debug {
                log::debug!(
                    "Repeat candidate #{id} accepted as FlowDiamond: {}",
                    explain_flow_diamond(&bubble.paths, &node_means)
                );
            }
            return Some(Candidate {
                id,
                bubble,
                kind: MotifKind::FlowDiamond,
                interior,
            });
        }
        if debug {
            log::debug!(
                "Repeat candidate #{id} rejected as FlowDiamond: {}",
                explain_flow_diamond(&bubble.paths, &node_means)
            );
        }
        return None;
    }

    if debug {
        let topology = if parallel {
            "pairwise-disjoint parallel routes"
        } else {
            "overlapping/non-parallel routes"
        };
        log::debug!(
            "Repeat candidate #{id} rejected: unsupported motif topology ({} routes, {topology}); supported cases are 2-route simple repeats, >=3-route parallel theta motifs, or 3-route non-parallel flow diamonds",
            bubble.paths.len()
        );
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    use sparrowhawk_graph::{CarryType, EdgeType, NodeStruct};

    fn candidate_from_states(
        entrance: OrientedNode,
        interior: &[OrientedNode],
        exit: OrientedNode,
    ) -> Candidate {
        Candidate {
            id: 0,
            bubble: Superbubble {
                entrance,
                exit,
                interior: interior.to_vec(),
                paths: Vec::new(),
            },
            kind: MotifKind::SimpleRepeat,
            interior: interior.iter().copied().collect(),
        }
    }

    fn candidates_from_graph(graph: &DbgGraph) -> Vec<Candidate> {
        find_all(graph)
            .into_iter()
            .enumerate()
            .map(|(id, bubble)| Candidate {
                id,
                interior: bubble.interior.iter().copied().collect(),
                bubble,
                kind: MotifKind::SimpleRepeat,
            })
            .collect()
    }

    fn sequence_walk(sequence: &[u8], k: usize) -> Vec<(u64, u64, u8, u64)> {
        let mut iterator = crate::kmer::Kmer::<u64>::new(
            Cow::Borrowed(sequence),
            sequence.len(),
            None,
            k,
            0,
            true,
        )
        .unwrap();
        let (hash, reverse, bases, packed) = iterator.get_curr_kmerhash_and_bases_and_kmer();
        let mut walk = vec![(hash, reverse, bases, packed)];
        while let Some((hash, reverse, bases, packed)) = iterator.get_next_kmer_and_give_us_things()
        {
            walk.push((hash, reverse, bases, packed));
        }
        walk
    }

    fn graph_with_counts(
        counts: &[u32],
        edges: &[(usize, usize)],
    ) -> (DbgGraph, IndexedKmers<u64>, Vec<u32>) {
        let mut graph = DbgGraph::new(3);
        let mut kmers = IndexedKmers::<u64>::with_capacity(counts.len());
        let mut nodes = Vec::with_capacity(counts.len());
        for (index, &count) in counts.iter().enumerate() {
            let hash = 11 + index as u64;
            nodes.push(graph.add_node(NodeStruct {
                counts: count,
                abs_ind: vec![hash],
                innerdir: None,
            }));
            kmers.push(hash, hash + 100, 0, count, hash);
        }
        for &(from, to) in edges {
            graph.add_bi_edge(nodes[from], nodes[to], EdgeType::MinToMin);
        }
        (graph, kmers, counts.to_vec())
    }

    #[test]
    fn supported_simple_repeat_becomes_two_full_contigs_and_leaves_valid_graph() {
        let k = 3;
        let sequences: [&[u8]; 2] = [b"GAACGCT", b"GAATGCT"];
        let walks = sequences.map(|sequence| sequence_walk(sequence, k));
        let mut info = std::collections::BTreeMap::<u64, (u64, u8, u64, u32)>::new();
        for walk in &walks {
            for (position, &(hash, reverse, bases, packed)) in walk.iter().enumerate() {
                let count = if position == 0 || position + 1 == walk.len() {
                    20
                } else {
                    10
                };
                info.entry(hash).or_insert((reverse, bases, packed, count));
            }
        }
        let mut graph = DbgGraph::new(k);
        let mut node_ids = std::collections::BTreeMap::new();
        for (&hash, &(_, _, _, count)) in &info {
            let node = graph.add_node(NodeStruct {
                counts: count,
                abs_ind: vec![hash],
                innerdir: None,
            });
            node_ids.insert(hash, node);
        }
        for walk in &walks {
            for pair in walk.windows(2) {
                graph.add_bi_edge(
                    node_ids[&pair[0].0],
                    node_ids[&pair[1].0],
                    EdgeType::MinToMin,
                );
            }
        }
        let mut kmers = IndexedKmers::<u64>::with_capacity(info.len());
        let mut counts = Vec::with_capacity(info.len());
        for (&hash, &(reverse, bases, packed, count)) in &info {
            kmers.push(hash, reverse, bases, count, packed);
            counts.push(count);
        }

        let recovered = recover(&mut graph, &kmers, &counts);
        assert_eq!(recovered.len(), 2);
        let mut expected_hash_routes = walks
            .iter()
            .map(|walk| {
                let forward = walk.iter().map(|entry| entry.0).collect::<Vec<_>>();
                let reverse = forward.iter().rev().copied().collect::<Vec<_>>();
                forward.min(reverse)
            })
            .collect::<Vec<_>>();
        let mut actual_hash_routes = recovered
            .iter()
            .map(|contig| {
                let forward = contig
                    .iter()
                    .flat_map(|node| node.abs_ind.iter().copied())
                    .collect::<Vec<_>>();
                let reverse = forward.iter().rev().copied().collect::<Vec<_>>();
                forward.min(reverse)
            })
            .collect::<Vec<_>>();
        expected_hash_routes.sort();
        actual_hash_routes.sort();
        assert_eq!(actual_hash_routes, expected_hash_routes);
        let spelled = recovered
            .iter()
            .map(|contig| {
                let route = contig
                    .iter()
                    .flat_map(|node| node.abs_ind.iter().copied())
                    .collect::<Vec<_>>();
                crate::graph_works::spell_path(&route, &kmers, k).expect("extracted route overlaps")
            })
            .collect::<Vec<_>>();
        assert!(spelled.iter().all(|sequence| sequence.len() == 7));
        assert_eq!(graph.node_count(), 0);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn weak_coverage_ratio_does_not_extract_a_bubble() {
        let mut graph = DbgGraph::new(3);
        let left = graph.add_node(NodeStruct {
            counts: 20,
            abs_ind: vec![11],
            innerdir: None,
        });
        let a = graph.add_node(NodeStruct {
            counts: 15,
            abs_ind: vec![12],
            innerdir: None,
        });
        let b = graph.add_node(NodeStruct {
            counts: 15,
            abs_ind: vec![13],
            innerdir: None,
        });
        let right = graph.add_node(NodeStruct {
            counts: 20,
            abs_ind: vec![14],
            innerdir: None,
        });
        for (from, to) in [(left, a), (a, right), (left, b), (b, right)] {
            graph.add_bi_edge(from, to, EdgeType::MinToMin);
        }
        let mut kmers = IndexedKmers::<u64>::with_capacity(4);
        for (hash, count) in [(11, 20), (12, 15), (13, 15), (14, 20)] {
            kmers.push(hash, hash + 100, 0, count, hash);
        }

        let recovered = recover(&mut graph, &kmers, &[20, 15, 15, 20]);
        assert!(recovered.is_empty());
        assert_eq!(graph.node_count(), 4);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn three_route_theta_is_recovered_when_branch_depths_sum_to_the_flanks() {
        let (mut graph, kmers, counts) = graph_with_counts(
            &[30, 10, 10, 10, 30],
            &[(0, 1), (1, 4), (0, 2), (2, 4), (0, 3), (3, 4)],
        );
        let recovered = recover(&mut graph, &kmers, &counts);

        assert_eq!(recovered.len(), 3);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn three_route_flow_diamond_is_recovered_without_route_multiplicity() {
        let (mut graph, kmers, counts) = graph_with_counts(
            &[4, 3, 1, 2, 3, 1, 4],
            &[
                (0, 1),
                (1, 2),
                (2, 6),
                (1, 3),
                (3, 4),
                (0, 5),
                (5, 4),
                (4, 6),
            ],
        );
        let recovered = recover(&mut graph, &kmers, &counts);

        assert_eq!(recovered.len(), 3);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn serial_bubbles_emit_local_routes_instead_of_cross_products() {
        let (mut graph, kmers, counts) = graph_with_counts(
            &[20, 10, 10, 20, 10, 10, 20],
            &[
                (0, 1),
                (1, 3),
                (0, 2),
                (2, 3),
                (3, 4),
                (4, 6),
                (3, 5),
                (5, 6),
            ],
        );
        let recovered = recover(&mut graph, &kmers, &counts);

        assert_eq!(
            recovered.len(),
            4,
            "local paths produce four routes, not 16 combinations"
        );
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn nested_index_requires_full_containment_not_interior_overlap() {
        let (graph, _, _) = graph_with_counts(&[1; 10], &[]);
        let states = graph
            .node_indices()
            .map(|node| OrientedNode::new(node, CarryType::Min))
            .collect::<Vec<_>>();
        let candidates = vec![
            candidate_from_states(states[0], &states[1..7], states[7]),
            candidate_from_states(states[1], &states[2..4], states[4]),
            candidate_from_states(states[8], &[states[2]], states[9]),
        ];
        assert_eq!(nested_flags(&candidates), vec![true, false, false]);
    }

    #[test]
    fn serial_search_scales_with_local_walks_for_disconnected_bubbles() {
        let bubble_count = 64;
        let counts = [20, 10, 10, 20]
            .into_iter()
            .cycle()
            .take(bubble_count * 4)
            .collect::<Vec<_>>();
        let edges = (0..bubble_count)
            .flat_map(|index| {
                let start = index * 4;
                [
                    (start, start + 1),
                    (start, start + 2),
                    (start + 1, start + 3),
                    (start + 2, start + 3),
                ]
            })
            .collect::<Vec<_>>();
        let (graph, _, _) = graph_with_counts(&counts, &edges);
        let candidates = candidates_from_graph(&graph);
        assert_eq!(candidates.len(), bubble_count);

        let (pairs, inspected) = count_serial_pairs(&graph, &candidates);
        assert_eq!(pairs, 0);
        assert!(
            inspected < candidates.len() * candidates.len(),
            "inspected {inspected} states for {} unrelated motifs",
            candidates.len()
        );
    }

    #[test]
    fn serial_search_finds_bubbles_sharing_a_boundary_on_both_strands() {
        let edges = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),
            (3, 4),
            (3, 5),
            (4, 6),
            (5, 6),
        ];
        let (graph, _, _) = graph_with_counts(&[20, 10, 10, 20, 10, 10, 20], &edges);
        let states = graph
            .node_indices()
            .map(|node| OrientedNode::new(node, CarryType::Min))
            .collect::<Vec<_>>();
        let candidates = vec![
            candidate_from_states(states[0], &[states[1], states[2]], states[3]),
            candidate_from_states(states[3], &[states[4], states[5]], states[6]),
        ];
        assert_eq!(count_serial_pairs(&graph, &candidates).0, 1);

        // Reverse one bubble's traversal: its entrance/exit index must still match the shared
        // boundary in the opposite orientation.
        let reverse_second = candidate_from_states(
            states[6].reverse_complement(),
            &[
                states[4].reverse_complement(),
                states[5].reverse_complement(),
            ],
            states[3].reverse_complement(),
        );
        assert_eq!(
            count_serial_pairs(&graph, &[candidates[0].clone(), reverse_second]).0,
            1
        );
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn serial_search_stops_at_branches_and_cycles() {
        let (branch_graph, _, _) = graph_with_counts(&[1; 10], &[(3, 4), (3, 5), (4, 6)]);
        let branch_states = branch_graph
            .node_indices()
            .map(|node| OrientedNode::new(node, CarryType::Min))
            .collect::<Vec<_>>();
        let source = candidate_from_states(
            branch_states[0],
            &[branch_states[1], branch_states[2]],
            branch_states[3],
        );
        let target = candidate_from_states(
            branch_states[6],
            &[branch_states[7], branch_states[8]],
            branch_states[9],
        );
        assert_eq!(
            branch_graph
                .out_neighbours_bi(branch_states[3].node, branch_states[3].carry())
                .len(),
            2
        );
        assert_eq!(count_serial_pairs(&branch_graph, &[source, target]).0, 0);

        let (cycle_graph, _, _) = graph_with_counts(&[1; 10], &[(3, 4), (4, 5), (5, 3)]);
        let cycle_states = cycle_graph
            .node_indices()
            .map(|node| OrientedNode::new(node, CarryType::Min))
            .collect::<Vec<_>>();
        let source = candidate_from_states(
            cycle_states[0],
            &[cycle_states[1], cycle_states[2]],
            cycle_states[3],
        );
        let target = candidate_from_states(
            cycle_states[6],
            &[cycle_states[7], cycle_states[8]],
            cycle_states[9],
        );
        let (pairs, inspected) = count_serial_pairs(&cycle_graph, &[source, target]);
        assert_eq!(pairs, 0);
        assert!(
            inspected < 2 * 2 * 61,
            "walk must stop when it revisits a state"
        );
    }
}
