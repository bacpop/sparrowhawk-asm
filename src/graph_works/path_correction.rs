//! Graph-only correction of short low-coverage alternative paths.
//!
//! Erroneous-connection removal and bubble/bulge removal are both wired into the correction loop, and both
//! judge a path with `corrector::CoverageRef` rather than a bare ratio. The old bubble collapser we had is
//! part of the bulge removal, as bubbles are special cases of bulges.

use crate::graph_works::corrector::{CoverageRef, PathCoverage};
use sparrowhawk_graph::{CarryType, DbgGraph, EdgeType, NodeId};
use std::cmp::Ordering;
use std::collections::HashSet;

/// The bulge rule considers paths up to the larger of this multiple of k...
const BULGE_LEN_K_MULT: usize = 3;
/// ...and k plus this many bases (converted to k-mers by `bulge_max_kmers`).
const BULGE_LEN_K_ADD: usize = 100;

/// Partial paths the alternative search expands before giving up.
const BULGE_ALT_PATH_POPS: usize = 131;
/// An alternative may differ in length from the candidate by this fraction of the candidate...
const BULGE_RELATIVE_LEN_DELTA: f64 = 0.10;
/// ...or by this many k-mers, whichever is larger.
const BULGE_MIN_LEN_DELTA: usize = 3;

/// Longest candidate the bulge rule considers, in k-mers.
fn bulge_max_kmers(k: usize) -> usize {
    let max_bases = (BULGE_LEN_K_MULT * k).max(k + BULGE_LEN_K_ADD);
    max_bases.saturating_sub(k.saturating_sub(1)).max(1)
}
const EC_LEN_K_MULT: usize = 9;

// Structs to apply bulge correction in a more structured way
/// OrientedNode is a normal petgraph node BUT with an exact direction (i.e. strand)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OrientedNode {
    node: NodeId,
    carry: CarryType,
}

#[derive(Clone, Debug)]
struct Candidate {
    entrance: OrientedNode,
    first: OrientedNode,
    first_edge: EdgeType,
    internal: Vec<OrientedNode>,
    exit: OrientedNode,
    last_edge: EdgeType,
    coverage: PathCoverage,
    length: usize,
    signature: Vec<u64>,
    canonical: Vec<u64>,
}

#[derive(Clone)]
struct SearchState {
    current: OrientedNode,
    nodes: Vec<OrientedNode>,
    coverage: PathCoverage,
    length: usize,
    signature: Vec<u64>,
}

#[derive(Clone)]
struct Alternative {
    coverage: PathCoverage,
    length: usize,
    signature: Vec<u64>,
}

enum AlternativeSearch {
    Complete(Option<Alternative>),
    /// Budget ran out with work outstanding. Any route already carried to the exit is complete and
    /// still usable, so it rides along rather than being discarded.
    Capped(Option<Alternative>),
}

fn oriented_hashes(graph: &DbgGraph, state: OrientedNode) -> Vec<u64> {
    let node = graph.node_weight(state.node).expect("live correction node");
    let mut hashes = node.abs_ind.clone();
    if let Some(inner) = node.innerdir {
        if state.carry != inner.get_from_and_to().0 {
            hashes.reverse();
        }
    } else if state.carry == CarryType::Max {
        hashes.reverse();
    }
    hashes
}

fn canonical_signature(signature: &[u64]) -> Vec<u64> {
    let reverse: Vec<u64> = signature.iter().rev().copied().collect();
    if reverse.as_slice() < signature {
        reverse
    } else {
        signature.to_vec()
    }
}

fn sorted_forward(graph: &DbgGraph, state: OrientedNode) -> Vec<(OrientedNode, EdgeType)> {
    let mut next: Vec<_> = graph
        .out_neighbours_bi(state.node, state.carry)
        .into_iter()
        .map(|(node, edge)| {
            (
                OrientedNode {
                    node,
                    carry: edge.get_from_and_to().1,
                },
                edge,
            )
        })
        .collect();

    next.sort_by(|(a, ae), (b, be)| {
        oriented_hashes(graph, *a)
            .cmp(&oriented_hashes(graph, *b))
            .then_with(|| edge_rank(*ae).cmp(&edge_rank(*be)))
    });
    next
}

fn sorted_backward(graph: &DbgGraph, state: OrientedNode) -> Vec<(OrientedNode, EdgeType)> {
    let mut previous: Vec<_> = graph
        .in_neighbours_bi(state.node, state.carry)
        .into_iter()
        .map(|(node, edge)| {
            (
                OrientedNode {
                    node,
                    carry: edge.get_from_and_to().0,
                },
                edge,
            )
        })
        .collect();
    previous.sort_by(|(a, ae), (b, be)| {
        oriented_hashes(graph, *a)
            .cmp(&oriented_hashes(graph, *b))
            .then_with(|| edge_rank(*ae).cmp(&edge_rank(*be)))
    });
    previous
}

fn edge_rank(edge: EdgeType) -> u8 {
    match edge {
        EdgeType::MinToMin => 0,
        EdgeType::MinToMax => 1,
        EdgeType::MaxToMin => 2,
        EdgeType::MaxToMax => 3,
    }
}

fn build_candidate(
    graph: &DbgGraph,
    entrance: OrientedNode,
    first: OrientedNode,
    first_edge: EdgeType,
    max_length: usize,
) -> Option<Candidate> {
    if first.node == entrance.node {
        return None;
    }

    let mut current = first;
    let mut internal = Vec::new();
    let mut coverage = PathCoverage::default();
    let mut length = 0usize;
    let mut signature = Vec::new();
    let mut seen = HashSet::new();
    seen.insert(entrance.node);

    loop {
        if !seen.insert(current.node) {
            return None;
        }

        let incoming = graph.in_degree_bi(current.node, current.carry);
        if incoming > 1 {
            if internal.is_empty() {
                return None;
            }
            let last = *internal.last()?;
            let last_edge = sorted_forward(graph, last)
                .into_iter()
                .find(|(next, _)| *next == current)?
                .1;
            let canonical = canonical_signature(&signature);
            return Some(Candidate {
                entrance,
                first,
                first_edge,
                internal,
                exit: current,
                last_edge,
                coverage,
                length,
                signature,
                canonical,
            });
        }
        if incoming != 1 {
            return None;
        }

        let next = sorted_forward(graph, current);
        if next.len() != 1 {
            return None;
        }

        let node = graph.node_weight(current.node)?;
        length = length.checked_add(node.abs_ind.len())?;
        if length > max_length {
            return None;
        }
        coverage.add_node(node);
        signature.extend(oriented_hashes(graph, current));
        internal.push(current);
        current = next[0].0;
    }
}

fn candidate_still_matches(graph: &DbgGraph, candidate: &Candidate, max_length: usize) -> bool {
    if !graph.contains_node(candidate.entrance.node)
        || !graph.contains_node(candidate.first.node)
        || !graph.contains_node(candidate.exit.node)
    {
        return false;
    }
    build_candidate(
        graph,
        candidate.entrance,
        candidate.first,
        candidate.first_edge,
        max_length,
    )
    .is_some_and(|fresh| {
        fresh.internal == candidate.internal
            && fresh.exit == candidate.exit
            && fresh.signature == candidate.signature
            && fresh.last_edge == candidate.last_edge
    })
}

fn candidate_order(graph: &DbgGraph) -> Vec<(OrientedNode, OrientedNode, EdgeType, Vec<u64>)> {
    let mut starts = Vec::new();
    for node in graph.node_indices() {
        for carry in [CarryType::Min, CarryType::Max] {
            let entrance = OrientedNode { node, carry };
            let outgoing = sorted_forward(graph, entrance);
            if outgoing.len() < 2 {
                continue;
            }
            for (first, edge) in outgoing {
                let mut key = oriented_hashes(graph, entrance);
                key.extend(oriented_hashes(graph, first));
                starts.push((entrance, first, edge, key));
            }
        }
    }
    starts.sort_by(|a, b| {
        a.3.cmp(&b.3)
            .then_with(|| edge_rank(a.2).cmp(&edge_rank(b.2)))
    });
    starts
}

fn state_order(a: &SearchState, b: &SearchState) -> Ordering {
    a.length
        .cmp(&b.length)
        .then_with(|| a.signature.cmp(&b.signature))
}

fn better_alternative(candidate_length: usize, new: &Alternative, old: &Alternative) -> bool {
    new.coverage
        .mean()
        .partial_cmp(&old.coverage.mean())
        .unwrap_or(Ordering::Equal)
        .then_with(|| {
            old.length
                .abs_diff(candidate_length)
                .cmp(&new.length.abs_diff(candidate_length))
        })
        .then_with(|| old.signature.cmp(&new.signature))
        == Ordering::Greater
}

fn find_alternative(graph: &DbgGraph, candidate: &Candidate) -> AlternativeSearch {
    let delta = BULGE_MIN_LEN_DELTA
        .max((candidate.length as f64 * BULGE_RELATIVE_LEN_DELTA).ceil() as usize);
    let min_length = candidate.length.saturating_sub(delta);
    let max_length = candidate.length.saturating_add(delta);
    let excluded: HashSet<NodeId> = candidate.internal.iter().map(|state| state.node).collect();
    let mut queue = Vec::new();

    for (next, _) in sorted_forward(graph, candidate.entrance) {
        if excluded.contains(&next.node) || next.node == candidate.entrance.node {
            continue;
        }
        if next == candidate.exit {
            continue;
        }
        let Some(node) = graph.node_weight(next.node) else {
            continue;
        };
        let length = node.abs_ind.len();
        if length <= max_length {
            let mut coverage = PathCoverage::default();
            coverage.add_node(node);
            queue.push(SearchState {
                current: next,
                nodes: vec![next],
                coverage,
                length,
                signature: oriented_hashes(graph, next),
            });
        }
    }

    let mut popped = 0usize;
    let mut best: Option<Alternative> = None;
    let cap = BULGE_ALT_PATH_POPS;
    while !queue.is_empty() && popped < cap {
        queue.sort_by(state_order);
        let state = queue.remove(0);
        popped += 1;

        for (next, _) in sorted_forward(graph, state.current) {
            if next == candidate.exit {
                if state.length >= min_length {
                    let alternative = Alternative {
                        coverage: state.coverage,
                        length: state.length,
                        signature: state.signature.clone(),
                    };
                    if best
                        .as_ref()
                        .is_none_or(|old| better_alternative(candidate.length, &alternative, old))
                    {
                        best = Some(alternative);
                    }
                }
                continue;
            }
            if excluded.contains(&next.node)
                || state.nodes.iter().any(|visited| visited.node == next.node)
                || next.node == candidate.entrance.node
            {
                continue;
            }
            let Some(node) = graph.node_weight(next.node) else {
                continue;
            };
            let Some(length) = state.length.checked_add(node.abs_ind.len()) else {
                continue;
            };
            if length > max_length {
                continue;
            }
            let mut next_state = state.clone();
            next_state.current = next;
            next_state.nodes.push(next);
            next_state.coverage.add_node(node);
            next_state.length = length;
            next_state.signature.extend(oriented_hashes(graph, next));
            queue.push(next_state);
        }
    }

    if !queue.is_empty() {
        AlternativeSearch::Capped(best)
    } else {
        AlternativeSearch::Complete(best)
    }
}

fn delete_internal(graph: &mut DbgGraph, candidate: &Candidate) {
    for state in &candidate.internal {
        graph.remove_node(state.node);
    }
}

/// The bubble rule: drop the candidate when it is far weaker than its alternative, or when it reads as error while the
/// alternative reads as genomic.
fn should_delete(candidate: f64, alternative: f64, pop_ratio: f32, coverage: &CoverageRef) -> bool {
    // Strict `<`, so two equally covered branches can never delete each other however the ratio is set.
    //
    // Compared in `f32`, as the ratio is given, and not by widening `pop_ratio` to `f64`:
    // `f64::from(0.1f32)` is 0.10000000149011612, which can overpass the threshold and add random branch removals
    let by_ratio = (candidate as f32) < pop_ratio * alternative as f32;
    let by_coverage = coverage.is_error_like(candidate.round() as u32)
        && coverage.is_genomic(alternative.round() as u32);
    by_ratio || by_coverage
}

pub(crate) fn remove_bulges(graph: &mut DbgGraph, pop_ratio: f32, coverage: &CoverageRef) -> bool {
    let max_length = bulge_max_kmers(graph.k());
    let mut seen = HashSet::new();
    let mut candidates = Vec::new();
    for (entrance, first, edge, _) in candidate_order(graph) {
        if let Some(candidate) = build_candidate(graph, entrance, first, edge, max_length) {
            if seen.insert(candidate.canonical.clone()) {
                candidates.push(candidate);
            }
        }
    }
    candidates.sort_by(|a, b| a.canonical.cmp(&b.canonical));

    let examined = candidates.len();
    let mut collapsed = 0usize;
    let (mut stale, mut no_alt, mut capped) = (0usize, 0usize, 0usize);
    let mut changed = false;
    for candidate in candidates {
        if !candidate_still_matches(graph, &candidate, max_length) {
            stale += 1;
            continue;
        }
        let search = find_alternative(graph, &candidate);
        if matches!(search, AlternativeSearch::Capped(_)) {
            capped += 1;
        }
        let (AlternativeSearch::Complete(Some(alternative))
        | AlternativeSearch::Capped(Some(alternative))) = search
        else {
            no_alt += 1;
            continue;
        };
        if should_delete(
            candidate.coverage.mean(),
            alternative.coverage.mean(),
            pop_ratio,
            coverage,
        ) && candidate_still_matches(graph, &candidate, max_length)
        {
            delete_internal(graph, &candidate);
            collapsed += 1;
            changed = true;
        }
    }

    // Examined vs collapsed, as EC reports: the shape test admits far more candidates than the
    // coverage test removes, and only the second number says whether the rule is doing anything.
    log::info!(
        "Bulge removal: {collapsed}/{examined} bulges collapsed ({stale} stale, {no_alt} no \
             alternative, {capped} search capped, ratio {pop_ratio}, ceiling {:?}). Graph has {} \
             nodes and {} edges",
        coverage.error_ceiling(),
        graph.node_count(),
        graph.edge_count()
    );

    changed
}

fn bounded_coverage(graph: &DbgGraph, first: OrientedNode, forward: bool) -> Option<PathCoverage> {
    let mut current = first;
    let mut coverage = PathCoverage::default();
    let mut seen = HashSet::new();
    loop {
        if !seen.insert(current.node) {
            return None;
        }
        coverage.add_node(graph.node_weight(current.node)?);
        let neighbours = if forward {
            sorted_forward(graph, current)
        } else {
            sorted_backward(graph, current)
        };
        if neighbours.len() != 1 {
            break;
        }
        current = neighbours[0].0;
    }
    (coverage.kmers > 0).then_some(coverage)
}

/// Mean coverage of the competing branches: length-weighted *within* each branch, then averaged
/// unweighted *across* them. Taken from Minia.
fn mean_competitors(coverages: &[PathCoverage]) -> Option<f64> {
    (!coverages.is_empty()).then(|| {
        coverages
            .iter()
            .map(|coverage| coverage.mean())
            .sum::<f64>()
            / coverages.len() as f64
    })
}

fn ec_competing_means(graph: &DbgGraph, candidate: &Candidate) -> Option<(f64, f64)> {
    let mut left = Vec::new();
    for (next, edge) in sorted_forward(graph, candidate.entrance) {
        if next == candidate.first && edge == candidate.first_edge {
            continue;
        }
        left.push(bounded_coverage(graph, next, true)?);
    }

    let connector_last = *candidate.internal.last()?;
    let mut right = Vec::new();
    for (previous, edge) in sorted_backward(graph, candidate.exit) {
        if previous == connector_last && edge == candidate.last_edge {
            continue;
        }
        right.push(bounded_coverage(graph, previous, false)?);
    }
    Some((mean_competitors(&left)?, mean_competitors(&right)?))
}

pub(crate) fn remove_erroneous_connections(
    graph: &mut DbgGraph,
    ratio: f64,
    require_both_flanks: bool,
) -> bool {
    let max_length = EC_LEN_K_MULT.saturating_mul(graph.k());
    let mut seen = HashSet::new();
    let mut candidates = Vec::new();
    for (entrance, first, edge, _) in candidate_order(graph) {
        if let Some(candidate) = build_candidate(graph, entrance, first, edge, max_length) {
            if graph.out_degree_bi(candidate.exit.node, candidate.exit.carry) == 0 {
                continue;
            }
            if seen.insert(candidate.canonical.clone()) {
                candidates.push(candidate);
            }
        }
    }
    candidates.sort_by(|a, b| a.canonical.cmp(&b.canonical));

    let examined = candidates.len();
    let mut removed = 0usize;
    let mut changed = false;
    for candidate in candidates {
        if !candidate_still_matches(graph, &candidate, max_length)
            || graph.out_degree_bi(candidate.exit.node, candidate.exit.carry) == 0
        {
            continue;
        }
        let Some((left, right)) = ec_competing_means(graph, &candidate) else {
            continue;
        };
        let connector = candidate.coverage.mean();
        // Minia ORs the two sides (`Simplifications.cpp:1786`, `isRCTC |= ...`), and we follow it.
        // Its author notes "TODO think hard, is it a |= or a &= ? FIXME for potential misassemblies",
        // so this was measured rather than assumed: across four real libraries OR removed about twice
        // as many connectors as AND, with contigs equal or slightly fewer and no change in
        // misassemblies, mismatches or duplication.
        let weak_enough = if require_both_flanks {
            left > ratio * connector && right > ratio * connector
        } else {
            left > ratio * connector || right > ratio * connector
        };
        if weak_enough && candidate_still_matches(graph, &candidate, max_length) {
            delete_internal(graph, &candidate);
            removed += 1;
            changed = true;
        }
    }

    // Examined vs removed, because the two diverge: the shape test admits far more candidates than
    // the coverage test removes, and only the second number says whether the rule is doing anything.
    log::info!(
        "Erroneous-connection removal: {removed}/{examined} connectors removed (ratio {ratio}, \
             {} flanks). Graph has {} nodes and {} edges",
        if require_both_flanks {
            "both"
        } else {
            "either"
        },
        graph.node_count(),
        graph.edge_count()
    );
    changed
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
    use crate::graph_works::shrinker::Shrinkable;
    use crate::preprocessing::PeakSource;
    use sparrowhawk_graph::NodeStruct;

    /// The ratio arm is strict at its boundary (widening `pop_ratio` to `f64` breaks this: `f64::from(0.1f32)`
    /// is 0.10000000149011612), and the coverage arm fires where the ratio cannot. Peak 100 gives a ceiling of 25.
    #[test]
    fn should_delete_cases() {
        let none = CoverageRef::unknown();
        let known = CoverageRef::new(PeakSource::Fitted(100), 12);
        assert_eq!(known.error_ceiling(), Some(25));
        // (candidate, alternative, coverage, delete)
        for (candidate, alternative, coverage, delete) in [
            (1.0, 10.0, &none, false),
            (0.999, 10.0, &none, true),
            (2.0, 10.0, &none, false),
            (10.0, 10.0, &none, false),
            (20.0, 100.0, &none, false),
            (20.0, 100.0, &known, true),
            (30.0, 100.0, &known, false),
        ] {
            assert_eq!(
                should_delete(candidate, alternative, 0.1, coverage),
                delete,
                "{candidate} against {alternative}, ceiling {:?}",
                coverage.error_ceiling()
            );
        }
    }

    fn node(hash: u64, counts: u32, length: usize) -> NodeStruct {
        NodeStruct {
            counts,
            abs_ind: (hash..hash + length as u64).collect(),
            innerdir: (length > 1).then_some(EdgeType::MinToMin),
        }
    }

    fn add_path(graph: &mut DbgGraph, from: NodeId, nodes: &[NodeId], to: NodeId, edge: EdgeType) {
        let mut previous = from;
        for &next in nodes.iter().chain(std::iter::once(&to)) {
            graph.add_bi_edge(previous, next, edge);
            previous = next;
        }
    }

    fn assert_paired(graph: &DbgGraph) {
        for node in graph.node_indices() {
            for (next, edge) in graph.outgoing_edges(node) {
                assert!(graph.outgoing_edges(next).contains(&(node, edge.rev())));
            }
        }
    }

    fn hashes(graph: &DbgGraph) -> Vec<u64> {
        let mut hashes: Vec<_> = graph
            .node_indices()
            .flat_map(|node| graph.node_weight(node).unwrap().abs_ind.iter().copied())
            .collect();
        hashes.sort_unstable();
        hashes
    }

    #[test]
    fn simple_bulge_is_removed_in_all_orientations() {
        for edge in [
            EdgeType::MinToMin,
            EdgeType::MinToMax,
            EdgeType::MaxToMin,
            EdgeType::MaxToMax,
        ] {
            let mut graph = DbgGraph::new(31);
            let start = graph.add_node(node(1, 20, 1));
            let weak = graph.add_node(node(10, 1, 2));
            let strong = graph.add_node(node(20, 20, 2));
            let exit = graph.add_node(node(30, 20, 1));
            let (_, middle_carry) = edge.get_from_and_to();
            let last_edge = EdgeType::from_carrytypes(middle_carry, middle_carry);
            graph.add_bi_edge(start, weak, edge);
            graph.add_bi_edge(weak, exit, last_edge);
            graph.add_bi_edge(start, strong, edge);
            graph.add_bi_edge(strong, exit, last_edge);
            assert!(remove_bulges(&mut graph, 0.1, &CoverageRef::unknown()));
            assert!(!graph.contains_node(weak));
            assert!(graph.contains_node(strong));
            assert_paired(&graph);
        }
    }

    /// Each bulge is found once from each strand, and the two finds collapse into one. The internal
    /// nodes hold several k-mers, so the reverse strand reads them backwards and only
    /// `canonical_signature` can tell the two finds are the same path.
    #[test]
    fn reverse_strand_candidates_are_deduplicated() {
        let mut graph = DbgGraph::new(31);
        let start = graph.add_node(node(1, 20, 1));
        let weak = graph.add_node(node(10, 1, 3));
        let strong = graph.add_node(node(20, 20, 3));
        let exit = graph.add_node(node(30, 20, 1));
        add_path(&mut graph, start, &[weak], exit, EdgeType::MinToMin);
        add_path(&mut graph, start, &[strong], exit, EdgeType::MinToMin);

        let max_length = bulge_max_kmers(graph.k());
        let mut raw = HashSet::new();
        let mut unique = HashSet::new();
        let mut oriented = 0;
        for (entrance, first, edge, _) in candidate_order(&graph) {
            if let Some(candidate) = build_candidate(&graph, entrance, first, edge, max_length) {
                oriented += 1;
                raw.insert(candidate.signature);
                unique.insert(candidate.canonical);
            }
        }
        assert_eq!(oriented, 4);
        assert_eq!(raw.len(), 4, "each strand reads the path differently");
        assert_eq!(
            unique,
            HashSet::from([vec![10, 11, 12], vec![20, 21, 22]]),
            "one canonical signature per bulge"
        );
    }

    #[test]
    fn multi_unitig_and_multiway_bulges_choose_the_strongest_path() {
        let mut graph = DbgGraph::new(31);
        let start = graph.add_node(node(1, 20, 1));
        let weak_a = graph.add_node(node(10, 1, 2));
        let weak_b = graph.add_node(node(12, 1, 3));
        let medium = graph.add_node(node(20, 5, 5));
        let strong_a = graph.add_node(node(30, 20, 2));
        let strong_b = graph.add_node(node(32, 20, 3));
        let exit = graph.add_node(node(40, 20, 1));
        add_path(
            &mut graph,
            start,
            &[weak_a, weak_b],
            exit,
            EdgeType::MinToMin,
        );
        add_path(&mut graph, start, &[medium], exit, EdgeType::MinToMin);
        add_path(
            &mut graph,
            start,
            &[strong_a, strong_b],
            exit,
            EdgeType::MinToMin,
        );

        assert!(remove_bulges(&mut graph, 0.3, &CoverageRef::unknown()));
        assert!(!graph.contains_node(weak_a));
        assert!(!graph.contains_node(weak_b));
        assert!(!graph.contains_node(medium));
        assert!(graph.contains_node(strong_a));
        assert!(graph.contains_node(strong_b));
        assert_paired(&graph);
    }

    #[test]
    fn nested_bulges_are_resolved_from_the_inside_out() {
        let mut graph = DbgGraph::new(31);
        let outer_start = graph.add_node(node(1, 20, 1));
        let nested_start = graph.add_node(node(10, 1, 1));
        let nested_weak = graph.add_node(node(20, 1, 1));
        let nested_strong = graph.add_node(node(30, 20, 1));
        let nested_exit = graph.add_node(node(40, 1, 1));
        let outer_strong = graph.add_node(node(50, 20, 3));
        let outer_exit = graph.add_node(node(60, 20, 1));
        graph.add_bi_edge(outer_start, nested_start, EdgeType::MinToMin);
        add_path(
            &mut graph,
            nested_start,
            &[nested_weak],
            nested_exit,
            EdgeType::MinToMin,
        );
        add_path(
            &mut graph,
            nested_start,
            &[nested_strong],
            nested_exit,
            EdgeType::MinToMin,
        );
        graph.add_bi_edge(nested_exit, outer_exit, EdgeType::MinToMin);
        add_path(
            &mut graph,
            outer_start,
            &[outer_strong],
            outer_exit,
            EdgeType::MinToMin,
        );

        assert!(remove_bulges(&mut graph, 0.5, &CoverageRef::unknown()));
        assert!(!graph.contains_node(nested_weak));
        graph.shrink();
        assert!(remove_bulges(&mut graph, 0.5, &CoverageRef::unknown()));
        assert!(!hashes(&graph).contains(&10));
        assert!(hashes(&graph).contains(&50));
        assert_paired(&graph);
    }

    /// Three routes from `start` (hash 1) to `exit` (60): `x` (hash 10, 1x, 1 k-mer), `y` (20, 3x, 4
    /// k-mers) and `z` (40, 9x, 7 k-mers). Only `y` is near enough `x`'s length to be its
    /// alternative, and `y`'s best is `z`, so popping `y` before `x` is judged leaves `x` with
    /// nothing to be judged against: the outcome depends on the order candidates are taken in.
    /// Nodes and edges go in forwards or backwards.
    fn competing_bulges(reverse: bool) -> (DbgGraph, std::collections::HashMap<u64, NodeId>) {
        let mut graph = DbgGraph::new(31);
        let mut specs = vec![(1, 20, 1), (10, 1, 1), (20, 3, 4), (40, 9, 7), (60, 20, 1)];
        let mut routes = vec![10, 20, 40];
        if reverse {
            specs.reverse();
            routes.reverse();
        }
        let mut ids = std::collections::HashMap::new();
        for (hash, count, length) in specs {
            ids.insert(hash, graph.add_node(node(hash, count, length)));
        }
        for route in routes {
            add_path(
                &mut graph,
                ids[&1],
                &[ids[&route]],
                ids[&60],
                EdgeType::MinToMin,
            );
        }
        (graph, ids)
    }

    /// Candidates are taken in an order fixed by their hashes, never by how the graph was built.
    /// The fixture is one where the order matters: `x` goes only if it is judged while `y` stands.
    #[test]
    fn correction_is_independent_of_node_and_edge_insertion_order() {
        let (mut forward, _) = competing_bulges(false);
        let (mut reverse, _) = competing_bulges(true);
        assert!(remove_bulges(&mut forward, 0.5, &CoverageRef::unknown()));
        assert!(remove_bulges(&mut reverse, 0.5, &CoverageRef::unknown()));
        let survivors: Vec<u64> = [1].into_iter().chain(40..47).chain([60]).collect();
        assert_eq!(hashes(&forward), survivors, "x, then y, are popped");
        assert_eq!(hashes(&reverse), survivors, "whatever the insertion order");
        assert_paired(&forward);
        assert_paired(&reverse);

        // The order does matter here: with `y` gone first, `x` has no alternative and stays.
        let (mut y_first, ids) = competing_bulges(false);
        y_first.remove_node(ids[&20]);
        assert!(!remove_bulges(&mut y_first, 0.5, &CoverageRef::unknown()));
        assert!(y_first.contains_node(ids[&10]));
    }

    /// Whether the one-node path `entrance -> first -> exit` would be popped against its best
    /// alternative at `pop_ratio`, were it accepted as a candidate at all.
    fn would_pop(
        graph: &DbgGraph,
        entrance: OrientedNode,
        first: OrientedNode,
        exit: OrientedNode,
        pop_ratio: f32,
    ) -> bool {
        let weight = graph.node_weight(first.node).unwrap();
        let mut coverage = PathCoverage::default();
        coverage.add_node(weight);
        let signature = oriented_hashes(graph, first);
        let candidate = Candidate {
            entrance,
            first,
            first_edge: EdgeType::from_carrytypes(entrance.carry, first.carry),
            internal: vec![first],
            exit,
            last_edge: EdgeType::from_carrytypes(first.carry, exit.carry),
            coverage,
            length: weight.abs_ind.len(),
            canonical: canonical_signature(&signature),
            signature,
        };
        match find_alternative(graph, &candidate) {
            AlternativeSearch::Complete(Some(alternative))
            | AlternativeSearch::Capped(Some(alternative)) => should_delete(
                coverage.mean(),
                alternative.coverage.mean(),
                pop_ratio,
                &CoverageRef::unknown(),
            ),
            _ => false,
        }
    }

    fn on(node: NodeId, carry: CarryType) -> OrientedNode {
        OrientedNode { node, carry }
    }

    /// A weak route that loops (through a self-loop, or back into its own entrance) is not a bulge,
    /// though a strong parallel route is there that would pop it if it were.
    #[test]
    fn cycles_are_rejected() {
        use CarryType::Min;

        // The weak node carries a self-loop.
        let mut graph = DbgGraph::new(3);
        let start = graph.add_node(node(1, 20, 1));
        let weak = graph.add_node(node(10, 1, 1));
        let strong = graph.add_node(node(20, 20, 1));
        let exit = graph.add_node(node(30, 20, 1));
        add_path(&mut graph, start, &[weak], exit, EdgeType::MinToMin);
        graph.add_bi_edge(weak, weak, EdgeType::MinToMin);
        add_path(&mut graph, start, &[strong], exit, EdgeType::MinToMin);
        assert!(would_pop(
            &graph,
            on(start, Min),
            on(weak, Min),
            on(exit, Min),
            0.1
        ));
        assert!(!remove_bulges(&mut graph, 0.1, &CoverageRef::unknown()));
        assert!(graph.contains_node(weak));
        assert_paired(&graph);

        // Both routes leave a hub and come back into it on the same strand.
        let mut graph = DbgGraph::new(3);
        let hub = graph.add_node(node(1, 20, 1));
        let weak = graph.add_node(node(10, 1, 1));
        let strong = graph.add_node(node(20, 20, 1));
        add_path(&mut graph, hub, &[weak], hub, EdgeType::MinToMin);
        add_path(&mut graph, hub, &[strong], hub, EdgeType::MinToMin);
        assert!(would_pop(
            &graph,
            on(hub, Min),
            on(weak, Min),
            on(hub, Min),
            0.1
        ));
        assert!(!remove_bulges(&mut graph, 0.1, &CoverageRef::unknown()));
        assert!(graph.contains_node(weak));
        assert_paired(&graph);
    }

    /// A weak route that folds back into its own entrance on the other strand is not a bulge either,
    /// though a strong route folding back the same way would pop it if it were.
    #[test]
    fn foldbacks_are_rejected() {
        use CarryType::{Max, Min};

        let mut graph = DbgGraph::new(3);
        let hub = graph.add_node(node(1, 20, 1));
        let weak = graph.add_node(node(10, 1, 1));
        let strong = graph.add_node(node(20, 20, 1));
        for route in [weak, strong] {
            graph.add_bi_edge(hub, route, EdgeType::MinToMin);
            graph.add_bi_edge(route, hub, EdgeType::MinToMax);
        }
        assert!(would_pop(
            &graph,
            on(hub, Min),
            on(weak, Min),
            on(hub, Max),
            0.1
        ));
        assert!(!remove_bulges(&mut graph, 0.1, &CoverageRef::unknown()));
        assert!(graph.contains_node(weak));
        assert_paired(&graph);
    }

    /// A bulge longer than `bulge_max_kmers` is left alone however weak it is; one at the limit is
    /// popped.
    #[test]
    fn overlong_bulges_are_rejected() {
        let max = bulge_max_kmers(3);
        for (length, popped) in [(max, true), (max + 1, false)] {
            let mut graph = DbgGraph::new(3);
            let start = graph.add_node(node(1, 20, 1));
            let weak = graph.add_node(node(300, 1, length));
            let strong = graph.add_node(node(500, 20, length));
            let exit = graph.add_node(node(700, 20, 1));
            add_path(&mut graph, start, &[weak], exit, EdgeType::MinToMin);
            add_path(&mut graph, start, &[strong], exit, EdgeType::MinToMin);
            assert_eq!(
                remove_bulges(&mut graph, 0.1, &CoverageRef::unknown()),
                popped,
                "{length} k-mers against a limit of {max}"
            );
            assert_eq!(graph.contains_node(weak), !popped, "{length} k-mers");
            assert!(graph.contains_node(strong));
            assert_paired(&graph);
        }
    }

    #[test]
    fn capped_search_still_uses_the_alternative_it_found() {
        let mut graph = DbgGraph::new(3);
        let start = graph.add_node(node(1, 20, 1));
        let weak = graph.add_node(node(10, 1, 1));
        let exit = graph.add_node(node(20, 20, 1));
        add_path(&mut graph, start, &[weak], exit, EdgeType::MinToMin);
        // More seeds than the pop budget, so the search caps with work outstanding. Every one of
        // them reaches the exit, so `best` is set on the very first pop.
        for i in 0..(BULGE_ALT_PATH_POPS + 70) {
            let alternative = graph.add_node(node(100 + (i as u64) * 2, 20, 1));
            add_path(&mut graph, start, &[alternative], exit, EdgeType::MinToMin);
        }
        assert!(remove_bulges(&mut graph, 0.1, &CoverageRef::unknown()));
        assert!(!graph.contains_node(weak));
        assert_paired(&graph);
    }

    /// The other half: capping with nothing carried to the exit still declines, as before.
    #[test]
    fn capped_search_without_an_alternative_makes_no_change() {
        let mut graph = DbgGraph::new(3);
        let start = graph.add_node(node(1, 20, 1));
        let weak = graph.add_node(node(10, 1, 1));
        let strong = graph.add_node(node(9_000, 20, 1));
        let exit = graph.add_node(node(20, 20, 1));
        add_path(&mut graph, start, &[weak], exit, EdgeType::MinToMin);
        add_path(&mut graph, start, &[strong], exit, EdgeType::MinToMin);
        // Dead-end seeds, hashed below `strong` so `state_order` pops them first and the budget is
        // gone before the one real alternative is ever reached.
        for i in 0..(BULGE_ALT_PATH_POPS + 70) {
            let spur = graph.add_node(node(100 + (i as u64) * 2, 20, 1));
            graph.add_bi_edge(start, spur, EdgeType::MinToMin);
        }
        assert!(!remove_bulges(&mut graph, 0.1, &CoverageRef::unknown()));
        assert!(graph.contains_node(weak));
        assert_paired(&graph);
    }

    /// `start` (hash 1) reaches `x` (30) by `p` (10, 1x) or `f` (20, 2x), and `x` goes on to `y`
    /// (50), which `s` (40-41, 50x) also reaches from `start`. Popping `p` leaves `x` with one way
    /// in, so `f`'s candidate no longer ends at `x` but runs on through it to `y`.
    fn reshaped_by_a_pop() -> (DbgGraph, [NodeId; 6]) {
        let mut graph = DbgGraph::new(31);
        let start = graph.add_node(node(1, 20, 1));
        let p = graph.add_node(node(10, 1, 1));
        let f = graph.add_node(node(20, 2, 1));
        let x = graph.add_node(node(30, 20, 1));
        let s = graph.add_node(node(40, 50, 2));
        let y = graph.add_node(node(50, 20, 1));
        let z = graph.add_node(node(60, 20, 1));
        add_path(&mut graph, start, &[p], x, EdgeType::MinToMin);
        add_path(&mut graph, start, &[f], x, EdgeType::MinToMin);
        add_path(&mut graph, x, &[], y, EdgeType::MinToMin);
        add_path(&mut graph, start, &[s], y, EdgeType::MinToMin);
        add_path(&mut graph, y, &[], z, EdgeType::MinToMin);
        (graph, [start, p, f, x, s, y])
    }

    /// A candidate an earlier pop reshaped, with its own first node still there, is stale: it is
    /// skipped for the rest of the pass rather than judged on a shape that no longer exists, and its
    /// new shape is judged on the next pass. Judging the new shape at once would pop `f` and `x`
    /// in the first pass.
    #[test]
    fn stale_candidate_is_not_deleted() {
        use CarryType::Min;

        // How staleness is seen: the candidate rebuilt from its first node differs.
        let (mut graph, [start, p, f, x, _, y]) = reshaped_by_a_pop();
        let max_length = bulge_max_kmers(graph.k());
        let candidate = build_candidate(
            &graph,
            on(start, Min),
            on(f, Min),
            EdgeType::MinToMin,
            max_length,
        )
        .unwrap();
        assert_eq!(candidate.exit, on(x, Min));
        graph.remove_node(p);
        assert!(graph.contains_node(f));
        assert!(!candidate_still_matches(&graph, &candidate, max_length));
        let fresh = build_candidate(
            &graph,
            candidate.entrance,
            candidate.first,
            candidate.first_edge,
            max_length,
        )
        .unwrap();
        assert_eq!(fresh.exit, on(y, Min), "the path now runs through x");

        // And through `remove_bulges`: `p` goes against `f`, which is then stale.
        let (mut graph, [_, p, f, x, s, _]) = reshaped_by_a_pop();
        assert!(remove_bulges(&mut graph, 0.6, &CoverageRef::unknown()));
        assert!(!graph.contains_node(p));
        assert!(graph.contains_node(f), "stale this pass, so not judged");
        assert!(graph.contains_node(x));
        assert_paired(&graph);

        // Next pass, `f -> x` is a fresh candidate and goes against `s`: 11x is under 0.6 * 50x.
        assert!(remove_bulges(&mut graph, 0.6, &CoverageRef::unknown()));
        assert!(!graph.contains_node(f));
        assert!(!graph.contains_node(x));
        assert!(graph.contains_node(s));
        assert_paired(&graph);
    }

    /// The ratio is strict: a flank at exactly 4x the connector (8 against 2) keeps it, even with
    /// both flanks required and the other one far over.
    #[test]
    fn ec_ratio_is_strict_at_exactly_four_times_the_connector() {
        let make_graph = |right_count| {
            let mut graph = DbgGraph::new(3);
            let start = graph.add_node(node(1, 30, 1));
            let connector = graph.add_node(node(10, 2, 1));
            let exit = graph.add_node(node(20, 30, 1));
            let left = graph.add_node(node(30, 20, 1));
            let right = graph.add_node(node(40, right_count, 1));
            let continuation = graph.add_node(node(50, 30, 1));
            add_path(&mut graph, start, &[connector], exit, EdgeType::MinToMin);
            graph.add_bi_edge(start, left, EdgeType::MinToMin);
            graph.add_bi_edge(right, exit, EdgeType::MinToMin);
            graph.add_bi_edge(exit, continuation, EdgeType::MinToMin);
            (graph, connector)
        };

        let (mut removable, connector) = make_graph(20);
        assert!(remove_erroneous_connections(&mut removable, 4.0, true));
        assert!(!removable.contains_node(connector));
        assert_paired(&removable);

        let (mut retained, connector) = make_graph(8);
        assert!(!remove_erroneous_connections(&mut retained, 4.0, true));
        assert!(retained.contains_node(connector));
        assert_paired(&retained);
    }

    #[test]
    fn direct_edges_and_overlong_connectors_are_retained() {
        let mut direct = DbgGraph::new(3);
        let start = direct.add_node(node(1, 20, 1));
        let exit = direct.add_node(node(2, 20, 1));
        let left = direct.add_node(node(3, 20, 1));
        let right = direct.add_node(node(4, 20, 1));
        let continuation = direct.add_node(node(5, 20, 1));
        direct.add_bi_edge(start, exit, EdgeType::MinToMin);
        direct.add_bi_edge(start, left, EdgeType::MinToMin);
        direct.add_bi_edge(right, exit, EdgeType::MinToMin);
        direct.add_bi_edge(exit, continuation, EdgeType::MinToMin);
        assert!(!remove_erroneous_connections(&mut direct, 4.0, true));

        let mut overlong = DbgGraph::new(3);
        let start = overlong.add_node(node(100, 20, 1));
        let connector = overlong.add_node(node(200, 1, 28));
        let exit = overlong.add_node(node(300, 20, 1));
        let left = overlong.add_node(node(400, 20, 1));
        let right = overlong.add_node(node(500, 20, 1));
        let continuation = overlong.add_node(node(600, 20, 1));
        add_path(&mut overlong, start, &[connector], exit, EdgeType::MinToMin);
        overlong.add_bi_edge(start, left, EdgeType::MinToMin);
        overlong.add_bi_edge(right, exit, EdgeType::MinToMin);
        overlong.add_bi_edge(exit, continuation, EdgeType::MinToMin);
        assert!(!remove_erroneous_connections(&mut overlong, 4.0, true));
        assert!(overlong.contains_node(connector));
        assert_paired(&overlong);
    }

    /// We do by default an OR from both flanks with erroneous corrections (from Minia). `--ec-require-both-flanks` switches to removing both. These tests check that the flag does something.
    #[test]
    fn either_flank_removes_what_both_flanks_keeps() {
        // Left flank strong (20 vs connector 2), right flank weak (3): passes on one side only.
        let make = || {
            let mut graph = DbgGraph::new(3);
            let start = graph.add_node(node(1, 30, 1));
            let connector = graph.add_node(node(10, 2, 1));
            let exit = graph.add_node(node(20, 30, 1));
            let left = graph.add_node(node(30, 20, 1));
            let right = graph.add_node(node(40, 3, 1));
            let continuation = graph.add_node(node(50, 30, 1));
            add_path(&mut graph, start, &[connector], exit, EdgeType::MinToMin);
            graph.add_bi_edge(start, left, EdgeType::MinToMin);
            graph.add_bi_edge(right, exit, EdgeType::MinToMin);
            graph.add_bi_edge(exit, continuation, EdgeType::MinToMin);
            (graph, connector)
        };

        let (mut strict, connector) = make();
        assert!(!remove_erroneous_connections(&mut strict, 4.0, true));
        assert!(strict.contains_node(connector), "AND must keep it");

        let (mut lenient, connector) = make();
        assert!(remove_erroneous_connections(&mut lenient, 4.0, false));
        assert!(!lenient.contains_node(connector), "OR must remove it");
        assert_paired(&lenient);
    }

    /// Competing branches are averaged per *route*, not pooled by k-mer — Minia's semantics.
    ///
    /// One long weak branch and one short strong one. Pooling by k-mer would dilute the strong route
    /// (6.3x) and spare the connector; averaging the routes (32.5x) condemns it, because a
    /// well-covered alternative exists regardless of how little sequence it spans.
    #[test]
    fn competing_branches_are_averaged_per_route_not_pooled() {
        let mut graph = DbgGraph::new(3);
        let start = graph.add_node(node(1, 30, 1));
        let connector = graph.add_node(node(10, 4, 1));
        let exit = graph.add_node(node(20, 30, 1));
        // Left competitors: 40 k-mers at 5x, and 1 k-mer at 60x.
        //   mean of means  = (5 + 60) / 2      = 32.5  -> 32.5 > 4*4, would condemn
        //   pooled by kmer = (40*5 + 1*60) / 41 = 6.34  -> 6.34 < 4*4, keeps
        let long_weak = graph.add_node(node(100, 5, 40));
        let short_strong = graph.add_node(node(200, 60, 1));
        let right = graph.add_node(node(40, 30, 1));
        let continuation = graph.add_node(node(50, 30, 1));
        add_path(&mut graph, start, &[connector], exit, EdgeType::MinToMin);
        graph.add_bi_edge(start, long_weak, EdgeType::MinToMin);
        graph.add_bi_edge(start, short_strong, EdgeType::MinToMin);
        graph.add_bi_edge(right, exit, EdgeType::MinToMin);
        graph.add_bi_edge(exit, continuation, EdgeType::MinToMin);

        assert!(remove_erroneous_connections(&mut graph, 4.0, true));
        assert!(
            !graph.contains_node(connector),
            "a short strong alternative route must still condemn the connector"
        );
    }
}
