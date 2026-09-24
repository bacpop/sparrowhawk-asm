//! Native graph-only correction of short low-coverage alternative paths.
//!
//! Erroneous-connection removal and bulge removal are both wired into the correction loop, and both
//! judge a path with `corrector::CoverageRef` rather than a bare ratio.

use crate::algorithms::corrector::{CoverageRef, PathCoverage};
use sparrowhawk_graph::{CarryType, DbgGraph, EdgeType, NodeId};
use std::cmp::Ordering;
use std::collections::HashSet;

const BULGE_LEN_K_MULT: usize = 3;
const BULGE_LEN_K_ADD: usize = 100;
/// Partial paths the alternative search expands before giving up. A pop budget, not a length —
/// Minia's `backtrackingLimit` counts bases traversed (`Simplifications.cpp:1042`).
const BULGE_ALT_PATH_POPS: usize = 131;
const BULGE_RELATIVE_LEN_DELTA: f64 = 0.10;
const BULGE_MIN_LEN_DELTA: usize = 3;

/// Longest candidate the bulge rule considers, in k-mers. Minia's `maxBulgeLength`
/// (`Simplifications.cpp:1298`) is a base count, and n k-mers span `k + n - 1` bases.
fn bulge_max_kmers(k: usize) -> usize {
    let max_bases = (BULGE_LEN_K_MULT * k).max(k + BULGE_LEN_K_ADD);
    max_bases.saturating_sub(k.saturating_sub(1)).max(1)
}
const EC_LEN_K_MULT: usize = 9;
/// Loop guard on the neighbour walk, not a modelling choice.
///
/// Minia walks consecutive unitigs to the next branching node with no bound at all
/// (`GraphUnitigs.cpp:1716-1733`); the "first 100 kmers" this was ported from is a *stale comment* in
/// `Simplifications.cpp:207`, describing a `getSimplePathCoverage` that has no definition anywhere in
/// its tree. Bounding at 100 made a connector's flanks look far weaker than Minia judges them, which
/// suppressed removals. Kept only so a pathological graph cannot spin.
const EC_NEIGHBOUR_LOOKAHEAD_KMERS: usize = usize::MAX;

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

/// The diamond popper's rule (`corrector::choose_branch_by_counts`) applied to a bulge: drop the
/// candidate when it is far weaker than its alternative, or when it reads as error while the
/// alternative reads as genomic.
fn should_delete(candidate: f64, alternative: f64, pop_ratio: f32, coverage: &CoverageRef) -> bool {
    // Strict `<`, so two equally covered branches can never delete each other however the ratio is
    // set. That is exactly what Minia's `covMult > 1` gives up: above 1.0 the test passes in both
    // directions for near-equal branches, and sort order decides which one dies.
    //
    // Compared in `f32`, like `choose_branch_by_counts`, and not by widening `pop_ratio` to `f64`:
    // `f64::from(0.1f32)` is 0.10000000149011612, which lifts the threshold just enough to delete a
    // branch sitting exactly on the boundary.
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
    crate::logw(
        &format!(
            "Bulge removal: {collapsed}/{examined} bulges collapsed ({stale} stale, {no_alt} no \
             alternative, {capped} search capped, ratio {pop_ratio}, ceiling {:?}). Graph has {} \
             nodes and {} edges",
            coverage.error_ceiling(),
            graph.node_count(),
            graph.edge_count()
        ),
        Some("info"),
    );

    changed
}

fn bounded_coverage(graph: &DbgGraph, first: OrientedNode, forward: bool) -> Option<PathCoverage> {
    let mut current = first;
    let mut remaining = EC_NEIGHBOUR_LOOKAHEAD_KMERS;
    let mut coverage = PathCoverage::default();
    let mut seen = HashSet::new();
    while remaining > 0 {
        if !seen.insert(current.node) {
            return None;
        }
        let node = graph.node_weight(current.node)?;
        let consumed = remaining.min(node.abs_ind.len());
        coverage.add_part(node, consumed);
        remaining -= consumed;
        if consumed < node.abs_ind.len() || remaining == 0 {
            break;
        }
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
/// unweighted *across* them.
///
/// That asymmetry is deliberate and matches Minia (`Simplifications.cpp:238` divides by the neighbour
/// count, over values that are each `coverage / seqLength` from `GraphUnitigs.cpp:1521`). The branches
/// are alternative routes, not pieces of one sequence, so what matters is whether *a* route is well
/// covered — pooling them by k-mer would let one long weak route mask a short strong one.
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
    crate::logw(
        &format!(
            "Erroneous-connection removal: {removed}/{examined} connectors removed (ratio {ratio}, \
             {} flanks). Graph has {} nodes and {} edges",
            if require_both_flanks { "both" } else { "either" },
            graph.node_count(),
            graph.edge_count()
        ),
        Some("info"),
    );
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::shrinker::Shrinkable;
    use crate::preprocessing::PeakSource;
    use sparrowhawk_graph::NodeStruct;

    /// The ratio arm must stay strict at the boundary. Widening `pop_ratio` to `f64` breaks this:
    /// `f64::from(0.1f32)` is 0.10000000149011612, so `1.0 < 0.1 * 10.0` flips to true.
    #[test]
    fn the_ratio_arm_is_strict_at_the_boundary() {
        let none = CoverageRef::unknown();
        assert!(!should_delete(1.0, 10.0, 0.1, &none));
        assert!(should_delete(0.999, 10.0, 0.1, &none));
        assert!(!should_delete(10.0, 10.0, 0.1, &none));
    }

    /// The absolute arm fires where the ratio cannot: a branch below the error ceiling against one
    /// above it. Peak 100 at the fitted fraction gives a ceiling of 25.
    #[test]
    fn the_coverage_arm_fires_where_the_ratio_cannot() {
        let known = CoverageRef::new(PeakSource::Fitted(100), 12);
        assert_eq!(known.error_ceiling(), Some(25));
        // 20 is not below 0.1 * 100, so only the absolute arm can delete it.
        assert!(!should_delete(20.0, 100.0, 0.1, &CoverageRef::unknown()));
        assert!(should_delete(20.0, 100.0, 0.1, &known));
        // Both above the ceiling: neither arm applies, whatever the gap.
        assert!(!should_delete(30.0, 100.0, 0.1, &known));
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
    fn path_coverage_weights_nodes_and_partial_prefixes() {
        let short = node(1, 10, 1);
        let long = node(10, 100, 9);
        let mut coverage = PathCoverage::default();
        coverage.add_node(&short);
        coverage.add_node(&long);
        assert_eq!(coverage.mean(), 91.0);

        let mut prefix = PathCoverage::default();
        prefix.add_part(&long, 3);
        assert_eq!(prefix.weighted_sum, 300);
        assert_eq!(prefix.kmers, 3);
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

    #[test]
    fn reverse_strand_candidates_are_deduplicated() {
        let mut graph = DbgGraph::new(31);
        let start = graph.add_node(node(1, 20, 1));
        let weak = graph.add_node(node(10, 1, 1));
        let strong = graph.add_node(node(20, 20, 1));
        let exit = graph.add_node(node(30, 20, 1));
        add_path(&mut graph, start, &[weak], exit, EdgeType::MinToMin);
        add_path(&mut graph, start, &[strong], exit, EdgeType::MinToMin);

        let max_length = bulge_max_kmers(graph.k());
        let mut unique = HashSet::new();
        let mut oriented = 0;
        for (entrance, first, edge, _) in candidate_order(&graph) {
            if let Some(candidate) = build_candidate(&graph, entrance, first, edge, max_length) {
                oriented += 1;
                unique.insert(candidate.canonical);
            }
        }
        assert_eq!(oriented, 4);
        assert_eq!(unique.len(), 2);
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

    fn ordered_bulge(reverse: bool) -> DbgGraph {
        let mut graph = DbgGraph::new(31);
        let specs = [(1, 20), (10, 1), (20, 20), (30, 20)];
        let mut ids = std::collections::HashMap::new();
        let iter: Box<dyn Iterator<Item = &(u64, u32)>> = if reverse {
            Box::new(specs.iter().rev())
        } else {
            Box::new(specs.iter())
        };
        for &(hash, count) in iter {
            ids.insert(hash, graph.add_node(node(hash, count, 1)));
        }
        let paths = [[10], [20]];
        let path_iter: Box<dyn Iterator<Item = &[u64; 1]>> = if reverse {
            Box::new(paths.iter().rev())
        } else {
            Box::new(paths.iter())
        };
        for path in path_iter {
            add_path(
                &mut graph,
                ids[&1],
                &[ids[&path[0]]],
                ids[&30],
                EdgeType::MinToMin,
            );
        }
        graph
    }

    #[test]
    fn correction_is_independent_of_node_and_edge_insertion_order() {
        let mut forward = ordered_bulge(false);
        let mut reverse = ordered_bulge(true);
        assert!(remove_bulges(&mut forward, 0.1, &CoverageRef::unknown()));
        assert!(remove_bulges(&mut reverse, 0.1, &CoverageRef::unknown()));
        assert_eq!(hashes(&forward), hashes(&reverse));
        assert_paired(&forward);
        assert_paired(&reverse);
    }

    #[test]
    fn cycles_foldbacks_and_overlong_bulges_are_rejected() {
        let mut cycle = DbgGraph::new(3);
        let start = cycle.add_node(node(1, 20, 1));
        let a = cycle.add_node(node(10, 1, 1));
        let strong = cycle.add_node(node(20, 20, 1));
        let exit = cycle.add_node(node(30, 20, 1));
        cycle.add_bi_edge(start, a, EdgeType::MinToMin);
        cycle.add_bi_edge(a, a, EdgeType::MinToMin);
        add_path(&mut cycle, start, &[strong], exit, EdgeType::MinToMin);
        assert!(!remove_bulges(&mut cycle, 0.1, &CoverageRef::unknown()));
        assert!(cycle.contains_node(a));

        let mut foldback = DbgGraph::new(3);
        let start = foldback.add_node(node(100, 20, 1));
        let a = foldback.add_node(node(110, 1, 1));
        let b = foldback.add_node(node(120, 1, 1));
        foldback.add_bi_edge(start, a, EdgeType::MinToMin);
        foldback.add_bi_edge(a, b, EdgeType::MinToMin);
        foldback.add_bi_edge(b, a, EdgeType::MinToMin);
        assert!(!remove_bulges(&mut foldback, 0.1, &CoverageRef::unknown()));

        let mut overlong = DbgGraph::new(3);
        let start = overlong.add_node(node(200, 20, 1));
        let weak = overlong.add_node(node(300, 1, 104));
        let strong = overlong.add_node(node(500, 20, 104));
        let exit = overlong.add_node(node(700, 20, 1));
        add_path(&mut overlong, start, &[weak], exit, EdgeType::MinToMin);
        add_path(&mut overlong, start, &[strong], exit, EdgeType::MinToMin);
        assert!(!remove_bulges(&mut overlong, 0.1, &CoverageRef::unknown()));
        assert!(overlong.contains_node(weak));
        assert_paired(&overlong);
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

    #[test]
    fn stale_candidate_is_not_deleted() {
        let mut graph = ordered_bulge(false);
        let max_length = bulge_max_kmers(graph.k());
        let candidate = candidate_order(&graph)
            .into_iter()
            .find_map(|(entrance, first, edge, _)| {
                build_candidate(&graph, entrance, first, edge, max_length)
            })
            .unwrap();
        graph.remove_node(candidate.internal[0].node);
        assert!(!candidate_still_matches(&graph, &candidate, max_length));
        assert_paired(&graph);
    }

    #[test]
    fn similar_coverage_and_strict_boundary_are_retained() {
        for weak_count in [2, 1] {
            let mut graph = DbgGraph::new(31);
            let start = graph.add_node(node(1, 20, 1));
            let weak = graph.add_node(node(10, weak_count, 1));
            let strong = graph.add_node(node(20, 10, 1));
            let exit = graph.add_node(node(30, 20, 1));
            add_path(&mut graph, start, &[weak], exit, EdgeType::MinToMin);
            add_path(&mut graph, start, &[strong], exit, EdgeType::MinToMin);
            assert!(!remove_bulges(&mut graph, 0.1, &CoverageRef::unknown()));
            assert!(graph.contains_node(weak));
            assert_paired(&graph);
        }
    }

    #[test]
    fn ec_requires_strong_alternatives_at_both_ends() {
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
    fn ec_threshold_is_strict() {
        let mut graph = DbgGraph::new(3);
        let start = graph.add_node(node(1, 20, 1));
        let connector = graph.add_node(node(10, 2, 1));
        let exit = graph.add_node(node(20, 20, 1));
        let left = graph.add_node(node(30, 8, 1));
        let right = graph.add_node(node(40, 8, 1));
        let continuation = graph.add_node(node(50, 20, 1));
        add_path(&mut graph, start, &[connector], exit, EdgeType::MinToMin);
        graph.add_bi_edge(start, left, EdgeType::MinToMin);
        graph.add_bi_edge(right, exit, EdgeType::MinToMin);
        graph.add_bi_edge(exit, continuation, EdgeType::MinToMin);
        assert!(!remove_erroneous_connections(&mut graph, 4.0, true));
        assert!(graph.contains_node(connector));
        assert_paired(&graph);
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

    /// Minia ORs the two flank tests; we default to AND. Same graph, opposite outcomes — so the flag
    /// is load-bearing and not decoration.
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
