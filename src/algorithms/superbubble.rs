//! Bounded superbubble search over oriented graph nodes.

use std::collections::BTreeSet;

use sparrowhawk_graph::{CarryType, DbgGraph, EdgeType, NodeId};

const MAX_INTERIOR_STATES: usize = 60;
const MAX_PATHS: usize = 16;

/// One graph unitig as traversed on one strand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct OrientedNode {
    pub(crate) node: NodeId,
    pub(crate) reverse: bool,
}

impl OrientedNode {
    pub(crate) fn new(node: NodeId, carry: CarryType) -> Self {
        Self {
            node,
            reverse: carry == CarryType::Max,
        }
    }

    pub(crate) fn carry(self) -> CarryType {
        if self.reverse {
            CarryType::Max
        } else {
            CarryType::Min
        }
    }

    pub(crate) fn reverse_complement(self) -> Self {
        Self {
            node: self.node,
            reverse: !self.reverse,
        }
    }
}

/// A bounded acyclic fork and its complete entrance-to-exit routes.
#[derive(Clone, Debug)]
pub(crate) struct Superbubble {
    pub(crate) entrance: OrientedNode,
    pub(crate) exit: OrientedNode,
    pub(crate) interior: Vec<OrientedNode>,
    /// Complete routes, including both boundary nodes.
    pub(crate) paths: Vec<Vec<OrientedNode>>,
}

fn successors(graph: &DbgGraph, state: OrientedNode) -> Vec<(OrientedNode, EdgeType)> {
    let mut neighbours = graph
        .out_neighbours_bi(state.node, state.carry())
        .into_iter()
        .map(|(node, edge)| {
            let (_, carry) = edge.get_from_and_to();
            (OrientedNode::new(node, carry), edge)
        })
        .collect::<Vec<_>>();
    neighbours.sort_by_key(|(state, _)| *state);
    neighbours
}

fn predecessors(graph: &DbgGraph, state: OrientedNode) -> Vec<(OrientedNode, EdgeType)> {
    let mut neighbours = graph
        .in_neighbours_bi(state.node, state.carry())
        .into_iter()
        .map(|(node, edge)| {
            let (carry, _) = edge.get_from_and_to();
            (OrientedNode::new(node, carry), edge)
        })
        .collect::<Vec<_>>();
    neighbours.sort_by_key(|(state, _)| *state);
    neighbours
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SearchFailure {
    NoExit,
    Cycle,
    FoldBack,
    TooLarge,
    Tip,
}

fn find_one(graph: &DbgGraph, entrance: OrientedNode) -> Result<Superbubble, SearchFailure> {
    let mut visited = BTreeSet::from([entrance]);
    let mut frontier = successors(graph, entrance)
        .into_iter()
        .map(|(state, _)| state)
        .collect::<BTreeSet<_>>();

    loop {
        if frontier.is_empty() {
            return Err(SearchFailure::NoExit);
        }

        if frontier.len() == 1 {
            let exit = *frontier.first().expect("one frontier item");
            if predecessors(graph, exit)
                .iter()
                .all(|(pred, _)| visited.contains(pred))
            {
                if exit == entrance || successors(graph, exit).iter().any(|(s, _)| *s == entrance) {
                    return Err(SearchFailure::Cycle);
                }
                if visited.contains(&exit.reverse_complement()) {
                    return Err(SearchFailure::FoldBack);
                }
                let interior = visited
                    .iter()
                    .copied()
                    .filter(|state| *state != entrance)
                    .collect::<Vec<_>>();
                let inside = interior.iter().copied().collect::<BTreeSet<_>>();
                let mut paths = Vec::new();
                let mut path = Vec::new();
                enumerate_paths(graph, entrance, exit, &inside, &mut path, &mut paths);
                if paths.len() < 2 {
                    return Err(SearchFailure::NoExit);
                }
                return Ok(Superbubble {
                    entrance,
                    exit,
                    interior,
                    paths,
                });
            }
        }

        let next = frontier.iter().copied().find(|candidate| {
            predecessors(graph, *candidate)
                .iter()
                .all(|(pred, _)| visited.contains(pred))
        });
        let Some(next) = next else {
            return Err(SearchFailure::NoExit);
        };
        if next == entrance {
            return Err(SearchFailure::Cycle);
        }
        if visited.contains(&next.reverse_complement()) {
            return Err(SearchFailure::FoldBack);
        }
        if visited.len() > MAX_INTERIOR_STATES {
            return Err(SearchFailure::TooLarge);
        }
        if successors(graph, next).is_empty() {
            return Err(SearchFailure::Tip);
        }

        frontier.remove(&next);
        visited.insert(next);
        for (successor, _) in successors(graph, next) {
            if !visited.contains(&successor) {
                frontier.insert(successor);
            }
        }
    }
}

fn enumerate_paths(
    graph: &DbgGraph,
    entrance: OrientedNode,
    exit: OrientedNode,
    interior: &BTreeSet<OrientedNode>,
    path: &mut Vec<OrientedNode>,
    output: &mut Vec<Vec<OrientedNode>>,
) {
    if output.len() > MAX_PATHS {
        return;
    }
    let current = path.last().copied().unwrap_or(entrance);
    for (next, _) in successors(graph, current) {
        if next == exit {
            // The seventeenth route is the first one that makes this candidate unusable.
            if output.len() == MAX_PATHS {
                output.push(Vec::new());
                return;
            }
            let mut route = Vec::with_capacity(path.len() + 2);
            route.push(entrance);
            route.extend(path.iter().copied());
            route.push(exit);
            output.push(route);
        } else if interior.contains(&next) && !path.contains(&next) {
            path.push(next);
            enumerate_paths(graph, entrance, exit, interior, path, output);
            path.pop();
            if output.len() > MAX_PATHS {
                return;
            }
        }
    }
}

fn strand_twin_key(entrance: OrientedNode, exit: OrientedNode) -> (OrientedNode, OrientedNode) {
    let forward = (entrance, exit);
    let reverse = (exit.reverse_complement(), entrance.reverse_complement());
    forward.min(reverse)
}

/// Find bounded superbubbles, deduplicated against their reverse-complement traversal.
pub(crate) fn find_all(graph: &DbgGraph) -> Vec<Superbubble> {
    let mut starts = graph
        .node_indices()
        .flat_map(|node| {
            [CarryType::Min, CarryType::Max]
                .into_iter()
                .map(move |carry| OrientedNode::new(node, carry))
        })
        .collect::<Vec<_>>();
    starts.sort();

    let mut seen = BTreeSet::new();
    let mut output = Vec::new();
    for entrance in starts {
        if successors(graph, entrance).len() < 2 {
            continue;
        }
        if let Ok(candidate) = find_one(graph, entrance) {
            if candidate.paths.iter().any(|path| path.is_empty()) {
                continue;
            }
            if candidate.paths.len() > MAX_PATHS {
                continue;
            }
            let key = strand_twin_key(candidate.entrance, candidate.exit);
            if seen.insert(key) {
                output.push(candidate);
            }
        }
    }
    output.sort_by_key(|candidate| (candidate.entrance, candidate.exit));
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparrowhawk_graph::NodeStruct;

    fn test_node(hash: u64) -> NodeStruct {
        NodeStruct {
            counts: 10,
            abs_ind: vec![hash],
            innerdir: None,
        }
    }

    #[test]
    fn finds_one_bubble_and_its_two_routes() {
        let mut graph = DbgGraph::new(3);
        let (left, a, b, right) = (
            graph.add_node(test_node(1)),
            graph.add_node(test_node(2)),
            graph.add_node(test_node(3)),
            graph.add_node(test_node(4)),
        );
        for (from, to) in [(left, a), (a, right), (left, b), (b, right)] {
            graph.add_bi_edge(from, to, EdgeType::MinToMin);
        }

        let bubbles = find_all(&graph);
        assert_eq!(bubbles.len(), 1);
        assert_eq!(bubbles[0].paths.len(), 2);
        assert!(bubbles[0].paths.iter().all(|path| path.len() == 3));
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn rejects_the_seventeenth_route_instead_of_truncating_silently() {
        let mut graph = DbgGraph::new(3);
        let left = graph.add_node(test_node(1));
        let right = graph.add_node(test_node(2));
        let mut middles = Vec::new();
        for hash in 3..20 {
            middles.push(graph.add_node(test_node(hash)));
        }
        for middle in middles {
            graph.add_bi_edge(left, middle, EdgeType::MinToMin);
            graph.add_bi_edge(middle, right, EdgeType::MinToMin);
        }

        assert!(find_all(&graph).is_empty());
        assert!(graph.validate().is_ok());
    }
}
