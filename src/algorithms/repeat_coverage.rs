//! Per-k-mer coverage measurements for repeat motif validation.

use std::collections::{BTreeMap, BTreeSet};

use sparrowhawk_graph::{DbgGraph, NodeStruct};

use crate::indexed_kmers::IndexedKmers;

use super::superbubble::OrientedNode;

pub(crate) const BUBBLE_RATIO_MIN: f64 = 1.8;
pub(crate) const BUBBLE_RATIO_MAX: f64 = 2.2;
pub(crate) const FLOW_MAX_RESIDUAL: f64 = 0.10;
const STEP_RATIO: f64 = 1.20;
const WINDOW: usize = 5;

/// Original, unaggregated counts aligned with the canonical-hash index.
pub(crate) struct CountSnapshot<'a, IntT> {
    kmers: &'a IndexedKmers<IntT>,
    counts: &'a [u32],
}

/// Expanded coverage detail used only by opt-in repeat-recovery debug logging.
#[derive(Debug)]
pub(crate) struct NodeCoverageDiagnostic {
    pub(crate) graph_node_average: Option<u32>,
    pub(crate) kmer_count: usize,
    pub(crate) kmer_mean: Option<f64>,
    pub(crate) kmer_min: Option<u32>,
    pub(crate) kmer_max: Option<u32>,
    pub(crate) missing_counts: usize,
    pub(crate) zero_counts: usize,
    pub(crate) depth_steps: Option<Vec<usize>>,
    pub(crate) homogeneous_mean: Option<f64>,
    pub(crate) issue: Option<String>,
}

impl<'a, IntT> CountSnapshot<'a, IntT> {
    pub(crate) fn new(kmers: &'a IndexedKmers<IntT>, counts: &'a [u32]) -> Self {
        Self { kmers, counts }
    }

    pub(crate) fn count(&self, hash: u64) -> Option<u32> {
        self.kmers.count_from_snapshot(self.counts, hash)
    }

    /// Collect the graph's stored node-average count and the original per-k-mer statistics. This
    /// is deliberately separate from the acceptance logic: callers use it only while debugging.
    pub(crate) fn node_coverage_diagnostic(
        &self,
        graph: &DbgGraph,
        state: OrientedNode,
    ) -> NodeCoverageDiagnostic {
        let graph_node_average = graph.node_weight(state.node).map(|weight| weight.counts);
        let Some(hashes) = self.oriented_hashes(graph, state) else {
            return NodeCoverageDiagnostic {
                graph_node_average,
                kmer_count: 0,
                kmer_mean: None,
                kmer_min: None,
                kmer_max: None,
                missing_counts: 0,
                zero_counts: 0,
                depth_steps: None,
                homogeneous_mean: None,
                issue: Some("graph node or oriented k-mer list is missing".to_owned()),
            };
        };

        let values = hashes
            .iter()
            .map(|&hash| self.count(hash))
            .collect::<Vec<_>>();
        let present = values.iter().filter_map(|value| *value).collect::<Vec<_>>();
        let missing_counts = values.iter().filter(|value| value.is_none()).count();
        let zero_counts = present.iter().filter(|&&value| value == 0).count();
        let kmer_mean = (missing_counts == 0 && !values.is_empty()).then(|| {
            present.iter().map(|&value| f64::from(value)).sum::<f64>() / present.len() as f64
        });
        let kmer_min = present.iter().copied().min();
        let kmer_max = present.iter().copied().max();
        let depth_steps = self.coverage_steps(&hashes);

        let issue = if missing_counts > 0 {
            Some(format!(
                "{missing_counts} k-mer count(s) are missing from the snapshot"
            ))
        } else if let Some(steps) = depth_steps.as_ref().filter(|steps| !steps.is_empty()) {
            Some(format!(
                "supported within-unitig depth step(s) at offsets {steps:?}"
            ))
        } else if zero_counts > 0 {
            Some(format!("{zero_counts} k-mer count(s) are zero"))
        } else if kmer_mean.is_none_or(|mean| mean <= 0.0) {
            Some("per-k-mer mean coverage is not positive".to_owned())
        } else {
            None
        };
        let homogeneous_mean = issue.is_none().then_some(kmer_mean).flatten();

        NodeCoverageDiagnostic {
            graph_node_average,
            kmer_count: hashes.len(),
            kmer_mean,
            kmer_min,
            kmer_max,
            missing_counts,
            zero_counts,
            depth_steps,
            homogeneous_mean,
            issue,
        }
    }

    fn oriented_hashes(&self, graph: &DbgGraph, state: OrientedNode) -> Option<Vec<u64>> {
        let weight = graph.node_weight(state.node)?;
        let mut hashes = weight.abs_ind.clone();
        if let Some(inner) = weight.innerdir {
            if inner.get_from_and_to().0 != state.carry() {
                hashes.reverse();
            }
        }
        Some(hashes)
    }

    /// Measure a unitig only when its per-k-mer counts do not contain a supported depth step.
    pub(crate) fn homogeneous_node_mean(
        &self,
        graph: &DbgGraph,
        state: OrientedNode,
    ) -> Option<f64> {
        let hashes = self.oriented_hashes(graph, state)?;
        if !self.coverage_steps(&hashes)?.is_empty() {
            return None;
        }
        if hashes.iter().any(|&hash| self.count(hash) == Some(0)) {
            return None;
        }
        let mean = mean_counts(&hashes, self)?;
        (mean > 0.0).then_some(mean)
    }

    pub(crate) fn route_mean(
        &self,
        graph: &DbgGraph,
        route: &[OrientedNode],
        include_boundaries: bool,
    ) -> Option<f64> {
        let range = if include_boundaries {
            0..route.len()
        } else {
            1..route.len().checked_sub(1)?
        };
        let mut hashes = Vec::new();
        for index in range {
            hashes.extend(self.oriented_hashes(graph, route[index])?);
        }
        mean_counts(&hashes, self)
    }

    /// Serialise each original unitig as one record, preserving its complete hash order.
    pub(crate) fn route_nodes(
        &self,
        graph: &DbgGraph,
        route: &[OrientedNode],
    ) -> Option<Vec<NodeStruct>> {
        let mut output = Vec::new();
        for &state in route {
            let hashes = self.oriented_hashes(graph, state)?;
            let mean = mean_counts(&hashes, self)?
                .round()
                .clamp(0.0, u32::MAX as f64) as u32;
            output.push(NodeStruct {
                counts: mean,
                abs_ind: hashes,
                innerdir: None,
            });
        }
        Some(output)
    }

    fn coverage_steps(&self, hashes: &[u64]) -> Option<Vec<usize>> {
        if hashes.len() < WINDOW * 2 {
            return Some(Vec::new());
        }
        let values = hashes
            .iter()
            .map(|&hash| self.count(hash).map(f64::from))
            .collect::<Option<Vec<_>>>()?;
        let mut candidates = Vec::new();
        for boundary in WINDOW..=hashes.len() - WINDOW {
            let left = values[boundary - WINDOW..boundary].iter().sum::<f64>() / WINDOW as f64;
            let right = values[boundary..boundary + WINDOW].iter().sum::<f64>() / WINDOW as f64;
            let low = left.min(right);
            let high = left.max(right);
            if high > 0.0 && (low == 0.0 || high / low >= STEP_RATIO) {
                candidates.push((
                    boundary,
                    if low == 0.0 {
                        f64::INFINITY
                    } else {
                        high / low
                    },
                ));
            }
        }
        let mut steps = Vec::new();
        let mut from = 0;
        while from < candidates.len() {
            let start = candidates[from].0;
            let mut to = from + 1;
            while to < candidates.len() && candidates[to].0 - start < WINDOW {
                to += 1;
            }
            let best = candidates[from..to]
                .iter()
                .max_by(|left, right| left.1.total_cmp(&right.1))
                .expect("a step cluster is non-empty");
            steps.push(best.0);
            from = to;
        }
        Some(steps)
    }
}

fn mean_counts<IntT>(hashes: &[u64], counts: &CountSnapshot<'_, IntT>) -> Option<f64> {
    if hashes.is_empty() {
        return None;
    }
    let mut sum = 0.0;
    for &hash in hashes {
        sum += f64::from(counts.count(hash)?);
    }
    Some(sum / hashes.len() as f64)
}

fn within_relative_residual(actual: f64, expected: f64) -> bool {
    let scale = actual.abs().max(expected.abs());
    scale > 0.0 && (actual - expected).abs() / scale <= FLOW_MAX_RESIDUAL
}

/// Does a two-route repeat have the expected one-half branch depth?
pub(crate) fn supports_simple_repeat<IntT>(
    graph: &DbgGraph,
    paths: &[Vec<OrientedNode>],
    counts: &CountSnapshot<'_, IntT>,
) -> bool {
    if paths.len() != 2 || paths.iter().any(|path| path.len() < 3) {
        return false;
    }
    let Some((entrance_mean, exit_mean)) = boundary_means(graph, paths, counts) else {
        return false;
    };
    paths.iter().all(|path| {
        counts.route_mean(graph, path, false).is_some_and(|branch| {
            branch > 0.0 && {
                let entrance_ratio = entrance_mean / branch;
                let exit_ratio = exit_mean / branch;
                (BUBBLE_RATIO_MIN..=BUBBLE_RATIO_MAX).contains(&entrance_ratio)
                    && (BUBBLE_RATIO_MIN..=BUBBLE_RATIO_MAX).contains(&exit_ratio)
            }
        })
    })
}

/// Does a multi-route theta have branch depths that sum to the boundary depth?
pub(crate) fn supports_theta<IntT>(
    graph: &DbgGraph,
    paths: &[Vec<OrientedNode>],
    counts: &CountSnapshot<'_, IntT>,
) -> bool {
    if paths.len() < 3 || paths.iter().any(|path| path.len() < 3) {
        return false;
    }
    let Some((entrance_mean, exit_mean)) = boundary_means(graph, paths, counts) else {
        return false;
    };
    let Some(branch_sum) = paths
        .iter()
        .map(|path| counts.route_mean(graph, path, false))
        .collect::<Option<Vec<_>>>()
        .map(|means| means.into_iter().sum::<f64>())
    else {
        return false;
    };
    within_relative_residual(entrance_mean, branch_sum)
        && within_relative_residual(exit_mean, branch_sum)
}

fn boundary_means<IntT>(
    graph: &DbgGraph,
    paths: &[Vec<OrientedNode>],
    counts: &CountSnapshot<'_, IntT>,
) -> Option<(f64, f64)> {
    let first = paths.first()?;
    let entrance = *first.first()?;
    let exit = *first.last()?;
    if paths
        .iter()
        .any(|path| path.first() != Some(&entrance) || path.last() != Some(&exit))
    {
        return None;
    }
    let entrance_hashes = counts.oriented_hashes(graph, entrance)?;
    let exit_hashes = counts.oriented_hashes(graph, exit)?;
    Some((
        mean_counts(&entrance_hashes, counts)?,
        mean_counts(&exit_hashes, counts)?,
    ))
}

/// Return length-weighted means for every motif node, rejecting missing counts, zero-depth nodes,
/// and supported internal depth steps before the motif is considered for extraction.
pub(crate) fn homogeneous_node_means<IntT>(
    graph: &DbgGraph,
    paths: &[Vec<OrientedNode>],
    counts: &CountSnapshot<'_, IntT>,
) -> Option<BTreeMap<OrientedNode, f64>> {
    let mut means = BTreeMap::new();
    for state in paths.iter().flatten().copied() {
        if let std::collections::btree_map::Entry::Vacant(entry) = means.entry(state) {
            entry.insert(counts.homogeneous_node_mean(graph, state)?);
        }
    }
    Some(means)
}

/// Coverage conservation across every split and merge in a non-parallel superbubble.
pub(crate) fn supports_flow_diamond(
    paths: &[Vec<OrientedNode>],
    means: &BTreeMap<OrientedNode, f64>,
) -> bool {
    if paths.len() < 3 || paths.iter().any(|path| path.len() < 3) {
        return false;
    }
    let mut outgoing: BTreeMap<OrientedNode, BTreeSet<OrientedNode>> = BTreeMap::new();
    let mut incoming: BTreeMap<OrientedNode, BTreeSet<OrientedNode>> = BTreeMap::new();
    for path in paths {
        for pair in path.windows(2) {
            outgoing.entry(pair[0]).or_default().insert(pair[1]);
            incoming.entry(pair[1]).or_default().insert(pair[0]);
        }
    }
    let nodes = outgoing
        .keys()
        .chain(incoming.keys())
        .copied()
        .collect::<BTreeSet<_>>();
    let mut checked = false;
    for node in nodes {
        let Some(&mean) = means.get(&node) else {
            return false;
        };
        if let Some(next) = outgoing.get(&node).filter(|edges| edges.len() > 1) {
            let Some(sum) = next
                .iter()
                .map(|state| means.get(state).copied())
                .collect::<Option<Vec<_>>>()
                .map(|values| values.into_iter().sum::<f64>())
            else {
                return false;
            };
            checked = true;
            if !within_relative_residual(mean, sum) {
                return false;
            }
        }
        if let Some(previous) = incoming.get(&node).filter(|edges| edges.len() > 1) {
            let Some(sum) = previous
                .iter()
                .map(|state| means.get(state).copied())
                .collect::<Option<Vec<_>>>()
                .map(|values| values.into_iter().sum::<f64>())
            else {
                return false;
            };
            checked = true;
            if !within_relative_residual(mean, sum) {
                return false;
            }
        }
    }
    // Unitig-to-unitig edges with exactly one predecessor and one successor should preserve depth.
    for (&from, next) in &outgoing {
        if next.len() != 1 {
            continue;
        }
        let to = *next.first().expect("one outgoing motif edge");
        if incoming
            .get(&to)
            .is_some_and(|previous| previous.len() == 1)
        {
            let (Some(&from_mean), Some(&to_mean)) = (means.get(&from), means.get(&to)) else {
                return false;
            };
            checked = true;
            if !within_relative_residual(from_mean, to_mean) {
                return false;
            }
        }
    }
    checked
}

/// Human-readable coverage evidence matching `supports_simple_repeat`, for rejected candidates.
pub(crate) fn explain_simple_repeat<IntT>(
    graph: &DbgGraph,
    paths: &[Vec<OrientedNode>],
    counts: &CountSnapshot<'_, IntT>,
) -> String {
    if paths.len() != 2 || paths.iter().any(|path| path.len() < 3) {
        return format!(
            "simple-repeat check requires exactly two routes of at least three nodes (got {} routes)",
            paths.len()
        );
    }
    let Some((entrance_mean, exit_mean)) = boundary_means(graph, paths, counts) else {
        return "simple-repeat check could not measure a shared entrance/exit boundary".to_owned();
    };

    let mut accepted = true;
    let mut branches = Vec::new();
    for (index, path) in paths.iter().enumerate() {
        let Some(branch_mean) = counts.route_mean(graph, path, false) else {
            accepted = false;
            branches.push(format!("route{}=unmeasurable", index + 1));
            continue;
        };
        if branch_mean <= 0.0 {
            accepted = false;
            branches.push(format!(
                "route{}=non-positive branch mean {branch_mean:.4}",
                index + 1
            ));
            continue;
        }
        let entrance_ratio = entrance_mean / branch_mean;
        let exit_ratio = exit_mean / branch_mean;
        let route_ok = (BUBBLE_RATIO_MIN..=BUBBLE_RATIO_MAX).contains(&entrance_ratio)
            && (BUBBLE_RATIO_MIN..=BUBBLE_RATIO_MAX).contains(&exit_ratio);
        accepted &= route_ok;
        branches.push(format!(
            "route{} branch_mean={branch_mean:.4}, entrance/branch={entrance_ratio:.4}, exit/branch={exit_ratio:.4}, each_required_in=[{BUBBLE_RATIO_MIN:.2},{BUBBLE_RATIO_MAX:.2}], pass={route_ok}",
            index + 1
        ));
    }
    format!(
        "simple-repeat coverage: graph-count-derived entrance_mean={entrance_mean:.4}, exit_mean={exit_mean:.4}; {}; result={}",
        branches.join("; "),
        if accepted { "supported" } else { "rejected" }
    )
}

/// Human-readable coverage evidence matching `supports_theta`, including the residual calculation.
pub(crate) fn explain_theta<IntT>(
    graph: &DbgGraph,
    paths: &[Vec<OrientedNode>],
    counts: &CountSnapshot<'_, IntT>,
) -> String {
    if paths.len() < 3 || paths.iter().any(|path| path.len() < 3) {
        return format!(
            "theta check requires at least three routes of at least three nodes (got {} routes)",
            paths.len()
        );
    }
    let Some((entrance_mean, exit_mean)) = boundary_means(graph, paths, counts) else {
        return "theta check could not measure a shared entrance/exit boundary".to_owned();
    };
    let Some(branch_means) = paths
        .iter()
        .map(|path| counts.route_mean(graph, path, false))
        .collect::<Option<Vec<_>>>()
    else {
        return "theta check could not measure one or more branch means".to_owned();
    };
    let branch_sum = branch_means.iter().sum::<f64>();
    let entrance_residual = relative_residual(entrance_mean, branch_sum);
    let exit_residual = relative_residual(exit_mean, branch_sum);
    let accepted = entrance_residual.is_some_and(|value| value <= FLOW_MAX_RESIDUAL)
        && exit_residual.is_some_and(|value| value <= FLOW_MAX_RESIDUAL);
    format!(
        "theta coverage: graph-count-derived entrance_mean={entrance_mean:.4}, exit_mean={exit_mean:.4}; branch_means={branch_means:.4?}, branch_sum={branch_sum:.4}; residual=abs(boundary-branch_sum)/max(abs(boundary),abs(branch_sum)); entrance_residual={entrance_residual:?}, exit_residual={exit_residual:?}, maximum={FLOW_MAX_RESIDUAL:.3}, result={}",
        if accepted { "supported" } else { "rejected" }
    )
}

/// Human-readable split/merge and linear-edge equations matching `supports_flow_diamond`.
pub(crate) fn explain_flow_diamond(
    paths: &[Vec<OrientedNode>],
    means: &BTreeMap<OrientedNode, f64>,
) -> String {
    if paths.len() < 3 || paths.iter().any(|path| path.len() < 3) {
        return format!(
            "flow-diamond check requires at least three routes of at least three nodes (got {} routes)",
            paths.len()
        );
    }
    let mut outgoing: BTreeMap<OrientedNode, BTreeSet<OrientedNode>> = BTreeMap::new();
    let mut incoming: BTreeMap<OrientedNode, BTreeSet<OrientedNode>> = BTreeMap::new();
    for path in paths {
        for pair in path.windows(2) {
            outgoing.entry(pair[0]).or_default().insert(pair[1]);
            incoming.entry(pair[1]).or_default().insert(pair[0]);
        }
    }
    let nodes = outgoing
        .keys()
        .chain(incoming.keys())
        .copied()
        .collect::<BTreeSet<_>>();
    let mut checked = false;
    let mut all_pass = true;
    let mut equations = Vec::new();
    for node in nodes {
        let Some(&mean) = means.get(&node) else {
            all_pass = false;
            equations.push(format!("node {node:?}: no homogeneous coverage mean"));
            continue;
        };
        if let Some(next) = outgoing.get(&node).filter(|edges| edges.len() > 1) {
            match next
                .iter()
                .map(|state| means.get(state).copied())
                .collect::<Option<Vec<_>>>()
            {
                Some(values) => {
                    let sum = values.iter().sum::<f64>();
                    let residual = relative_residual(mean, sum);
                    let pass = residual.is_some_and(|value| value <= FLOW_MAX_RESIDUAL);
                    checked = true;
                    all_pass &= pass;
                    equations.push(format!(
                        "split at {node:?}: node_mean={mean:.4} vs successor_means={values:.4?} sum={sum:.4}, residual={residual:?}, pass={pass}"
                    ));
                }
                None => {
                    all_pass = false;
                    equations.push(format!(
                        "split at {node:?}: a successor has no homogeneous mean"
                    ));
                }
            }
        }
        if let Some(previous) = incoming.get(&node).filter(|edges| edges.len() > 1) {
            match previous
                .iter()
                .map(|state| means.get(state).copied())
                .collect::<Option<Vec<_>>>()
            {
                Some(values) => {
                    let sum = values.iter().sum::<f64>();
                    let residual = relative_residual(mean, sum);
                    let pass = residual.is_some_and(|value| value <= FLOW_MAX_RESIDUAL);
                    checked = true;
                    all_pass &= pass;
                    equations.push(format!(
                        "merge at {node:?}: node_mean={mean:.4} vs predecessor_means={values:.4?} sum={sum:.4}, residual={residual:?}, pass={pass}"
                    ));
                }
                None => {
                    all_pass = false;
                    equations.push(format!(
                        "merge at {node:?}: a predecessor has no homogeneous mean"
                    ));
                }
            }
        }
    }
    for (&from, next) in &outgoing {
        if next.len() != 1 {
            continue;
        }
        let to = *next.first().expect("one outgoing motif edge");
        if incoming
            .get(&to)
            .is_some_and(|previous| previous.len() == 1)
        {
            match (means.get(&from), means.get(&to)) {
                (Some(&from_mean), Some(&to_mean)) => {
                    let residual = relative_residual(from_mean, to_mean);
                    let pass = residual.is_some_and(|value| value <= FLOW_MAX_RESIDUAL);
                    checked = true;
                    all_pass &= pass;
                    equations.push(format!(
                        "linear edge {from:?}->{to:?}: means={from_mean:.4}/{to_mean:.4}, residual={residual:?}, pass={pass}"
                    ));
                }
                _ => {
                    all_pass = false;
                    equations.push(format!(
                        "linear edge {from:?}->{to:?}: missing homogeneous mean"
                    ));
                }
            }
        }
    }
    let accepted = checked && all_pass;
    format!(
        "flow-diamond coverage equations (relative residual limit={FLOW_MAX_RESIDUAL:.3}): {}; result={}",
        equations.join("; "),
        if accepted { "supported" } else { "rejected" }
    )
}

fn relative_residual(actual: f64, expected: f64) -> Option<f64> {
    let scale = actual.abs().max(expected.abs());
    (scale > 0.0).then(|| (actual - expected).abs() / scale)
}

/// Pairwise-disjoint internal routes that together cover the superbubble's interior.
pub(crate) fn is_parallel_bundle(paths: &[Vec<OrientedNode>], interior: &[OrientedNode]) -> bool {
    let mut covered = BTreeSet::new();
    for path in paths {
        if path.len() < 3 {
            return false;
        }
        for state in &path[1..path.len() - 1] {
            if !covered.insert(*state) {
                return false;
            }
        }
    }
    covered == interior.iter().copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexed_kmers::IndexedKmers;
    use sparrowhawk_graph::{CarryType, NodeStruct};

    fn graph_from_units(
        unit_counts: &[Vec<u32>],
    ) -> (DbgGraph, IndexedKmers<u64>, Vec<u32>, Vec<OrientedNode>) {
        let mut graph = DbgGraph::new(3);
        let mut kmers = IndexedKmers::<u64>::with_capacity(unit_counts.iter().map(Vec::len).sum());
        let mut flat_counts = Vec::new();
        let mut states = Vec::new();
        let mut hash = 100_u64;
        for unit in unit_counts {
            let hashes = unit
                .iter()
                .map(|&count| {
                    hash += 1;
                    kmers.push(hash, hash + 1000, 0, count, hash);
                    flat_counts.push(count);
                    hash
                })
                .collect();
            let node = graph.add_node(NodeStruct {
                counts: unit.iter().sum::<u32>() / unit.len() as u32,
                abs_ind: hashes,
                innerdir: None,
            });
            states.push(OrientedNode::new(node, CarryType::Min));
        }
        (graph, kmers, flat_counts, states)
    }

    #[test]
    fn per_kmer_counts_are_weighted_across_a_route() {
        let mut kmers = IndexedKmers::<u64>::with_capacity(3);
        kmers.push(11, 101, 0, 10, 11);
        kmers.push(12, 102, 0, 20, 12);
        kmers.push(13, 103, 0, 40, 13);
        let counts = vec![10, 20, 40];
        let snapshot = CountSnapshot::new(&kmers, &counts);

        assert_eq!(mean_counts(&[11, 12, 13], &snapshot), Some(70.0 / 3.0));
    }

    #[test]
    fn depth_step_requires_five_kmers_on_both_sides_and_twenty_percent_change() {
        let mut kmers = IndexedKmers::<u64>::with_capacity(12);
        for i in 0..12_u64 {
            kmers.push(i + 1, i + 101, 0, if i < 6 { 10 } else { 13 }, i + 1);
        }
        let counts = (0..12)
            .map(|i| if i < 6 { 10 } else { 13 })
            .collect::<Vec<_>>();
        let snapshot = CountSnapshot::new(&kmers, &counts);
        assert_eq!(
            snapshot.coverage_steps(&(1..=12).collect::<Vec<_>>()),
            Some(vec![6])
        );
        assert!(snapshot
            .coverage_steps(&(1..=9).collect::<Vec<_>>())
            .unwrap()
            .is_empty());
    }

    #[test]
    fn simple_repeat_checks_entrance_and_exit_separately() {
        for entrance in [18, 20, 22] {
            let (graph, kmers, counts, state) =
                graph_from_units(&[vec![entrance], vec![10], vec![10], vec![20]]);
            let paths = vec![
                vec![state[0], state[1], state[3]],
                vec![state[0], state[2], state[3]],
            ];
            assert!(supports_simple_repeat(
                &graph,
                &paths,
                &CountSnapshot::new(&kmers, &counts)
            ));
        }

        for boundaries in [[10, 30], [1, 39]] {
            let (graph, kmers, counts, state) =
                graph_from_units(&[vec![boundaries[0]], vec![10], vec![10], vec![boundaries[1]]]);
            let paths = vec![
                vec![state[0], state[1], state[3]],
                vec![state[0], state[2], state[3]],
            ];
            assert!(!supports_simple_repeat(
                &graph,
                &paths,
                &CountSnapshot::new(&kmers, &counts)
            ));
        }
    }

    #[test]
    fn boundary_means_are_length_weighted_and_theta_checks_each_end() {
        let (graph, kmers, counts, state) =
            graph_from_units(&[vec![10, 30], vec![10], vec![10], vec![20]]);
        let paths = vec![
            vec![state[0], state[1], state[3]],
            vec![state[0], state[2], state[3]],
        ];
        assert!(supports_simple_repeat(
            &graph,
            &paths,
            &CountSnapshot::new(&kmers, &counts)
        ));

        let (graph, kmers, counts, state) =
            graph_from_units(&[vec![30], vec![9], vec![9], vec![9], vec![27]]);
        let theta = vec![
            vec![state[0], state[1], state[4]],
            vec![state[0], state[2], state[4]],
            vec![state[0], state[3], state[4]],
        ];
        assert!(supports_theta(
            &graph,
            &theta,
            &CountSnapshot::new(&kmers, &counts)
        ));

        let (graph, kmers, counts, state) =
            graph_from_units(&[vec![10], vec![10], vec![10], vec![10], vec![50]]);
        let theta = vec![
            vec![state[0], state[1], state[4]],
            vec![state[0], state[2], state[4]],
            vec![state[0], state[3], state[4]],
        ];
        assert!(!supports_theta(
            &graph,
            &theta,
            &CountSnapshot::new(&kmers, &counts)
        ));
    }

    #[test]
    fn missing_count_is_not_treated_as_no_coverage_step() {
        let (graph, kmers, counts, state) =
            graph_from_units(&[vec![10, 10, 10, 10, 10, 20, 20, 20, 20, 20]]);
        let snapshot = CountSnapshot::new(&kmers, &counts[..counts.len() - 1]);
        assert_eq!(snapshot.homogeneous_node_mean(&graph, state[0]), None);
    }

    #[test]
    fn supported_internal_step_makes_a_unitig_unusable_for_recovery() {
        let unit = (0..12)
            .map(|index| if index < 6 { 10 } else { 13 })
            .collect::<Vec<_>>();
        let (graph, kmers, counts, states) = graph_from_units(&[unit]);
        assert_eq!(
            CountSnapshot::new(&kmers, &counts).homogeneous_node_mean(&graph, states[0]),
            None
        );

        let unit = vec![0, 0, 0, 0, 0, 20, 20, 20, 20, 20];
        let (graph, kmers, counts, states) = graph_from_units(&[unit]);
        assert_eq!(
            CountSnapshot::new(&kmers, &counts).homogeneous_node_mean(&graph, states[0]),
            None,
            "zero-to-positive depth changes must be rejected"
        );
    }

    #[test]
    fn flow_diamond_rejects_coverage_spike_along_a_linear_section() {
        let (graph, kmers, counts, state) = graph_from_units(&[
            vec![4],
            vec![3],
            vec![1],
            vec![1],
            vec![200],
            vec![1],
            vec![2],
            vec![3],
            vec![1],
            vec![4],
        ]);
        let paths = vec![
            vec![
                state[0], state[1], state[2], state[3], state[4], state[5], state[9],
            ],
            vec![state[0], state[1], state[6], state[7], state[9]],
            vec![state[0], state[8], state[7], state[9]],
        ];
        let means = homogeneous_node_means(&graph, &paths, &CountSnapshot::new(&kmers, &counts))
            .expect("single-kmer unitigs have no internal step");
        assert!(!supports_flow_diamond(&paths, &means));
    }
}
