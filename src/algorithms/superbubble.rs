//! Finding superbubbles, and asking the multi-k oracle about them.
//!
//! Everything the assembler corrects today is a **simple bubble**: `corrector::bubble_parts` demands
//! `out_degree(start) == 3` and exactly two `Min`-carry start edges, and `check_bubble_structure` then
//! requires each branch to be a *single* node with in- and out-degree 1, both reaching the same end. A
//! three-way fork, a branch two unitigs long, or a bubble nested inside another is not recognised at
//! all — it simply breaks the contig.
//!
//! This module recognises them, and asks the same question of them that `multik::judge_bubble` asks of
//! a simple bubble: does the evidence k corroborate one path, several, or none? It is **read-only on
//! the graph**. That is the point — it produces the number that says whether acting on these is worth
//! anything, at zero risk to the assembly.
//!
//! The bidirected graph makes two things different from a textbook superbubble search, and both are
//! easy to get wrong:
//!
//! - A traversal state is a node *plus the strand it is read on*, `(NodeId, CarryType)`. Successors of
//!   a state come from `out_neighbours_bi`, and the next state's carry is the edge's **target** carry;
//!   walking backwards it is the **source** carry.
//! - Every superbubble exists twice, once per strand. `(s, t)` and `(RC(t), RC(s))` are the same
//!   object and must be counted once — and the pair to canonicalise is that one, *not* the unordered
//!   `{s, t}`.

use nohash_hasher::NoHashHasher;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    hash::BuildHasherDefault,
};

use sparrowhawk_graph::{CarryType, DbgGraph, EdgeType, NodeId, NodeStruct};

use crate::algorithms::multik::{
    gather_left, gather_right, oriented, spell_branch, BranchEvidence, BranchWalk, EvidenceGraph,
    SplitOutcome, FLANK_SUPPORT, MAX_SHARED_CHAIN,
};
use crate::bit_encoding::UInt;
use crate::graph_works::spell_path;

/// Interior states we are willing to explore before giving up on an entrance.
///
/// A guard against pathological tangles, not a tuning knob: the flank census put the contigs around
/// these structures at a median 38-69 bp, so a real superbubble here spans a handful of unitigs. Cap
/// hits are counted and reported rather than silently dropping a locus.
const MAX_SB_INTERIOR: usize = 60;

/// Paths we are willing to enumerate between entrance and exit. Also reported when hit.
const MAX_SB_PATHS: usize = 16;

/// A node together with the strand it is traversed on — the state a walk is actually in.
///
/// `CarryType` derives no `Ord` and no `Hash`, and it lives in the graph crate, which is a git
/// dependency; so the total order that the search and the strand dedup both need is defined here
/// rather than there.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Oriented {
    /// The unitig.
    pub node: NodeId,
    /// `true` means `CarryType::Max`. A bool rather than the enum purely so this can be ordered.
    pub max: bool,
}

impl Oriented {
    fn new(node: NodeId, carry: CarryType) -> Self {
        Self {
            node,
            max: carry == CarryType::Max,
        }
    }

    fn carry(self) -> CarryType {
        if self.max {
            CarryType::Max
        } else {
            CarryType::Min
        }
    }

    /// The same unitig read on the other strand: the reverse complement of this state.
    fn rc(self) -> Self {
        Self {
            node: self.node,
            max: !self.max,
        }
    }
}

/// Successors of a state, sorted.
///
/// Forwards, the next node's carry is the edge's **target** carry (`.1`); backwards it is the
/// **source** carry (`.0`). This is the rule `gather_left`/`gather_right` follow, and the single
/// easiest thing here to get backwards.
///
/// Sorted because `forward_neighbors` collects from petgraph's `edges_directed`, whose order is an
/// artefact of edge insertion. It is stable within a run, but nothing in that API promises an order,
/// and a survey whose counts depend on one is not a measurement.
fn succs(g: &DbgGraph, s: Oriented) -> Vec<(Oriented, EdgeType)> {
    let mut v: Vec<(Oriented, EdgeType)> = g
        .out_neighbours_bi(s.node, s.carry())
        .into_iter()
        .map(|(n, t)| (Oriented::new(n, t.get_from_and_to().1), t))
        .collect();
    v.sort_by_key(|(o, _)| *o);
    v
}

/// Predecessors of a state, sorted. See [`succs`] for the carry rule.
fn preds(g: &DbgGraph, s: Oriented) -> Vec<(Oriented, EdgeType)> {
    let mut v: Vec<(Oriented, EdgeType)> = g
        .in_neighbours_bi(s.node, s.carry())
        .into_iter()
        .map(|(n, t)| (Oriented::new(n, t.get_from_and_to().0), t))
        .collect();
    v.sort_by_key(|(o, _)| *o);
    v
}

/// Why an entrance yielded no superbubble.
///
/// Counted, never silently discarded: a survey that quietly drops candidates cannot be read as
/// coverage of anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SbReject {
    /// The frontier reached a state with no successors — the region runs off the end of the graph.
    Tip,
    /// The search came back to the entrance itself.
    Cycle,
    /// The interior would contain both strands of one unitig: an inverted repeat meeting its own
    /// reverse complement, which is not a superbubble. A search written for a directed graph walks
    /// straight into these.
    FoldBack,
    /// More than `MAX_SB_INTERIOR` interior states.
    TooBig,
    /// The frontier could not be resolved down to a single exit.
    NoExit,
}

/// A superbubble: an entrance state, an exit state, and everything between them.
#[derive(Debug, Clone)]
pub struct Superbubble {
    /// The fork.
    pub entrance: Oriented,
    /// The join.
    pub exit: Oriented,
    /// Interior states, excluding entrance and exit. Sorted, so the record is reproducible.
    pub interior: Vec<Oriented>,
    /// Paths from entrance to exit, each the interior states in walk order.
    pub paths: Vec<Vec<Oriented>>,
    /// Whether path enumeration hit `MAX_SB_PATHS`.
    pub paths_capped: bool,
}

impl Superbubble {
    /// Interior size in bases: k-mers plus the `k1 - 1` overhang of the first one.
    pub fn interior_bases(&self, g: &DbgGraph, k1: usize) -> usize {
        let kmers: usize = self
            .interior
            .iter()
            .map(|o| g.node_weight(o.node).map_or(0, |w| w.abs_ind.len()))
            .sum();
        if kmers == 0 {
            0
        } else {
            kmers + k1 - 1
        }
    }
}

/// Find the superbubble entered at `entrance`, if there is one.
///
/// The standard "a state is resolved once every predecessor of it has been visited" formulation, made
/// safe for a bidirected graph. Picking the **smallest** resolvable state rather than an arbitrary one
/// is what makes the interior set — and so every size and nesting figure derived from it —
/// independent of iteration order.
fn find_superbubble(g: &DbgGraph, entrance: Oriented) -> Result<Superbubble, SbReject> {
    let mut visited: BTreeSet<Oriented> = BTreeSet::new();
    visited.insert(entrance);

    // A map rather than a set only so the frontier has a defined smallest element to pick.
    let mut frontier: BTreeMap<Oriented, ()> = BTreeMap::new();
    for (u, _) in succs(g, entrance) {
        frontier.insert(u, ());
    }

    loop {
        if frontier.is_empty() {
            return Err(SbReject::NoExit);
        }

        let resolvable = |f: &BTreeMap<Oriented, ()>, vis: &BTreeSet<Oriented>| -> Option<Oriented> {
            f.keys()
                .find(|u| preds(g, **u).iter().all(|(p, _)| vis.contains(p)))
                .copied()
        };

        // One candidate left, and nothing outside the bubble still points into it: that is the exit.
        if frontier.len() == 1 {
            let u = *frontier.keys().next().unwrap();
            if preds(g, u).iter().all(|(p, _)| visited.contains(p)) {
                if u == entrance {
                    return Err(SbReject::Cycle);
                }
                if u.rc() == entrance || visited.contains(&u.rc()) {
                    return Err(SbReject::FoldBack);
                }
                // The back edge the frontier never looks at. Every state inside was resolved by its
                // *predecessors*, so an edge leading from the exit back to the entrance is invisible
                // to that rule — and it is exactly what makes the region cyclic rather than a
                // superbubble. A cycle *within* the interior needs no separate check: its states can
                // never have all their predecessors visited, so the search stalls into `NoExit`.
                if succs(g, u).iter().any(|(v, _)| *v == entrance) {
                    return Err(SbReject::Cycle);
                }
                let mut interior: Vec<Oriented> =
                    visited.iter().copied().filter(|x| *x != entrance).collect();
                interior.sort();
                let (paths, capped) = enumerate_paths(g, entrance, u, &interior);
                return Ok(Superbubble {
                    entrance,
                    exit: u,
                    interior,
                    paths,
                    paths_capped: capped,
                });
            }
        }

        let Some(u) = resolvable(&frontier, &visited) else {
            // Every remaining candidate still has an unvisited predecessor, so something outside the
            // region reaches into it: not a superbubble.
            return Err(SbReject::NoExit);
        };

        if u == entrance {
            return Err(SbReject::Cycle);
        }
        if u.rc() == entrance || visited.contains(&u.rc()) {
            return Err(SbReject::FoldBack);
        }
        let out = succs(g, u);
        if out.is_empty() {
            return Err(SbReject::Tip);
        }
        if visited.len() > MAX_SB_INTERIOR {
            return Err(SbReject::TooBig);
        }

        frontier.remove(&u);
        visited.insert(u);
        for (v, _) in out {
            if !visited.contains(&v) {
                frontier.insert(v, ());
            }
        }
    }
}

/// Every path from `entrance` to `exit` through `interior`, in walk order, capped.
///
/// Depth-first over the sorted successor lists, so the path list is reproducible. Returns whether the
/// cap was hit, which the caller reports rather than passing off a truncated enumeration as complete.
fn enumerate_paths(
    g: &DbgGraph,
    entrance: Oriented,
    exit: Oriented,
    interior: &[Oriented],
) -> (Vec<Vec<Oriented>>, bool) {
    let inside: BTreeSet<Oriented> = interior.iter().copied().collect();
    let mut out: Vec<Vec<Oriented>> = Vec::new();
    let mut capped = false;
    let mut path: Vec<Oriented> = Vec::new();

    // Explicit stack of (state, index of the next successor to try) so a pathological interior cannot
    // blow the real stack.
    let mut stack: Vec<(Oriented, usize)> = vec![(entrance, 0)];
    while let Some((cur, i)) = stack.pop() {
        let nbrs = succs(g, cur);
        if i >= nbrs.len() {
            path.pop();
            continue;
        }
        stack.push((cur, i + 1));
        let nxt = nbrs[i].0;

        if nxt == exit {
            out.push(path.clone());
            if out.len() >= MAX_SB_PATHS {
                capped = true;
                break;
            }
            continue;
        }
        if !inside.contains(&nxt) || path.contains(&nxt) {
            continue;
        }
        path.push(nxt);
        stack.push((nxt, 0));
    }

    (out, capped)
}

/// Canonical form of a superbubble and its strand twin.
///
/// The twin of `(s, t)` is `(RC(t), RC(s))` — the same region walked the other way. Note this is
/// **not** the unordered pair `{s, t}`: using that was what made an earlier prototype report every
/// superbubble twice.
fn twin_key(s: Oriented, t: Oriented) -> (Oriented, Oriented) {
    let fwd = (s, t);
    let rev = (t.rc(), s.rc());
    if fwd <= rev {
        fwd
    } else {
        rev
    }
}

/// Find every superbubble in the graph, deduplicated across strands.
///
/// Returns the superbubbles and the reject tally. Entrances with fewer than two successors are not
/// candidates at all and are not counted as rejects — that is almost every state, and says nothing.
pub fn find_all(g: &DbgGraph) -> (Vec<Superbubble>, RejectCounts) {
    let mut rejects = RejectCounts::default();
    let mut seen: BTreeSet<(Oriented, Oriented)> = BTreeSet::new();
    let mut out: Vec<Superbubble> = Vec::new();

    let mut entrances: Vec<Oriented> = Vec::new();
    for n in g.node_indices() {
        entrances.push(Oriented { node: n, max: false });
        entrances.push(Oriented { node: n, max: true });
    }
    entrances.sort();

    for s in entrances {
        if succs(g, s).len() < 2 {
            continue;
        }
        rejects.entrances += 1;
        match find_superbubble(g, s) {
            Ok(sb) => {
                if seen.insert(twin_key(sb.entrance, sb.exit)) {
                    out.push(sb);
                } else {
                    rejects.strand_twin += 1;
                }
            }
            Err(r) => rejects.record(r),
        }
    }

    out.sort_by_key(|sb| (sb.entrance, sb.exit));
    (out, rejects)
}

/// Why the candidate entrances did not become superbubbles.
#[derive(Debug, Clone, Default)]
pub struct RejectCounts {
    /// States with two or more successors, i.e. genuine candidates.
    pub entrances: usize,
    /// Already counted from the other strand.
    pub strand_twin: usize,
    /// Ran off a tip.
    pub tip: usize,
    /// Cyclic region.
    pub cycle: usize,
    /// Inverted repeat.
    pub fold_back: usize,
    /// Interior over `MAX_SB_INTERIOR`.
    pub too_big: usize,
    /// The frontier never resolved to one exit.
    pub no_exit: usize,
}

impl RejectCounts {
    fn record(&mut self, r: SbReject) {
        match r {
            SbReject::Tip => self.tip += 1,
            SbReject::Cycle => self.cycle += 1,
            SbReject::FoldBack => self.fold_back += 1,
            SbReject::TooBig => self.too_big += 1,
            SbReject::NoExit => self.no_exit += 1,
        }
    }

    fn accumulate(&mut self, o: &RejectCounts) {
        self.entrances += o.entrances;
        self.strand_twin += o.strand_twin;
        self.tip += o.tip;
        self.cycle += o.cycle;
        self.fold_back += o.fold_back;
        self.too_big += o.too_big;
        self.no_exit += o.no_exit;
    }
}

/// How deep inside other superbubbles each one sits.
///
/// A post-pass rather than part of the search: nesting is a property of the whole collection, and the
/// search deliberately knows nothing beyond its own entrance.
fn nesting_depths(sbs: &[Superbubble]) -> Vec<usize> {
    let sets: Vec<BTreeSet<Oriented>> = sbs
        .iter()
        .map(|sb| sb.interior.iter().copied().collect())
        .collect();
    sbs.iter()
        .enumerate()
        .map(|(i, _)| {
            sets.iter()
                .enumerate()
                .filter(|(j, outer)| {
                    *j != i && sets[i].len() < outer.len() && sets[i].is_subset(outer)
                })
                .count()
        })
        .collect()
}

/// Reconstruct one path of a superbubble as a k-mer walk, with flanking context on both sides.
///
/// `multik::branch_walk` cannot be reused — it takes a two-branch `BubbleParts` — but everything it
/// delegates to can be, and the `BranchWalk` it produces is the same object.
///
/// **`n_left` is the index of the last k-mer *before* the divergent middle**, which is what
/// `BranchWalk::window` actually consumes. For a simple bubble the entrance holds exactly one k-mer,
/// so that equals the flank length, which is what the original code passed; here the entrance is a
/// whole unitig. Getting this wrong does not error — it silently windows over shared flank, which
/// discriminates nothing and makes the whole test a no-op.
fn superbubble_walk(
    g: &DbgGraph,
    sb: &Superbubble,
    path: &[Oriented],
    flank_budget: usize,
) -> Option<BranchWalk> {
    // Context on both sides. As with a simple bubble, an ambiguous flank is no context at all.
    let ins = preds(g, sb.entrance);
    let outs = succs(g, sb.exit);
    if ins.len() != 1 || outs.len() != 1 {
        return None;
    }
    let (left, lt, _) = gather_left(g, (ins[0].0.node, ins[0].1), flank_budget);
    let (right, rt, _) = gather_right(g, (outs[0].0.node, outs[0].1), flank_budget);

    let mut hashes = left;
    hashes.extend(oriented(g, sb.entrance.node, sb.entrance.carry()));
    if hashes.is_empty() {
        return None;
    }
    let n_left = hashes.len() - 1;

    let mid: Vec<u64> = path
        .iter()
        .flat_map(|o| oriented(g, o.node, o.carry()))
        .collect();
    let n_mid = mid.len();

    hashes.extend(mid);
    hashes.extend(oriented(g, sb.exit.node, sb.exit.carry()));
    hashes.extend(right);

    Some(BranchWalk {
        hashes,
        n_left,
        n_mid,
        truncated: lt || rt,
    })
}

/// What the evidence says about a superbubble. Generalises `multik::Verdict` from two paths to N.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SbVerdict {
    /// Exactly one path corroborated at k2 and every other truly absent: a genuine complex error
    /// bubble, collapsible on *evidence* rather than coverage.
    ResolvedError(usize),
    /// Two or more corroborated, and every corroborated one is a single unforked k2 unitig: k2 spans
    /// whatever collapsed at k1, so the locus can be split rather than popped. **The headline.**
    ResolvableRepeat,
    /// Two or more corroborated, but k2 forks here too — the repeat is longer than k2 can span.
    ProtectedRepeat,
    /// Not enough corroborated flank to build a trustworthy context.
    InconclusiveContext,
    /// No path corroborated, all absent. Weak evidence: k2 coverage is thinner than k1.
    InconclusiveAbsent,
    /// Partial support. Do not guess.
    InconclusivePartial,
    /// `spell_path` rejected a walk. **Our bug, not the data's** — must be zero.
    InconclusiveSpell,
}

/// Ask the evidence graph about one superbubble.
///
/// The rule is `multik::judge_bubble`'s, generalised: a path is only ever removed on **positive**
/// support for the others, never merely because it is itself absent. k2 coverage is thinner than k1,
/// so absence there is weak evidence, and acting on it would delete true sequence wherever coverage
/// is thin.
#[allow(clippy::too_many_arguments)]
pub fn judge_superbubble<IntT>(
    ev: &EvidenceGraph,
    g: &DbgGraph,
    sb: &Superbubble,
    dict: &HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    k1: usize,
    flank_budget: usize,
    min_evidence: usize,
    max_nodes: usize,
) -> (SbVerdict, bool)
where
    IntT: for<'a> UInt<'a>,
{
    let mut truncated = false;
    let mut evs: Vec<BranchEvidence> = Vec::with_capacity(sb.paths.len());

    if sb.paths.len() < 2 {
        return (SbVerdict::InconclusiveContext, truncated);
    }

    for path in &sb.paths {
        let Some(w) = superbubble_walk(g, sb, path, flank_budget) else {
            return (SbVerdict::InconclusiveContext, truncated);
        };
        truncated |= w.truncated;

        let seq = match spell_branch(&w, dict, k1) {
            Ok(s) => s,
            Err(_) => return (SbVerdict::InconclusiveSpell, truncated),
        };

        let window = w.window(k1, ev.k);
        let e = ev.evidence_for(&seq, window);
        if e.flank_fraction < FLANK_SUPPORT || e.n_window < min_evidence {
            return (SbVerdict::InconclusiveContext, truncated);
        }
        evs.push(e);
    }

    let sup: Vec<bool> = evs
        .iter()
        .map(|e| e.is_supported(min_evidence) && e.window_runs.len() <= max_nodes)
        .collect();
    let abs: Vec<bool> = evs.iter().map(|e| e.is_absent()).collect();

    let n_sup = sup.iter().filter(|x| **x).count();
    let n_abs = abs.iter().filter(|x| **x).count();

    let verdict = if n_sup == 1 && n_abs == sup.len() - 1 {
        SbVerdict::ResolvedError(sup.iter().position(|x| *x).unwrap())
    } else if n_sup >= 2 {
        let all_unforked = sup
            .iter()
            .zip(evs.iter())
            .filter(|(s, _)| **s)
            .all(|(_, e)| e.is_unforked_at_k2());
        if all_unforked {
            SbVerdict::ResolvableRepeat
        } else {
            SbVerdict::ProtectedRepeat
        }
    } else if n_sup == 0 && n_abs == sup.len() {
        SbVerdict::InconclusiveAbsent
    } else {
        SbVerdict::InconclusivePartial
    };

    (verdict, truncated)
}

/// Are the paths a **parallel bundle** — pairwise node-disjoint, and together covering the interior?
///
/// This is the precondition of the whole surgery, which is "move path *i* onto clone set *i*". That is
/// only well defined if the paths partition the interior. Three things are rejected:
///
/// - a state on more than one path: the surgery could not decide which locus it belongs to, and moving
///   it would need it cloned as well — a different and much rarer problem;
/// - an interior state on no path at all: a side branch the enumeration did not reach, which moving the
///   paths would strand;
/// - an empty path, i.e. an edge straight from entrance to exit: an indel bubble, not a collapsed
///   repeat, with nothing to move onto a clone set.
///
/// A superbubble that hit `MAX_SB_PATHS` is excluded outright, before the covering test can be passed
/// by accident on a truncated enumeration.
///
/// Note this also rules out nesting for free. Two paths through an inner bubble both traverse that
/// bubble's own entrance and exit, so an outer superbubble containing one is never disjoint.
fn is_parallel_bundle(sb: &Superbubble) -> bool {
    if sb.paths_capped || sb.paths.len() < 2 {
        return false;
    }
    let mut seen: BTreeSet<Oriented> = BTreeSet::new();
    for path in &sb.paths {
        if path.is_empty() {
            return false;
        }
        for o in path {
            if !seen.insert(*o) {
                return false;
            }
        }
    }
    // Paths are drawn from the interior, so equal cardinality means equal sets.
    seen.len() == sb.interior.len()
}

/// The shared repeat either side of a superbubble, in walk order.
///
/// `left` is `[J, .., entrance]`, where `J` is the node the repeat is *entered* at and forks N ways;
/// `right` is `[exit, .., K]`, where `K` is the node it is *left* at and joins N ways. These are exactly
/// the nodes the genome traverses N times, and so exactly the nodes that must be duplicated for the N
/// loci to become independent.
#[derive(Debug, Clone)]
pub struct SbSharedChain {
    /// `[J, .., entrance]` — the repeat's entry side.
    pub left: Vec<Oriented>,
    /// `[exit, .., K]` — the repeat's exit side.
    pub right: Vec<Oriented>,
}

/// Find the shared repeat around a superbubble, generalising `multik::shared_chain` to N paths.
///
/// The junction must fork **exactly** N ways, not "at least two". The genome runs N loci through this
/// repeat, one per path, so a junction of any other degree is not the structure this surgery models —
/// there would be a locus with no path or a path with no locus, and the pairing below could not be a
/// bijection. This is also what makes the surgery self-limiting in N without a separate cap: a 16-path
/// split would need a 16-way fork in the assembly graph.
fn sb_shared_chain(g: &DbgGraph, sb: &Superbubble) -> Option<SbSharedChain> {
    let n = sb.paths.len();

    // Back from the entrance through unique predecessors, until one that forks N ways.
    let mut left = vec![sb.entrance];
    loop {
        let prev = preds(g, *left.last().unwrap());
        if prev.len() == n {
            break; // the last pushed state is the junction
        }
        if prev.len() != 1 {
            return None;
        }
        left.push(prev[0].0);
        if left.len() > MAX_SHARED_CHAIN {
            return None;
        }
    }
    left.reverse(); // [J, .., entrance]

    // Mirrored forwards from the exit.
    let mut right = vec![sb.exit];
    loop {
        let next = succs(g, *right.last().unwrap());
        if next.len() == n {
            break;
        }
        if next.len() != 1 {
            return None;
        }
        right.push(next[0].0);
        if right.len() > MAX_SHARED_CHAIN {
            return None;
        }
    }

    // A node cannot be duplicated into N independent loci if it appears twice in the chain, nor may a
    // path be part of its own shared context.
    let chain_nodes: Vec<NodeId> = left.iter().chain(right.iter()).map(|o| o.node).collect();
    let chain: BTreeSet<NodeId> = chain_nodes.iter().copied().collect();
    if chain.len() != chain_nodes.len() {
        return None;
    }
    let inside: BTreeSet<NodeId> = sb.interior.iter().map(|o| o.node).collect();
    if inside.iter().any(|n| chain.contains(n)) {
        return None;
    }

    // The nodes the loci arrive from and leave by must lie outside the region being duplicated. If one
    // of them is part of the chain or of a path, the repeat sits on a cycle, and the rewiring below
    // would be moving an edge that is itself part of the structure it is rewiring.
    let outside = |v: &[(Oriented, EdgeType)]| {
        v.iter()
            .all(|(o, _)| !chain.contains(&o.node) && !inside.contains(&o.node))
    };
    if !outside(&preds(g, left[0])) || !outside(&succs(g, *right.last().unwrap())) {
        return None;
    }

    Some(SbSharedChain { left, right })
}

/// Which predecessor and which successor of the collapsed repeat belong with which path.
#[derive(Debug, Clone)]
pub struct SbPairing {
    /// `pred_for[i]` is the predecessor the reads pair with path `i`.
    pub pred_for: Vec<(NodeId, EdgeType)>,
    /// `succ_for[i]` is the successor the reads pair with path `i`.
    pub succ_for: Vec<(NodeId, EdgeType)>,
}

/// Exactly one supported neighbour per path, and a different one for each.
///
/// Deliberately stricter than a maximum matching. A path supported by two neighbours is genuinely
/// ambiguous, and guessing there manufactures a misassembly — which is strictly worse than the repeat
/// collapse it would be replacing.
fn bijection(sup: &[Vec<bool>]) -> Option<Vec<usize>> {
    let mut out = Vec::with_capacity(sup.len());
    for row in sup {
        let mut hit = None;
        for (i, v) in row.iter().enumerate() {
            if *v {
                if hit.is_some() {
                    return None; // two neighbours support this path
                }
                hit = Some(i);
            }
        }
        out.push(hit?);
    }
    let uniq: BTreeSet<usize> = out.iter().copied().collect();
    (uniq.len() == out.len()).then_some(out)
}

/// Work out which predecessor and successor of the collapsed repeat go with which path.
///
/// `multik::pair_ends`' test, widened from two branches of one node to N paths of any length. It falls
/// straight out of what `ResolvableRepeat` already means: for a resolvable superbubble each path's whole
/// walk is a **single** k2 unitig, so extending it out through a *specific* neighbour and asking "is it
/// still a single unitig, with every k2-mer present?" is exactly the pairing test. Only the true
/// neighbour keeps the walk unbroken at k2; a wrong one spells a sequence no read ever contained.
///
/// At most `MAX_SB_PATHS` paths, so the N x N matrix is at most 256 spell-and-lookup pairs per locus.
#[allow(clippy::too_many_arguments)]
fn sb_pair_ends<IntT>(
    ev: &EvidenceGraph,
    g: &DbgGraph,
    sb: &Superbubble,
    chain: &SbSharedChain,
    dict: &HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    k1: usize,
    flank_budget: usize,
) -> Option<SbPairing>
where
    IntT: for<'a> UInt<'a>,
{
    let n = sb.paths.len();
    let pred_list: Vec<(NodeId, EdgeType)> = preds(g, chain.left[0])
        .into_iter()
        .map(|(o, t)| (o.node, t))
        .collect();
    let succ_list: Vec<(NodeId, EdgeType)> = succs(g, *chain.right.last().unwrap())
        .into_iter()
        .map(|(o, t)| (o.node, t))
        .collect();
    if pred_list.len() != n || succ_list.len() != n {
        return None;
    }

    // Parallel edges to one node would let two paths be "paired" with the same neighbour under two
    // indices, and the cut below would then take both edges out at once. And a node that is both a
    // predecessor and a successor means the repeat closes a loop, which is not an N-locus collapse.
    let ends: BTreeSet<NodeId> = pred_list
        .iter()
        .chain(succ_list.iter())
        .map(|(x, _)| *x)
        .collect();
    if ends.len() != 2 * n {
        return None;
    }

    // The shared chain's own hashes, in walk order. `oriented` handles the strand; we owe `spell_path`
    // only the order.
    let left_shared: Vec<u64> = chain
        .left
        .iter()
        .flat_map(|o| oriented(g, o.node, o.carry()))
        .collect();
    let right_shared: Vec<u64> = chain
        .right
        .iter()
        .flat_map(|o| oriented(g, o.node, o.carry()))
        .collect();

    // Is `[neighbour .. shared .. path .. shared']` a single, fully-present k2 unitig?
    let whole = |hashes: &[u64]| -> bool {
        let Ok(seq) = spell_path(hashes, dict, k1) else {
            return false;
        };
        let e = ev.evidence_for(&seq, (0, 0));
        e.flank_fraction >= 1.0 && e.runs.len() == 1
    };

    let mut pred_sup: Vec<Vec<bool>> = Vec::with_capacity(n);
    let mut succ_sup: Vec<Vec<bool>> = Vec::with_capacity(n);

    for path in &sb.paths {
        let mid: Vec<u64> = path
            .iter()
            .flat_map(|o| oriented(g, o.node, o.carry()))
            .collect();

        let mut prow = Vec::with_capacity(n);
        for p in &pred_list {
            let (mut h, _, _) = gather_left(g, *p, flank_budget);
            h.extend(left_shared.iter().copied());
            h.extend(mid.iter().copied());
            h.extend(right_shared.iter().copied());
            prow.push(whole(&h));
        }
        pred_sup.push(prow);

        let mut srow = Vec::with_capacity(n);
        for s in &succ_list {
            let (head, _, _) = gather_right(g, *s, flank_budget);
            let mut h = left_shared.clone();
            h.extend(mid.iter().copied());
            h.extend(right_shared.iter().copied());
            h.extend(head);
            srow.push(whole(&h));
        }
        succ_sup.push(srow);
    }

    let pi = bijection(&pred_sup)?;
    let si = bijection(&succ_sup)?;

    Some(SbPairing {
        pred_for: pi.iter().map(|&i| pred_list[i]).collect(),
        succ_for: si.iter().map(|&i| succ_list[i]).collect(),
    })
}

/// Duplicate the shared repeat so each of the N paths gets its own copy of it, and rewire.
///
/// `multik::split_repeat` with the two hard-wired assumptions lifted: one clone set becomes `N - 1`, and
/// a branch that was one node becomes a path of any length. Before, the genome's N loci are forced
/// through one shared chain, so no contig can walk through and the coverage heuristic deletes all but
/// one of them:
///
/// ```text
///   P0 -.                                    .- Q0
///   P1 --->- [J .. entrance] -+- A -+- [exit .. K] -<--- Q1
///   P2 -'                     +- B -+                `- Q2
///                             `- C -'
/// ```
///
/// After, they are independent, each carrying one real copy. `shrink` then folds each into a single
/// unitig, so contigs run **straight through** the repeat.
///
/// Path 0 keeps the originals; every other path moves onto its own clone set. Coverage is divided by N
/// on the copies *and* on the originals: each now carries one locus' worth of reads rather than N.
///
/// Every edge type is captured **before** anything moves, every clone is an exact copy — same `abs_ind`,
/// same `innerdir` — and every clone-to-clone edge reuses the type of the original it mirrors. No
/// orientation is ever recomputed. That is what makes `split_repeat` safe and it is the property
/// preserved here: the bidirected bookkeeping is the easiest thing in this module to get subtly wrong,
/// so the surgery is arranged so it never has to do any.
fn sb_split_repeat(
    g: &mut DbgGraph,
    sb: &Superbubble,
    chain: &SbSharedChain,
    pairing: &SbPairing,
) -> SplitOutcome {
    let n = sb.paths.len();
    let junction = chain.left[0].node;
    let tail = chain.right.last().unwrap().node;

    let edge_between = |g: &DbgGraph, a: Oriented, b: NodeId| -> Option<EdgeType> {
        g.out_neighbours_bi(a.node, a.carry())
            .into_iter()
            .find(|(x, _)| *x == b)
            .map(|(_, t)| t)
    };

    // --- capture every edge type we must reproduce, before anything moves -------------------------
    let mut left_edges = Vec::with_capacity(chain.left.len().saturating_sub(1));
    for w in chain.left.windows(2) {
        match edge_between(g, w[0], w[1].node) {
            Some(t) => left_edges.push(t),
            None => return SplitOutcome::SurgeryFailedLeft,
        }
    }
    let mut right_edges = Vec::with_capacity(chain.right.len().saturating_sub(1));
    for w in chain.right.windows(2) {
        match edge_between(g, w[0], w[1].node) {
            Some(t) => right_edges.push(t),
            None => return SplitOutcome::SurgeryFailedRight,
        }
    }
    // Entry and exit edge of every path. Computed for all N, including the one that stays put, because
    // a missing edge there means the structure is not what was judged and nothing should move.
    let mut path_edges = Vec::with_capacity(n);
    for path in &sb.paths {
        let (Some(first), Some(last)) = (path.first(), path.last()) else {
            return SplitOutcome::SurgeryFailedMid;
        };
        let (Some(into), Some(out)) = (
            edge_between(g, sb.entrance, first.node),
            edge_between(g, *last, sb.exit.node),
        ) else {
            return SplitOutcome::SurgeryFailedMid;
        };
        path_edges.push((into, out));
    }

    // --- clone the shared nodes, one set per path that moves --------------------------------------
    //
    // Read the originals' counts on every pass, before any of them is rewritten below: each clone must
    // get its share of the *whole* repeat's coverage, not of a already-divided one.
    let share = |w: &mut NodeStruct| w.counts = w.counts.div_ceil(n as u32);
    let mut left_sets: Vec<Vec<NodeId>> = Vec::with_capacity(n - 1);
    let mut right_sets: Vec<Vec<NodeId>> = Vec::with_capacity(n - 1);
    for _ in 1..n {
        let mut lc = Vec::with_capacity(chain.left.len());
        for o in &chain.left {
            let mut w = g.node_weight(o.node).unwrap().clone();
            share(&mut w);
            lc.push(g.add_node(w));
        }
        let mut rc = Vec::with_capacity(chain.right.len());
        for o in &chain.right {
            let mut w = g.node_weight(o.node).unwrap().clone();
            share(&mut w);
            rc.push(g.add_node(w));
        }
        left_sets.push(lc);
        right_sets.push(rc);
    }
    // The originals now carry one locus, not N.
    for o in chain.left.iter().chain(chain.right.iter()) {
        share(g.node_weight_mut(o.node).unwrap());
    }

    // --- wire each clone chain, mirroring the originals exactly -----------------------------------
    for set in &left_sets {
        for (i, t) in left_edges.iter().enumerate() {
            g.add_bi_edge(set[i], set[i + 1], *t);
        }
    }
    for set in &right_sets {
        for (i, t) in right_edges.iter().enumerate() {
            g.add_bi_edge(set[i], set[i + 1], *t);
        }
    }

    // --- move paths 1..N onto their clone sets ----------------------------------------------------
    let cut = |g: &mut DbgGraph, a: NodeId, b: NodeId| {
        for e in g.edges_between(a, b) {
            g.remove_edge(e);
        }
        for e in g.edges_between(b, a) {
            g.remove_edge(e);
        }
    };

    for i in 1..n {
        let lc = &left_sets[i - 1];
        let rc = &right_sets[i - 1];
        let path = &sb.paths[i];
        let first = path[0].node;
        let last = path[path.len() - 1].node;
        let (into, out) = path_edges[i];

        cut(g, sb.entrance.node, first);
        g.add_bi_edge(*lc.last().unwrap(), first, into);

        cut(g, last, sb.exit.node);
        g.add_bi_edge(last, rc[0], out);

        cut(g, pairing.pred_for[i].0, junction);
        g.add_bi_edge(pairing.pred_for[i].0, lc[0], pairing.pred_for[i].1);

        cut(g, tail, pairing.succ_for[i].0);
        g.add_bi_edge(*rc.last().unwrap(), pairing.succ_for[i].0, pairing.succ_for[i].1);
    }

    SplitOutcome::Applied
}

/// Are the paired neighbours still the ones hanging off the repeat's two ends?
///
/// The chain having re-derived identically already fixes the junction's degree at N; this is what says
/// the N neighbours are still the same N. They were distinct when the pairing was made, so equal size
/// plus containment is equality.
fn pairing_still_attached(g: &DbgGraph, chain: &SbSharedChain, pairing: &SbPairing) -> bool {
    let here: BTreeSet<NodeId> = preds(g, chain.left[0]).iter().map(|(o, _)| o.node).collect();
    let there: BTreeSet<NodeId> = succs(g, *chain.right.last().unwrap())
        .iter()
        .map(|(o, _)| o.node)
        .collect();
    pairing.pred_for.iter().all(|(n, _)| here.contains(n))
        && pairing.succ_for.iter().all(|(n, _)| there.contains(n))
}

/// One superbubble that the evidence says can be split, together with the plan for splitting it.
type SplitPlan = (Superbubble, SbSharedChain, SbPairing);

/// Find every superbubble, judge it, and work out which ones could be split — **without touching `g`**.
///
/// The read-only guarantee is the `&DbgGraph`: the surgery's whole input is computed here, so it can be
/// counted and reported before the surgery is enabled anywhere. That the same function feeds both the
/// survey and the correction is the point — the attrition the survey reports is the attrition the
/// correction will actually see, not an estimate of it.
#[allow(clippy::too_many_arguments)]
fn judge_and_assess<IntT>(
    ev: &EvidenceGraph,
    g: &DbgGraph,
    dict: &HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    k1: usize,
    flank_budget: usize,
    min_evidence: usize,
    max_nodes: usize,
    stats: &mut SuperbubbleStats,
) -> Vec<SplitPlan>
where
    IntT: for<'a> UInt<'a>,
{
    let (sbs, rejects) = find_all(g);
    stats.rejects.accumulate(&rejects);

    let depths = nesting_depths(&sbs);
    let mut plans: Vec<SplitPlan> = Vec::new();

    for (i, sb) in sbs.iter().enumerate() {
        // A two-path superbubble whose paths are one unitig each, and whose entrance the existing
        // detector also recognises, is a simple bubble we already act on. Counting those in the
        // headline would overstate what this adds — and splitting them here as well would clone the
        // shared chain twice over the same locus.
        let simple = sb.paths.len() == 2
            && sb.paths.iter().all(|p| p.len() == 1)
            && crate::algorithms::corrector::bubble_parts(g, sb.entrance.node).is_some();

        let (v, truncated) = judge_superbubble::<IntT>(
            ev,
            g,
            sb,
            dict,
            k1,
            flank_budget,
            min_evidence,
            max_nodes,
        );
        stats.record(v, truncated, simple, sb, g, k1, depths[i]);

        if v != SbVerdict::ResolvableRepeat || simple {
            continue;
        }
        if !is_parallel_bundle(sb) {
            stats.not_a_bundle += 1;
        } else if let Some(chain) = sb_shared_chain(g, sb) {
            match sb_pair_ends::<IntT>(ev, g, sb, &chain, dict, k1, flank_budget) {
                Some(pr) => {
                    stats.pairing_ok += 1;
                    plans.push((sb.clone(), chain, pr));
                }
                None => stats.pairing_ambiguous += 1,
            }
        } else {
            stats.chain_absent += 1;
        }
    }

    plans
}

/// Survey every superbubble in the graph. **Read-only**: `g` is not modified.
#[allow(clippy::too_many_arguments)]
pub fn survey_superbubbles<IntT>(
    ev: &EvidenceGraph,
    g: &DbgGraph,
    dict: &HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    k1: usize,
    flank_budget: usize,
    min_evidence: usize,
    max_nodes: usize,
    stats: &mut SuperbubbleStats,
) where
    IntT: for<'a> UInt<'a>,
{
    // The plans are built and thrown away: what the survey reports is how many there were.
    let _ = judge_and_assess::<IntT>(
        ev,
        g,
        dict,
        k1,
        flank_budget,
        min_evidence,
        max_nodes,
        stats,
    );
}

/// Judge every superbubble and split the complex ones the evidence resolves.
///
/// Splits are collected first and applied afterwards, exactly as the simple-bubble pass does and for the
/// same reason: each one rewires the neighbourhood the next was judged against. Each is therefore
/// re-derived immediately before it is applied, and abandoned if the structure moved.
///
/// This used to return the entrances it could not split, so the caller could veto the coverage
/// heuristic on them. That is no longer needed: the popper leaves a comparable-coverage locus alone, so
/// a superbubble whose split was skipped survives on its own merits rather than on an exemption list.
#[allow(clippy::too_many_arguments)]
pub fn correct_superbubbles_with_evidence<IntT>(
    ev: &EvidenceGraph,
    g: &mut DbgGraph,
    dict: &HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    k1: usize,
    flank_budget: usize,
    min_evidence: usize,
    max_nodes: usize,
    stats: &mut SuperbubbleStats,
) where
    IntT: for<'a> UInt<'a>,
{
    let plans = judge_and_assess::<IntT>(
        ev,
        &*g,
        dict,
        k1,
        flank_budget,
        min_evidence,
        max_nodes,
        stats,
    );

    for (sb, chain, pairing) in &plans {
        let outcome = if !g.contains_node(sb.entrance.node) {
            SplitOutcome::StaleNoBubble
        } else {
            match find_superbubble(g, sb.entrance) {
                Err(_) => SplitOutcome::StaleNoBubble,
                Ok(fresh) if fresh.exit != sb.exit || fresh.paths != sb.paths => {
                    SplitOutcome::StaleMidsChanged
                }
                // The shared chain and the pairing both live *outside* the superbubble, so
                // re-deriving the superbubble says nothing about them. An earlier split in this pass
                // can leave every edge the surgery reads still present while changing a *degree* — at
                // which point `cut` would remove nothing and the `add_bi_edge` that follows would
                // invent a link. Re-deriving the chain catches that, and it is the cheap half of the
                // work: no spelling, no evidence lookups.
                Ok(_) => match sb_shared_chain(g, sb) {
                    Some(fresh) if fresh.left == chain.left && fresh.right == chain.right => {
                        if pairing_still_attached(g, chain, pairing) {
                            sb_split_repeat(g, sb, chain, pairing)
                        } else {
                            SplitOutcome::StaleMidsChanged
                        }
                    }
                    _ => SplitOutcome::StaleMidsChanged,
                },
            }
        };
        stats.record_split(outcome);

        if outcome != SplitOutcome::Applied {
            // One line per locus: there are few enough of these to be worth reading individually
            // rather than inferring from totals, and they are the only window on whether the
            // collect-then-apply discipline is costing anything.
            log::info!(
                "    superbubble split skipped ({outcome:?}): entrance={:?} exit={:?} \
                 paths={} chain={}L+{}R",
                sb.entrance.node,
                sb.exit.node,
                sb.paths.len(),
                chain.left.len(),
                chain.right.len(),
            );
        }
    }
}

/// Coverage of a path: the weakest node on it.
///
/// Minimum, not mean — a path is only as real as its least-supported unitig, and a mean lets one
/// well-covered unitig carry a noise unitig through the test below.
fn path_coverage(g: &DbgGraph, path: &[Oriented]) -> u32 {
    path.iter()
        .map(|o| g.node_weight(o.node).map_or(0, |w| w.counts))
        .min()
        .unwrap_or(0)
}

/// Which path, if any, the coverage clearly favours: one winner, every other under `pop_ratio` of it.
///
/// Deliberately the same shape as `corrector::choose_branch_by_counts`, and strictly harder to satisfy
/// as N grows — which is the intent. The more branches a locus has, the less a coverage argument for
/// deleting all but one of them is worth.
fn coverage_winner(g: &DbgGraph, sb: &Superbubble, pop_ratio: f32) -> Option<usize> {
    let covs: Vec<u32> = sb.paths.iter().map(|p| path_coverage(g, p)).collect();
    if covs.len() < 2 {
        return None;
    }
    let mut best = 0usize;
    for (i, c) in covs.iter().enumerate() {
        if *c > covs[best] {
            best = i; // strict, so a tie for the top keeps the lowest index and is deterministic
        }
    }
    // Strict `<` on the losers, so paths of equal coverage — a collapsed repeat — never collapse,
    // however the ratio is set. A tie for the top therefore always falls through to `None`.
    let hi = covs[best] as f32;
    covs.iter()
        .enumerate()
        .all(|(i, c)| i == best || (*c as f32) < pop_ratio * hi)
        .then_some(best)
}

/// Collapse a superbubble onto one path by deleting the others.
///
/// Deliberately **not** modelled on `corrector::apply_bubble_collapse`, which fuses start, winner and
/// end into a single node and asserts that start and end hold exactly one k-mer each. Neither holds
/// here: a superbubble's entrance and exit are whole unitigs. Deleting the losing paths and letting the
/// next `shrink` fuse `entrance -> winner -> exit` reaches the same state with none of the bidirected
/// bookkeeping, and `remove_node` drops the incident edges for us.
///
/// Sound **only on a parallel bundle** — paths pairwise disjoint and together covering the interior.
/// Otherwise deleting them would either remove a node another surviving path needs, or strand an
/// interior node that no path owns.
fn collapse_onto(g: &mut DbgGraph, sb: &Superbubble, winner: usize) {
    for (i, path) in sb.paths.iter().enumerate() {
        if i == winner {
            continue;
        }
        for o in path {
            g.remove_node(o.node);
        }
    }
}

/// Counts for the post-ladder, coverage-only superbubble collapse.
#[derive(Debug, Clone, Default)]
pub struct SbCollapseStats {
    /// Superbubbles examined, summed over every pass.
    pub seen: usize,
    /// …that are simple bubbles, so the simple popper owns them.
    pub simple_equivalent: usize,
    /// …whose paths are not a parallel bundle, so deletion could strand or steal a node.
    pub not_a_bundle: usize,
    /// …where no single path dominates. **Expected to be the overwhelming majority.**
    pub no_clear_winner: usize,
    /// Collapses performed.
    pub applied: usize,
    /// Planned, but the neighbourhood moved before it could be applied.
    pub stale: usize,
}

impl SbCollapseStats {
    /// Report at `info`, and check that every superbubble seen landed in exactly one bucket.
    pub fn report(&self) {
        assert_eq!(
            self.simple_equivalent + self.not_a_bundle + self.no_clear_winner + self.applied + self.stale,
            self.seen,
            "superbubble collapse accounting does not balance for {} superbubbles seen",
            self.seen
        );
        if self.seen == 0 {
            return;
        }
        log::info!("Superbubble collapse (coverage only), over all passes:");
        log::info!("  superbubbles examined              {:5}", self.seen);
        log::info!("    simple bubble (other pass owns)  {:5}", self.simple_equivalent);
        log::info!("    not a parallel bundle            {:5}", self.not_a_bundle);
        log::info!("    no path dominates (left alone)   {:5}", self.no_clear_winner);
        log::info!("    stale by the time it was applied {:5}", self.stale);
        log::info!("  COLLAPSED                          {:5}", self.applied);
    }
}

/// Collapse superbubbles where coverage clearly favours one path. **Deletes nodes.**
///
/// The post-ladder counterpart to `corrector::pop_bubbles_by_coverage`, on the same rule and the same
/// threshold: this runs on what the evidence k could not resolve, and declines unless one path
/// dominates every other. `no_clear_winner` is expected to dominate the tally — a collapsed repeat's
/// paths have comparable coverage by construction, and those must survive untouched.
///
/// Collect-then-apply with re-derivation, exactly as `correct_superbubbles_with_evidence` does and for
/// the same reason: each collapse reshapes the neighbourhood the next one was judged against.
pub fn collapse_superbubbles_by_coverage(
    g: &mut DbgGraph,
    pop_ratio: f32,
    stats: &mut SbCollapseStats,
) -> bool {
    let (sbs, _) = find_all(g);
    let mut plan: Vec<(Superbubble, usize)> = Vec::new();

    for sb in &sbs {
        stats.seen += 1;
        let simple = sb.paths.len() == 2
            && sb.paths.iter().all(|p| p.len() == 1)
            && crate::algorithms::corrector::bubble_parts(g, sb.entrance.node).is_some();
        if simple {
            // The simple popper's business. Acting here too would collapse the same locus twice.
            stats.simple_equivalent += 1;
        } else if !is_parallel_bundle(sb) {
            stats.not_a_bundle += 1;
        } else if let Some(w) = coverage_winner(g, sb, pop_ratio) {
            plan.push((sb.clone(), w));
        } else {
            stats.no_clear_winner += 1;
        }
    }

    let mut changed = false;
    for (sb, w) in &plan {
        let fresh = if g.contains_node(sb.entrance.node) {
            find_superbubble(g, sb.entrance).ok()
        } else {
            None
        };
        match fresh {
            Some(f) if f.exit == sb.exit && f.paths == sb.paths => {
                collapse_onto(g, sb, *w);
                stats.applied += 1;
                changed = true;
            }
            _ => stats.stale += 1,
        }
    }
    changed
}

/// Counts of what the oracle said about superbubbles.
///
/// Mirrors `multik::MultiKStats` in shape. The `complex_*` counters are the ones the decision gate is
/// read off: they exclude the superbubbles that are simple bubbles the assembler already handles.
#[derive(Debug, Clone, Default)]
pub struct SuperbubbleStats {
    /// Correction rounds run. `0` for a pure survey, which does exactly one read-only pass.
    pub rounds: usize,
    /// Superbubbles found, after strand dedup.
    pub found: usize,
    /// …of which are simple bubbles the existing detector already recognises.
    pub simple_equivalent: usize,
    /// …of which are not: the population this module exists to measure.
    pub complex: usize,

    /// Why the candidate entrances that yielded nothing yielded nothing.
    pub rejects: RejectCounts,

    // Verdict buckets, over every superbubble.
    /// One path corroborated, the rest absent.
    pub resolved_error: usize,
    /// Two or more corroborated, none forking at k2.
    pub resolvable_repeat: usize,
    /// Two or more corroborated, but k2 forks too.
    pub protected_repeat: usize,
    /// Flanks not corroborated, or too little context.
    pub inconclusive_context: usize,
    /// No path corroborated.
    pub inconclusive_absent: usize,
    /// Partial support somewhere.
    pub inconclusive_partial: usize,
    /// **Must be zero.** A path taken from the graph is a walk by construction.
    pub inconclusive_spell: usize,

    // The same two verdicts restricted to complex superbubbles — the headline.
    /// `ResolvedError` restricted to complex superbubbles.
    pub complex_resolved_error: usize,
    /// `ResolvableRepeat` restricted to complex superbubbles.
    pub complex_resolvable_repeat: usize,

    /// Path-count histogram: exactly 2, 3, 4, and 5 or more.
    /// Exactly two paths.
    pub paths_2: usize,
    /// Exactly three.
    pub paths_3: usize,
    /// Exactly four.
    pub paths_4: usize,
    /// Five or more.
    pub paths_5plus: usize,

    /// Interior size, summed over all superbubbles, in unitigs and in bases.
    /// Interior unitigs, summed over all superbubbles.
    pub interior_unitigs: usize,
    /// Interior bases, summed over all superbubbles.
    pub interior_bases: usize,
    /// Superbubbles lying strictly inside another.
    pub nested: usize,
    /// Deepest nesting seen.
    pub max_depth: usize,

    /// Path enumeration hit `MAX_SB_PATHS`.
    pub paths_capped: usize,
    /// A flank hit the context budget.
    pub flanks_truncated: usize,

    // ── attrition from "resolvable" to "actually splittable" ─────────────────
    //
    // These are the surgery's input, counted whether or not the surgery runs, because the number that
    // says what splitting superbubbles is worth is `pairing_ok` — not `complex_resolvable_repeat`. On
    // simple bubbles the same two steps already lose about half.
    /// Complex and resolvable, but the paths are not a parallel bundle: they overlap, leave an interior
    /// state uncovered, run straight from entrance to exit, or were capped.
    pub not_a_bundle: usize,
    /// A bundle, but there is no clean shared repeat around it forking exactly as many ways as there
    /// are paths.
    pub chain_absent: usize,
    /// Chain found, and the reads pair every path with a distinct neighbour on both sides. **These are
    /// the ones the surgery can actually split.**
    pub pairing_ok: usize,
    /// Chain found, but the pairing was ambiguous. Never split on a guess.
    pub pairing_ambiguous: usize,

    // ── what became of the planned splits ────────────────────────────────────
    /// Splits actually performed.
    pub split_applied: usize,
    /// No superbubble at this entrance any more — usually an earlier split in the same pass reshaped it.
    pub stale_no_bubble: usize,
    /// Still a superbubble, but with a different exit or a different path set than was judged.
    pub stale_mids_changed: usize,
    /// An expected edge in the left shared chain was absent.
    pub surgery_failed_left: usize,
    /// An expected edge in the right shared chain was absent.
    pub surgery_failed_right: usize,
    /// A path's edge to the entrance or the exit was absent.
    pub surgery_failed_mid: usize,
}

impl SuperbubbleStats {
    /// Record one superbubble and its verdict.
    #[allow(clippy::too_many_arguments)]
    fn record(
        &mut self,
        v: SbVerdict,
        truncated: bool,
        simple: bool,
        sb: &Superbubble,
        g: &DbgGraph,
        k1: usize,
        depth: usize,
    ) {
        self.found += 1;
        if simple {
            self.simple_equivalent += 1;
        } else {
            self.complex += 1;
        }
        if truncated {
            self.flanks_truncated += 1;
        }
        if sb.paths_capped {
            self.paths_capped += 1;
        }

        match sb.paths.len() {
            // 0 and 1 cannot occur — `judge_superbubble` needs two paths and a fork has at least
            // two successors — but they are folded in here rather than left to panic a histogram.
            0..=2 => self.paths_2 += 1,
            3 => self.paths_3 += 1,
            4 => self.paths_4 += 1,
            _ => self.paths_5plus += 1,
        }

        self.interior_unitigs += sb.interior.len();
        self.interior_bases += sb.interior_bases(g, k1);
        if depth > 0 {
            self.nested += 1;
        }
        self.max_depth = self.max_depth.max(depth);

        match v {
            SbVerdict::ResolvedError(_) => {
                self.resolved_error += 1;
                if !simple {
                    self.complex_resolved_error += 1;
                }
            }
            SbVerdict::ResolvableRepeat => {
                self.resolvable_repeat += 1;
                if !simple {
                    self.complex_resolvable_repeat += 1;
                }
            }
            SbVerdict::ProtectedRepeat => self.protected_repeat += 1,
            SbVerdict::InconclusiveContext => self.inconclusive_context += 1,
            SbVerdict::InconclusiveAbsent => self.inconclusive_absent += 1,
            SbVerdict::InconclusivePartial => self.inconclusive_partial += 1,
            SbVerdict::InconclusiveSpell => self.inconclusive_spell += 1,
        }
    }

    /// Total planned splits that did not happen, however they failed.
    pub fn split_skipped(&self) -> usize {
        self.stale_no_bubble
            + self.stale_mids_changed
            + self.surgery_failed_left
            + self.surgery_failed_right
            + self.surgery_failed_mid
    }

    /// Record one attempted split against its outcome.
    pub fn record_split(&mut self, o: SplitOutcome) {
        match o {
            SplitOutcome::Applied => self.split_applied += 1,
            SplitOutcome::StaleNoBubble => self.stale_no_bubble += 1,
            SplitOutcome::StaleMidsChanged => self.stale_mids_changed += 1,
            SplitOutcome::SurgeryFailedLeft => self.surgery_failed_left += 1,
            SplitOutcome::SurgeryFailedRight => self.surgery_failed_right += 1,
            SplitOutcome::SurgeryFailedMid => self.surgery_failed_mid += 1,
        }
    }

    /// Fold one round's counts into a running total.
    ///
    /// `max_depth` is a maximum and `rounds` is set by the caller; everything else is a tally and sums.
    pub fn accumulate(&mut self, r: &SuperbubbleStats) {
        self.found += r.found;
        self.simple_equivalent += r.simple_equivalent;
        self.complex += r.complex;
        self.rejects.accumulate(&r.rejects);
        self.resolved_error += r.resolved_error;
        self.resolvable_repeat += r.resolvable_repeat;
        self.protected_repeat += r.protected_repeat;
        self.inconclusive_context += r.inconclusive_context;
        self.inconclusive_absent += r.inconclusive_absent;
        self.inconclusive_partial += r.inconclusive_partial;
        self.inconclusive_spell += r.inconclusive_spell;
        self.complex_resolved_error += r.complex_resolved_error;
        self.complex_resolvable_repeat += r.complex_resolvable_repeat;
        self.paths_2 += r.paths_2;
        self.paths_3 += r.paths_3;
        self.paths_4 += r.paths_4;
        self.paths_5plus += r.paths_5plus;
        self.interior_unitigs += r.interior_unitigs;
        self.interior_bases += r.interior_bases;
        self.nested += r.nested;
        self.max_depth = self.max_depth.max(r.max_depth);
        self.paths_capped += r.paths_capped;
        self.flanks_truncated += r.flanks_truncated;
        self.not_a_bundle += r.not_a_bundle;
        self.chain_absent += r.chain_absent;
        self.pairing_ok += r.pairing_ok;
        self.pairing_ambiguous += r.pairing_ambiguous;
        self.split_applied += r.split_applied;
        self.stale_no_bubble += r.stale_no_bubble;
        self.stale_mids_changed += r.stale_mids_changed;
        self.surgery_failed_left += r.surgery_failed_left;
        self.surgery_failed_right += r.surgery_failed_right;
        self.surgery_failed_mid += r.surgery_failed_mid;
    }

    /// The seven verdict buckets must account for every superbubble found.
    pub fn check_accounting(&self) {
        let sum = self.resolved_error
            + self.resolvable_repeat
            + self.protected_repeat
            + self.inconclusive_context
            + self.inconclusive_absent
            + self.inconclusive_partial
            + self.inconclusive_spell;
        assert_eq!(
            sum, self.found,
            "superbubble verdict accounting does not balance: {sum} verdicts for {} superbubbles",
            self.found
        );
        assert_eq!(
            self.simple_equivalent + self.complex,
            self.found,
            "superbubble simple/complex split does not balance"
        );
        // Every complex resolvable superbubble goes down exactly one arm of the attrition. If this
        // ever drifts, `pairing_ok` is no longer "of the resolvable, how many are splittable", and the
        // gate the whole phase is read off means nothing.
        let attrition =
            self.not_a_bundle + self.chain_absent + self.pairing_ok + self.pairing_ambiguous;
        assert_eq!(
            attrition, self.complex_resolvable_repeat,
            "superbubble split-attrition accounting does not balance: {attrition} outcomes for {} \
             complex resolvable superbubbles",
            self.complex_resolvable_repeat
        );
    }

    /// One TSV row, so sweeps are scriptable without scraping the log.
    pub fn tsv_header() -> &'static str {
        "k\trounds\tfound\tsimple_equivalent\tcomplex\tresolved_error\tresolvable_repeat\t\
         complex_resolved_error\tcomplex_resolvable_repeat\tprotected_repeat\t\
         inconclusive_context\tinconclusive_absent\tinconclusive_partial\tinconclusive_spell\t\
         paths_2\tpaths_3\tpaths_4\tpaths_5plus\tinterior_unitigs\tinterior_bases\tnested\t\
         max_depth\tpaths_capped\tflanks_truncated\tentrances\tstrand_twin\treject_tip\t\
         reject_cycle\treject_foldback\treject_toobig\treject_noexit\t\
         not_a_bundle\tchain_absent\tpairing_ok\tpairing_ambiguous\tsplit_applied\tsplit_skipped\t\
         stale_no_bubble\tstale_mids_changed\tsurgery_failed_left\tsurgery_failed_right\t\
         surgery_failed_mid"
    }

    /// `label` is the evidence k, or `TOTAL`.
    pub fn tsv_row(&self, label: &str) -> String {
        let r = &self.rejects;
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\
             {}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            label,
            self.rounds,
            self.found,
            self.simple_equivalent,
            self.complex,
            self.resolved_error,
            self.resolvable_repeat,
            self.complex_resolved_error,
            self.complex_resolvable_repeat,
            self.protected_repeat,
            self.inconclusive_context,
            self.inconclusive_absent,
            self.inconclusive_partial,
            self.inconclusive_spell,
            self.paths_2,
            self.paths_3,
            self.paths_4,
            self.paths_5plus,
            self.interior_unitigs,
            self.interior_bases,
            self.nested,
            self.max_depth,
            self.paths_capped,
            self.flanks_truncated,
            r.entrances,
            r.strand_twin,
            r.tip,
            r.cycle,
            r.fold_back,
            r.too_big,
            r.no_exit,
            self.not_a_bundle,
            self.chain_absent,
            self.pairing_ok,
            self.pairing_ambiguous,
            self.split_applied,
            self.split_skipped(),
            self.stale_no_bubble,
            self.stale_mids_changed,
            self.surgery_failed_left,
            self.surgery_failed_right,
            self.surgery_failed_mid,
        )
    }

    /// Report at `info`. This is the number the decision gate is read off, so it is never hidden.
    ///
    /// `rounds == 0` marks a pure survey, which is the only case where "nothing was corrected" can be
    /// claimed; otherwise these are the totals over the correction rounds.
    pub fn report(&self, k1: usize, k2: usize) {
        self.check_accounting();
        let n = self.found;
        let pc = |x: usize| if n == 0 { 0.0 } else { 100.0 * x as f64 / n as f64 };

        if self.rounds == 0 {
            log::info!("Superbubble survey (k1={k1}, k2={k2}) — READ-ONLY, nothing was corrected:");
        } else {
            log::info!(
                "Superbubble correction summary (k1={k1}, k2={k2}), {} rounds:",
                self.rounds
            );
        }
        log::info!("  candidate entrances (>=2 succ)     {:5}", self.rejects.entrances);
        log::info!("    rejected: runs off a tip         {:5}", self.rejects.tip);
        log::info!("    rejected: cycle                  {:5}", self.rejects.cycle);
        log::info!("    rejected: fold-back (inv. repeat){:5}", self.rejects.fold_back);
        log::info!("    rejected: interior over cap      {:5}", self.rejects.too_big);
        log::info!("    rejected: no single exit         {:5}", self.rejects.no_exit);
        log::info!("    same bubble from the other strand{:5}", self.rejects.strand_twin);
        log::info!("  SUPERBUBBLES found                 {n:5}");
        log::info!("    already handled (simple bubble)  {:5}  {:5.1}%", self.simple_equivalent, pc(self.simple_equivalent));
        log::info!("    COMPLEX (new to this survey)     {:5}  {:5.1}%", self.complex, pc(self.complex));
        log::info!("  paths per superbubble: 2={} 3={} 4={} 5+={}", self.paths_2, self.paths_3, self.paths_4, self.paths_5plus);
        log::info!("  interior: {} unitigs, {} bases total; {} nested, max depth {}", self.interior_unitigs, self.interior_bases, self.nested, self.max_depth);
        log::info!("  verdicts over all superbubbles:");
        log::info!("    resolved as error                {:5}  {:5.1}%", self.resolved_error, pc(self.resolved_error));
        log::info!("    RESOLVABLE repeat                {:5}  {:5.1}%", self.resolvable_repeat, pc(self.resolvable_repeat));
        log::info!("    protected repeat                 {:5}  {:5.1}%", self.protected_repeat, pc(self.protected_repeat));
        log::info!("    inconclusive: no context         {:5}  {:5.1}%", self.inconclusive_context, pc(self.inconclusive_context));
        log::info!("    inconclusive: all absent         {:5}  {:5.1}%", self.inconclusive_absent, pc(self.inconclusive_absent));
        log::info!("    inconclusive: partial            {:5}  {:5.1}%", self.inconclusive_partial, pc(self.inconclusive_partial));
        log::info!("    inconclusive: SPELL FAILED       {:5}  {:5.1}%  <- must be 0", self.inconclusive_spell, pc(self.inconclusive_spell));
        log::info!("  THE GATE — on complex superbubbles only:");
        log::info!("    resolved_error                   {:5}", self.complex_resolved_error);
        log::info!("    RESOLVABLE repeat                {:5}", self.complex_resolvable_repeat);
        log::info!("  of those resolvable, what the surgery can take:");
        log::info!("    not a parallel bundle            {:5}", self.not_a_bundle);
        log::info!("    no N-way shared repeat           {:5}", self.chain_absent);
        log::info!("    pairing ambiguous (will not guess){:4}", self.pairing_ambiguous);
        log::info!("    PAIRED (can be split)            {:5}", self.pairing_ok);
        log::info!("  superbubbles SPLIT                 {:5}", self.split_applied);
        if self.split_skipped() > 0 {
            log::info!("    NOT split, by cause              {:5}", self.split_skipped());
            log::info!("      stale: no superbubble any more {:5}", self.stale_no_bubble);
            log::info!("      stale: exit or paths changed   {:5}", self.stale_mids_changed);
            log::info!("      surgery: left chain edge gone  {:5}", self.surgery_failed_left);
            log::info!("      surgery: right chain edge gone {:5}", self.surgery_failed_right);
            log::info!("      surgery: path end edge gone    {:5}", self.surgery_failed_mid);
        }
        log::info!("  caps hit: paths {}, flanks truncated {}", self.paths_capped, self.flanks_truncated);

        if self.inconclusive_spell > 0 {
            log::warn!(
                "{} superbubble paths failed to spell. A path taken from the graph is a walk by \
                 construction, so this is a bug in the orientation bookkeeping — every other number \
                 above is suspect.",
                self.inconclusive_spell
            );
        }
        if self.paths_capped > 0 || self.rejects.too_big > 0 {
            log::warn!(
                "{} superbubbles hit the path cap ({MAX_SB_PATHS}) and {} entrances hit the interior \
                 cap ({MAX_SB_INTERIOR}). Those are bounded, not covered.",
                self.paths_capped,
                self.rejects.too_big
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparrowhawk_graph::NodeStruct;

    /// A one-k-mer node, as `corrector`'s bubble tests use.
    fn make_node() -> NodeStruct {
        NodeStruct {
            counts: 10,
            abs_ind: vec![0u64],
            innerdir: None,
        }
    }

    fn omin(n: NodeId) -> Oriented {
        Oriented { node: n, max: false }
    }

    /// A collapsed N-copy repeat: `n` predecessors join at `J`, run through the shared chain
    /// `J -> L -> entrance`, fork `n` ways, rejoin at `exit`, run through `exit -> R -> K`, and split
    /// back to `n` successors. This is the shape the surgery exists to take apart.
    ///
    /// `path_len` gives the length in unitigs of each branch, so a two-unitig branch — the 75-80% case
    /// the survey measured — is one argument away.
    struct Collapsed {
        g: DbgGraph,
        entrance: NodeId,
        preds_in: Vec<NodeId>,
        paths: Vec<Vec<NodeId>>,
    }

    fn collapsed_repeat(path_lens: &[usize]) -> Collapsed {
        let n = path_lens.len();
        let mut g = DbgGraph::new(3);
        let preds_in: Vec<NodeId> = (0..n).map(|_| g.add_node(make_node())).collect();
        let j = g.add_node(make_node());
        let l = g.add_node(make_node());
        let s = g.add_node(make_node());
        for p in &preds_in {
            g.add_bi_edge(*p, j, EdgeType::MinToMin);
        }
        g.add_bi_edge(j, l, EdgeType::MinToMin);
        g.add_bi_edge(l, s, EdgeType::MinToMin);

        // Branches first, so their ids sort ahead of the exit side and the walk order is the id order.
        let paths: Vec<Vec<NodeId>> = path_lens
            .iter()
            .map(|len| (0..*len).map(|_| g.add_node(make_node())).collect())
            .collect();

        let e = g.add_node(make_node());
        let r = g.add_node(make_node());
        let kk = g.add_node(make_node());
        g.add_bi_edge(e, r, EdgeType::MinToMin);
        g.add_bi_edge(r, kk, EdgeType::MinToMin);
        for path in &paths {
            g.add_bi_edge(s, path[0], EdgeType::MinToMin);
            for w in path.windows(2) {
                g.add_bi_edge(w[0], w[1], EdgeType::MinToMin);
            }
            g.add_bi_edge(*path.last().unwrap(), e, EdgeType::MinToMin);
        }
        for _ in 0..n {
            let q = g.add_node(make_node());
            g.add_bi_edge(kk, q, EdgeType::MinToMin);
        }

        Collapsed {
            g,
            entrance: s,
            preds_in,
            paths,
        }
    }

    /// The pairing a successful `sb_pair_ends` would return for the identity assignment: path `i` with
    /// the `i`-th predecessor and the `i`-th successor. Built from the graph rather than by hand, so
    /// the edge types are the real ones.
    fn identity_pairing(g: &DbgGraph, chain: &SbSharedChain) -> SbPairing {
        SbPairing {
            pred_for: preds(g, chain.left[0])
                .into_iter()
                .map(|(o, t)| (o.node, t))
                .collect(),
            succ_for: succs(g, *chain.right.last().unwrap())
                .into_iter()
                .map(|(o, t)| (o.node, t))
                .collect(),
        }
    }

    /// S -> M1 -> E, S -> M2 -> E, with flanks either side.
    #[test]
    fn simple_bubble_is_found_and_marked_simple() {
        let mut g = DbgGraph::new(3);
        let f0 = g.add_node(make_node());
        let s = g.add_node(make_node());
        let m1 = g.add_node(make_node());
        let m2 = g.add_node(make_node());
        let e = g.add_node(make_node());
        let f1 = g.add_node(make_node());
        g.add_bi_edge(f0, s, EdgeType::MinToMin);
        g.add_bi_edge(s, m1, EdgeType::MinToMin);
        g.add_bi_edge(s, m2, EdgeType::MinToMin);
        g.add_bi_edge(m1, e, EdgeType::MinToMin);
        g.add_bi_edge(m2, e, EdgeType::MinToMin);
        g.add_bi_edge(e, f1, EdgeType::MinToMin);

        let (sbs, _) = find_all(&g);
        let hit: Vec<_> = sbs.iter().filter(|sb| sb.entrance == omin(s)).collect();
        assert_eq!(hit.len(), 1, "expected exactly one superbubble entered at S");
        assert_eq!(hit[0].exit, omin(e));
        assert_eq!(hit[0].interior.len(), 2);
        assert_eq!(hit[0].paths.len(), 2);
        assert!(hit[0].paths.iter().all(|p| p.len() == 1));
    }

    /// Three parallel branches — a three-way fork the existing detector cannot see at all.
    #[test]
    fn three_path_superbubble_is_found() {
        let mut g = DbgGraph::new(3);
        let s = g.add_node(make_node());
        let e = g.add_node(make_node());
        let mids: Vec<NodeId> = (0..3).map(|_| g.add_node(make_node())).collect();
        for m in &mids {
            g.add_bi_edge(s, *m, EdgeType::MinToMin);
            g.add_bi_edge(*m, e, EdgeType::MinToMin);
        }
        let (sbs, _) = find_all(&g);
        let hit: Vec<_> = sbs.iter().filter(|sb| sb.entrance == omin(s)).collect();
        assert_eq!(hit.len(), 1);
        assert_eq!(hit[0].paths.len(), 3, "three branches, three paths");
        assert_eq!(hit[0].interior.len(), 3);
    }

    /// One branch two unitigs long: rejected outright by `check_bubble_structure` today.
    #[test]
    fn a_multi_unitig_path_is_found() {
        let mut g = DbgGraph::new(3);
        let s = g.add_node(make_node());
        let a1 = g.add_node(make_node());
        let a2 = g.add_node(make_node());
        let b = g.add_node(make_node());
        let e = g.add_node(make_node());
        g.add_bi_edge(s, a1, EdgeType::MinToMin);
        g.add_bi_edge(a1, a2, EdgeType::MinToMin);
        g.add_bi_edge(a2, e, EdgeType::MinToMin);
        g.add_bi_edge(s, b, EdgeType::MinToMin);
        g.add_bi_edge(b, e, EdgeType::MinToMin);

        let (sbs, _) = find_all(&g);
        let hit: Vec<_> = sbs.iter().filter(|sb| sb.entrance == omin(s)).collect();
        assert_eq!(hit.len(), 1);
        assert_eq!(hit[0].interior.len(), 3);
        let mut lens: Vec<usize> = hit[0].paths.iter().map(|p| p.len()).collect();
        lens.sort();
        assert_eq!(lens, vec![1, 2], "one path of one unitig, one of two");
    }

    /// A bubble inside a bubble: both are found, and the inner one is reported at depth 1.
    #[test]
    fn nested_superbubbles_report_depth() {
        let mut g = DbgGraph::new(3);
        let s = g.add_node(make_node());
        let x = g.add_node(make_node());
        let i1 = g.add_node(make_node());
        let i2 = g.add_node(make_node());
        let y = g.add_node(make_node());
        let e = g.add_node(make_node());
        let outer = g.add_node(make_node());
        // Outer: s -> x .. y -> e, and s -> outer -> e
        g.add_bi_edge(s, x, EdgeType::MinToMin);
        g.add_bi_edge(x, i1, EdgeType::MinToMin);
        g.add_bi_edge(x, i2, EdgeType::MinToMin);
        g.add_bi_edge(i1, y, EdgeType::MinToMin);
        g.add_bi_edge(i2, y, EdgeType::MinToMin);
        g.add_bi_edge(y, e, EdgeType::MinToMin);
        g.add_bi_edge(s, outer, EdgeType::MinToMin);
        g.add_bi_edge(outer, e, EdgeType::MinToMin);

        let (sbs, _) = find_all(&g);
        let depths = nesting_depths(&sbs);
        let inner = sbs.iter().position(|sb| sb.entrance == omin(x));
        let outer_i = sbs.iter().position(|sb| sb.entrance == omin(s));
        assert!(inner.is_some(), "inner bubble at X not found");
        assert!(outer_i.is_some(), "outer bubble at S not found");
        assert_eq!(depths[inner.unwrap()], 1, "inner sits inside exactly one other");
        assert_eq!(depths[outer_i.unwrap()], 0, "outer sits inside nothing");
    }

    /// A cycle in the interior must not be reported as a superbubble.
    #[test]
    fn a_cycle_is_rejected() {
        let mut g = DbgGraph::new(3);
        let s = g.add_node(make_node());
        let a = g.add_node(make_node());
        let b = g.add_node(make_node());
        g.add_bi_edge(s, a, EdgeType::MinToMin);
        g.add_bi_edge(s, b, EdgeType::MinToMin);
        g.add_bi_edge(a, b, EdgeType::MinToMin);
        g.add_bi_edge(b, s, EdgeType::MinToMin); // back to the entrance

        let r = find_superbubble(&g, omin(s));
        assert!(
            matches!(r, Err(SbReject::Cycle) | Err(SbReject::NoExit)),
            "a cycle back to the entrance must be rejected, got {r:?}",
        );
    }

    /// An inverted repeat: the region meets its own reverse complement. `MinToMax` flips the strand,
    /// so the interior would hold both `(n, Min)` and `(n, Max)`.
    #[test]
    fn a_fold_back_is_rejected() {
        let mut g = DbgGraph::new(3);
        let s = g.add_node(make_node());
        let a = g.add_node(make_node());
        g.add_bi_edge(s, a, EdgeType::MinToMin);
        g.add_bi_edge(s, a, EdgeType::MinToMax);
        g.add_bi_edge(a, s, EdgeType::MaxToMax);

        let r = find_superbubble(&g, omin(s));
        assert!(r.is_err(), "an inverted repeat must not be a superbubble");
    }

    /// A superbubble and its reverse complement are one object, and must be counted once.
    #[test]
    fn strand_twins_dedup_to_one() {
        let mut g = DbgGraph::new(3);
        let s = g.add_node(make_node());
        let m1 = g.add_node(make_node());
        let m2 = g.add_node(make_node());
        let e = g.add_node(make_node());
        g.add_bi_edge(s, m1, EdgeType::MinToMin);
        g.add_bi_edge(s, m2, EdgeType::MinToMin);
        g.add_bi_edge(m1, e, EdgeType::MinToMin);
        g.add_bi_edge(m2, e, EdgeType::MinToMin);

        // Walking from E on the other strand finds the same bubble backwards.
        let fwd = find_superbubble(&g, omin(s));
        let rev = find_superbubble(&g, Oriented { node: e, max: true });
        assert!(fwd.is_ok(), "forward direction should find it");
        if let (Ok(f), Ok(r)) = (&fwd, &rev) {
            assert_eq!(
                twin_key(f.entrance, f.exit),
                twin_key(r.entrance, r.exit),
                "a bubble and its strand twin must share a canonical key"
            );
        }
        let (sbs, rejects) = find_all(&g);
        assert_eq!(sbs.len(), 1, "the twin must be deduped away");
        assert!(rejects.strand_twin >= 1, "and counted as a twin");
    }

    /// A three-way collapsed repeat becomes three independent loci, each carrying one copy.
    #[test]
    fn a_three_path_bundle_splits_into_three_loci() {
        let mut c = collapsed_repeat(&[1, 1, 1]);
        let sb = find_superbubble(&c.g, omin(c.entrance)).expect("a three-path superbubble");
        assert_eq!(sb.paths.len(), 3);
        assert!(is_parallel_bundle(&sb), "three disjoint one-node paths are a bundle");

        let chain = sb_shared_chain(&c.g, &sb).expect("J -> L -> entrance and exit -> R -> K");
        assert_eq!(chain.left.len(), 3, "[J, L, entrance]");
        assert_eq!(chain.right.len(), 3, "[exit, R, K]");

        let pairing = identity_pairing(&c.g, &chain);
        let before = c.g.node_count();
        assert_eq!(
            sb_split_repeat(&mut c.g, &sb, &chain, &pairing),
            SplitOutcome::Applied
        );

        // Two clone sets of six nodes: the shared chain duplicated once per path that moved.
        assert_eq!(c.g.node_count(), before + 12);

        // The originals now serve exactly one locus, and so does every predecessor's junction.
        assert_eq!(
            c.g.out_neighbours_bi(c.entrance, CarryType::Min).len(),
            1,
            "the entrance keeps only the path that stayed put"
        );
        for p in &c.preds_in {
            assert_eq!(
                c.g.out_neighbours_bi(*p, CarryType::Min).len(),
                1,
                "each locus enters its own copy of the repeat"
            );
        }
        // Each branch is now on an unforked chain: one way in, one way out.
        for path in &c.paths {
            let m = path[0];
            assert_eq!(c.g.in_neighbours_bi(m, CarryType::Min).len(), 1);
            assert_eq!(c.g.out_neighbours_bi(m, CarryType::Min).len(), 1);
        }
    }

    /// A branch several unitigs long — the case `check_bubble_structure` rejects outright, and three
    /// quarters of the complex population — moves onto the clones whole.
    #[test]
    fn a_two_path_bundle_with_a_multi_unitig_path_splits() {
        let mut c = collapsed_repeat(&[2, 1]);
        let sb = find_superbubble(&c.g, omin(c.entrance)).expect("a two-path superbubble");
        let mut lens: Vec<usize> = sb.paths.iter().map(|p| p.len()).collect();
        lens.sort();
        assert_eq!(lens, vec![1, 2]);
        assert!(is_parallel_bundle(&sb));

        let chain = sb_shared_chain(&c.g, &sb).expect("a two-way shared chain");
        let pairing = identity_pairing(&c.g, &chain);
        let before = c.g.node_count();
        assert_eq!(
            sb_split_repeat(&mut c.g, &sb, &chain, &pairing),
            SplitOutcome::Applied
        );
        assert_eq!(c.g.node_count(), before + 6, "one clone set of six");

        // The interior edge inside the two-unitig path is untouched: a bundle's paths move wholesale.
        let long = c.paths.iter().find(|p| p.len() == 2).unwrap();
        assert_eq!(
            c.g.out_neighbours_bi(long[0], CarryType::Min)
                .iter()
                .map(|(n, _)| *n)
                .collect::<Vec<_>>(),
            vec![long[1]],
        );
        assert_eq!(c.g.out_neighbours_bi(c.entrance, CarryType::Min).len(), 1);
    }

    /// Coverage is divided by the number of paths, not halved — on the copies *and* on the originals.
    #[test]
    fn coverage_is_divided_by_the_number_of_paths() {
        let mut c = collapsed_repeat(&[1, 1, 1]);
        let sb = find_superbubble(&c.g, omin(c.entrance)).unwrap();
        let chain = sb_shared_chain(&c.g, &sb).unwrap();
        let pairing = identity_pairing(&c.g, &chain);
        let before: BTreeSet<NodeId> = c.g.node_indices().collect();
        let junction = chain.left[0].node;
        assert_eq!(c.g.node_weight(junction).unwrap().counts, 10);

        assert_eq!(
            sb_split_repeat(&mut c.g, &sb, &chain, &pairing),
            SplitOutcome::Applied
        );

        // `div_ceil(10, 3) == 4`, on the original and on both clones. Halving would have given 5.
        assert_eq!(c.g.node_weight(junction).unwrap().counts, 4);
        let clones: Vec<u32> = c
            .g
            .node_indices()
            .filter(|n| !before.contains(n))
            .map(|n| c.g.node_weight(n).unwrap().counts)
            .collect();
        assert_eq!(clones.len(), 12);
        assert!(
            clones.iter().all(|x| *x == 4),
            "every clone carries one locus' worth of reads, got {clones:?}"
        );
    }

    /// Paths that share an interior node are not a bundle: an outer superbubble wrapping a nested one
    /// is the everyday case, and its paths both run through the inner bubble's own entrance and exit.
    #[test]
    fn overlapping_paths_are_not_a_bundle() {
        let mut g = DbgGraph::new(3);
        let s = g.add_node(make_node());
        let x = g.add_node(make_node());
        let i1 = g.add_node(make_node());
        let i2 = g.add_node(make_node());
        let y = g.add_node(make_node());
        let e = g.add_node(make_node());
        let outer = g.add_node(make_node());
        g.add_bi_edge(s, x, EdgeType::MinToMin);
        g.add_bi_edge(x, i1, EdgeType::MinToMin);
        g.add_bi_edge(x, i2, EdgeType::MinToMin);
        g.add_bi_edge(i1, y, EdgeType::MinToMin);
        g.add_bi_edge(i2, y, EdgeType::MinToMin);
        g.add_bi_edge(y, e, EdgeType::MinToMin);
        g.add_bi_edge(s, outer, EdgeType::MinToMin);
        g.add_bi_edge(outer, e, EdgeType::MinToMin);

        let out = find_superbubble(&g, omin(s)).expect("the outer superbubble");
        assert!(
            !is_parallel_bundle(&out),
            "two of its paths share X and Y, so no path owns them"
        );
        let inner = find_superbubble(&g, omin(x)).expect("the inner superbubble");
        assert!(is_parallel_bundle(&inner), "the inner one is disjoint");
    }

    /// A path straight from entrance to exit is an indel bubble, not a collapsed repeat: there is
    /// nothing to move onto a clone set.
    #[test]
    fn an_empty_path_is_not_a_bundle() {
        let mut g = DbgGraph::new(3);
        let s = g.add_node(make_node());
        let m = g.add_node(make_node());
        let e = g.add_node(make_node());
        g.add_bi_edge(s, m, EdgeType::MinToMin);
        g.add_bi_edge(m, e, EdgeType::MinToMin);
        g.add_bi_edge(s, e, EdgeType::MinToMin);

        let sb = find_superbubble(&g, omin(s)).expect("s .. e is still a superbubble");
        assert!(sb.paths.iter().any(|p| p.is_empty()), "one path is the bare edge");
        assert!(!is_parallel_bundle(&sb));
    }

    /// The junction must fork exactly as many ways as there are paths. A two-way junction feeding a
    /// three-way fork means one of the three has no locus to belong to.
    #[test]
    fn a_junction_of_the_wrong_degree_has_no_shared_chain() {
        let mut c = collapsed_repeat(&[1, 1, 1]);
        let sb = find_superbubble(&c.g, omin(c.entrance)).unwrap();
        let chain = sb_shared_chain(&c.g, &sb).expect("three preds, three paths");
        let junction = chain.left[0].node;

        // Drop one predecessor: the junction now forks two ways under a three-way bubble.
        let stray = c.preds_in[2];
        for e in c.g.edges_between(stray, junction) {
            c.g.remove_edge(e);
        }
        for e in c.g.edges_between(junction, stray) {
            c.g.remove_edge(e);
        }
        assert!(
            sb_shared_chain(&c.g, &sb).is_none(),
            "a 2-way junction under a 3-path bubble is not this structure"
        );
    }

    /// The bijection is the gate on every split, and it is deliberately stricter than a maximum
    /// matching. Tested directly: it is the predicate that decides whether the graph is touched at all,
    /// and a `None` here is exactly "the surgery is never reached".
    #[test]
    fn only_a_strict_bijection_is_accepted() {
        let clean = vec![vec![true, false, false], vec![false, true, false], vec![false, false, true]];
        assert_eq!(bijection(&clean), Some(vec![0, 1, 2]));

        let two_supports = vec![vec![true, true], vec![false, true]];
        assert_eq!(bijection(&two_supports), None, "an ambiguous path is not a guess");

        let no_support = vec![vec![false, false], vec![false, true]];
        assert_eq!(bijection(&no_support), None, "an unsupported path decides nothing");

        // A maximum matching would reject this too, but only after searching; the point is that one
        // neighbour cannot serve two loci, however the rows are read.
        let collision = vec![vec![false, true], vec![false, true]];
        assert_eq!(bijection(&collision), None, "one neighbour, two paths");
    }

    // ── the coverage-only collapse ───────────────────────────────────────────

    /// Give every node of path `i` the coverage `covs[i]`, leaving the shared chain alone.
    fn set_path_counts(c: &mut Collapsed, covs: &[u32]) {
        for (path, cov) in c.paths.iter().zip(covs) {
            for n in path {
                c.g.node_weight_mut(*n).unwrap().counts = *cov;
            }
        }
    }

    /// **The regression the threshold exists for.** Three paths at equal coverage is what a collapsed
    /// three-copy repeat looks like; nothing may be deleted.
    #[test]
    fn a_bundle_with_equal_coverage_is_left_alone() {
        let mut c = collapsed_repeat(&[1, 1, 1]);
        set_path_counts(&mut c, &[40, 40, 40]);
        let sb = find_superbubble(&c.g, omin(c.entrance)).unwrap();
        assert!(coverage_winner(&c.g, &sb, 0.1).is_none());

        let before = c.g.node_count();
        let mut stats = SbCollapseStats::default();
        assert!(!collapse_superbubbles_by_coverage(&mut c.g, 0.1, &mut stats));
        assert_eq!(c.g.node_count(), before, "the graph must be untouched");
        assert!(stats.no_clear_winner >= 1);
        assert_eq!(stats.applied, 0);
        stats.report(); // also exercises the accounting assert
    }

    /// One dominant path, the other two noise: collapse onto the winner and delete the losers.
    #[test]
    fn a_bundle_with_one_dominant_path_collapses() {
        let mut c = collapsed_repeat(&[1, 1, 1]);
        set_path_counts(&mut c, &[100, 3, 4]);
        let sb = find_superbubble(&c.g, omin(c.entrance)).unwrap();
        // Which index dominates is whatever `enumerate_paths` produced; ask rather than assume.
        let w = coverage_winner(&c.g, &sb, 0.1).expect("100 vs 3 and 4 is a clear winner");
        let winner_node = sb.paths[w][0].node;

        let before = c.g.node_count();
        let mut stats = SbCollapseStats::default();
        assert!(collapse_superbubbles_by_coverage(&mut c.g, 0.1, &mut stats));
        assert_eq!(stats.applied, 1);
        // Two one-node paths deleted, nothing else.
        assert_eq!(c.g.node_count(), before - 2);
        assert!(c.g.contains_node(winner_node), "the winner survives");
        assert!(c.g.contains_node(c.entrance), "the entrance survives");
        for path in &c.paths {
            if path[0] != winner_node {
                assert!(!c.g.contains_node(path[0]), "a losing path must be gone");
            }
        }
        stats.report();
    }

    /// A multi-unitig path is judged by its *weakest* node, so one well-covered unitig cannot carry a
    /// noise unitig through the test.
    #[test]
    fn path_coverage_is_the_weakest_node() {
        let mut c = collapsed_repeat(&[2, 1]);
        let sb = find_superbubble(&c.g, omin(c.entrance)).unwrap();
        let long = sb.paths.iter().find(|p| p.len() == 2).unwrap().clone();
        c.g.node_weight_mut(long[0].node).unwrap().counts = 100;
        c.g.node_weight_mut(long[1].node).unwrap().counts = 2;
        assert_eq!(path_coverage(&c.g, &long), 2, "the mean would have said 51");
    }

    /// A superbubble that is a plain simple bubble belongs to `corrector`; collapsing it here as well
    /// would act on the same locus twice.
    #[test]
    fn a_simple_bubble_is_left_to_the_simple_popper() {
        let mut g = DbgGraph::new(3);
        let f0 = g.add_node(make_node());
        let s = g.add_node(make_node());
        let m1 = g.add_node(make_node());
        let m2 = g.add_node(make_node());
        let e = g.add_node(make_node());
        let f1 = g.add_node(make_node());
        g.add_bi_edge(f0, s, EdgeType::MinToMin);
        g.add_bi_edge(s, m1, EdgeType::MinToMin);
        g.add_bi_edge(s, m2, EdgeType::MinToMin);
        g.add_bi_edge(m1, e, EdgeType::MinToMin);
        g.add_bi_edge(m2, e, EdgeType::MinToMin);
        g.add_bi_edge(e, f1, EdgeType::MinToMin);
        g.node_weight_mut(m1).unwrap().counts = 100;
        g.node_weight_mut(m2).unwrap().counts = 2;

        let before = g.node_count();
        let mut stats = SbCollapseStats::default();
        assert!(!collapse_superbubbles_by_coverage(&mut g, 0.1, &mut stats));
        assert_eq!(g.node_count(), before);
        assert!(stats.simple_equivalent >= 1, "recognised as the other pass's business");
        assert_eq!(stats.applied, 0);
        stats.report();
    }

    /// Detection must not depend on the order edges were inserted.
    #[test]
    fn detection_is_insertion_order_independent() {
        let build = |flip: bool| {
            let mut g = DbgGraph::new(3);
            let s = g.add_node(make_node());
            let m1 = g.add_node(make_node());
            let m2 = g.add_node(make_node());
            let e = g.add_node(make_node());
            if flip {
                g.add_bi_edge(s, m2, EdgeType::MinToMin);
                g.add_bi_edge(s, m1, EdgeType::MinToMin);
                g.add_bi_edge(m2, e, EdgeType::MinToMin);
                g.add_bi_edge(m1, e, EdgeType::MinToMin);
            } else {
                g.add_bi_edge(s, m1, EdgeType::MinToMin);
                g.add_bi_edge(s, m2, EdgeType::MinToMin);
                g.add_bi_edge(m1, e, EdgeType::MinToMin);
                g.add_bi_edge(m2, e, EdgeType::MinToMin);
            }
            let (sbs, _) = find_all(&g);
            sbs.iter()
                .map(|sb| (sb.interior.len(), sb.paths.len()))
                .collect::<Vec<_>>()
        };
        assert_eq!(build(false), build(true));
    }
}
