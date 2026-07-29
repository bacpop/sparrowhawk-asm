//! Using a larger k as *evidence* to judge the branches of a smaller-k assembly graph.
//!
//! The idea in one line: **the k2 k-mer set is already a compressed index of read content at k2-base
//! resolution.** Asking "did any read contain this exact 63-mer, at least `min_count` times?" is a
//! single hash lookup. SKESA answers the same question by extending each branch ~100 steps and seeing
//! which survives; we get it for the price of a lookup, and without ever touching the reads again.
//!
//! What the bubbles in this assembler's k=31 graph actually *are* is worth stating, because it decides
//! what this module is for. They are **not** error bubbles — `min_count` already removed those. They are
//! **collapsed two-copy repeats**: the flanking context is a repeat present twice in the genome (hence
//! double coverage), and the two branches are its two distinct copies, each at single-locus coverage.
//! Both branches are real sequence. A coverage heuristic cannot possibly choose correctly between them,
//! because their coverages are identical — so today one real repeat copy is deleted and the other
//! duplicated in its place.

use nohash_hasher::NoHashHasher;
use std::borrow::Cow;
use std::{
    collections::{BTreeSet, HashMap},
    hash::BuildHasherDefault,
};

use sparrowhawk_graph::{CarryType, DbgGraph, EdgeType, NodeId, NodeStruct};

use crate::algorithms::corrector::BubbleParts;
use crate::algorithms::shrinker::Shrinkable;
use crate::bit_encoding::{UInt, U256, U512};
use crate::graph_works::populate_neighbours;
use crate::kmer::Kmer;
use crate::preprocessing::PreprocessedK;
use crate::graph_works::{spell_path, SpellError};

/// The larger-k graph, kept purely as evidence about the reads.
///
/// Built with unitig compaction only — no bubble collapse, no dead-end removal, no contig collapse. It
/// must stay faithful to what the reads actually said; the moment we start "correcting" it, it stops
/// being evidence.
pub struct EvidenceGraph {
    /// The evidence k (k2).
    pub k: usize,
    /// The compacted k2 graph. We need it, and not merely the k-mer set, to answer one question: does
    /// the region *fork* at k2? A k-mer set can say "this sequence was in the reads"; only the graph can
    /// say "and at k2 there is no ambiguity here".
    pub graph: DbgGraph,
    /// canonical k2-mer -> (node holding it, its offset within that node's `abs_ind`).
    ///
    /// The key set *is* the min-count-filtered k2-mer set — `from_kmer_map` creates a node for every key
    /// of `themap` and for no hash outside it — so this doubles as the presence test. The offset is what
    /// lets us tell a run from a fork.
    pub kmer_to_pos: HashMap<u64, (NodeId, u32), BuildHasherDefault<NoHashHasher<u64>>>,
}

/// Build the evidence graph from a preprocessed k2, consuming it.
///
/// `pre` is dropped on return: the evidence side never spells sequence, so its `thedict` is dead weight,
/// and `themap`'s neighbour lists now live inside the graph.
pub fn build_evidence<IntT>(mut pre: PreprocessedK<IntT>) -> EvidenceGraph
where
    IntT: for<'a> UInt<'a>,
{
    let k = pre.k;
    log::info!("Building the k={k} evidence graph from {} k-mers", pre.themap.len());

    populate_neighbours(k, &mut pre.themap, &pre.maxmindict);
    let mut graph = DbgGraph::from_kmer_map(k, &pre.themap);
    graph.remove_self_loops();
    graph.shrink(); // loops to a fixed point internally

    let mut kmer_to_pos: HashMap<u64, (NodeId, u32), BuildHasherDefault<NoHashHasher<u64>>> =
        HashMap::with_capacity_and_hasher(pre.themap.len(), BuildHasherDefault::default());
    for n in graph.node_indices() {
        for (i, h) in graph.node_weight(n).unwrap().abs_ind.iter().enumerate() {
            kmer_to_pos.insert(*h, (n, i as u32));
        }
    }

    log::info!(
        "Evidence graph (k={k}): {} unitigs, {} edges, indexing {} k-mers",
        graph.node_count(),
        graph.edge_count(),
        kmer_to_pos.len()
    );

    EvidenceGraph {
        k,
        graph,
        kmer_to_pos,
    }
}

/// A maximal stretch of consecutive k2-mers that landed in the same evidence node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Run {
    /// The evidence node they landed in.
    pub node: NodeId,
    /// Offset within that node's `abs_ind` of the first k2-mer of the run.
    pub first: u32,
    /// Offset of the last.
    pub last: u32,
}

/// What the evidence graph has to say about one candidate branch.
#[derive(Debug, Clone, Default)]
pub struct BranchEvidence {
    /// k2-mers in the *discriminating* window — those overlapping the divergent middle. These are the
    /// only ones that differ between the two branches, so they are the only ones that can decide.
    pub n_window: usize,
    /// How many of those are present in the evidence graph.
    pub n_present: usize,
    /// Fraction of the *flank* k2-mers present. The flank is shared by both branches, so it
    /// discriminates nothing — but if it is not corroborated at k2, our context is not trustworthy and
    /// the whole test is void. This is the guard, and the real job the flanking unitigs do.
    pub flank_fraction: f64,
    /// Runs over the **full** path (flanks included). `len() == 1` means the region does not fork at k2
    /// — which is what makes the repeat *resolvable*, as opposed to merely confirmed.
    pub runs: Vec<Run>,
    /// Runs restricted to the discriminating window.
    pub window_runs: Vec<Run>,
}

impl BranchEvidence {
    /// Every discriminating k2-mer present: the reads corroborate this branch.
    pub fn is_supported(&self, min_evidence: usize) -> bool {
        self.n_window >= min_evidence && self.n_present == self.n_window
    }

    /// No discriminating k2-mer present at all: the reads never contained this branch in a k2-length
    /// context.
    pub fn is_absent(&self) -> bool {
        self.n_window > 0 && self.n_present == 0
    }

    /// The region is a single unbranched stretch at k2 — so k2 spans whatever collapsed at k1, and the
    /// locus can be *resolved*, not merely protected.
    pub fn is_unforked_at_k2(&self) -> bool {
        self.runs.len() == 1
    }
}

impl EvidenceGraph {
    /// Look up every k2-mer of `seq`, splitting them into the discriminating window and the flanking
    /// guard.
    ///
    /// **Not generic**, and deliberately so. The packed-k-mer width this needs is set by *this graph's*
    /// k, which the graph already knows — so it dispatches on `self.k` rather than making every caller
    /// carry a type parameter sized for the evidence k. That matters: `judge_bubble`,
    /// `judge_superbubble`, `correct_with_evidence`, `survey_bubbles` and every superbubble entry point
    /// would otherwise need a second type parameter threaded through them, which is exactly why the
    /// whole run used to take one width from the largest k and pay for it on the assembly k's pass.
    ///
    /// The cost is four monomorphisations of `evidence_for_impl` instead of one; `EvidenceGraph` itself
    /// stores no packed k-mers, so nothing of this width escapes the call.
    pub fn evidence_for(&self, seq: &[u8], window: (usize, usize)) -> BranchEvidence {
        match self.k {
            0..=32 => self.evidence_for_impl::<u64>(seq, window),
            33..=64 => self.evidence_for_impl::<u128>(seq, window),
            65..=128 => self.evidence_for_impl::<U256>(seq, window),
            _ => self.evidence_for_impl::<U512>(seq, window),
        }
    }

    /// `window` is the half-open range of k2-mer *start positions* that overlap the divergent middle.
    ///
    /// Hashing reuses `Kmer`'s light iterators, which yield `(hc, hnc, b)` without the packed k-mer —
    /// exactly what a presence test needs. `seq` is our own reconstructed ACGT, so there are no Ns and
    /// no quality mask, and the roller never restarts its window: one hash per position.
    fn evidence_for_impl<IntT>(&self, seq: &[u8], window: (usize, usize)) -> BranchEvidence
    where
        IntT: for<'a> UInt<'a>,
    {
        let mut ev = BranchEvidence::default();
        let mut it = match Kmer::<IntT>::new(Cow::Borrowed(seq), seq.len(), None, self.k, 0, true) {
            Some(it) => it,
            // Shorter than k2: no evidence to be had.
            None => return ev,
        };

        let (lo, hi) = window;
        let mut flank_total = 0usize;
        let mut flank_present = 0usize;
        let mut pos = 0usize;

        let mut hc = it.get_curr_hash_and_bases().0;
        loop {
            let hit = self.kmer_to_pos.get(&hc).copied();

            // Runs over the full path -> the fork test.
            match (hit, ev.runs.last_mut()) {
                (Some((n, off)), Some(r)) if r.node == n => r.last = off,
                (Some((n, off)), _) => ev.runs.push(Run {
                    node: n,
                    first: off,
                    last: off,
                }),
                (None, _) => {}
            }

            if pos >= lo && pos < hi {
                ev.n_window += 1;
                if let Some((n, off)) = hit {
                    ev.n_present += 1;
                    match ev.window_runs.last_mut() {
                        Some(r) if r.node == n => r.last = off,
                        _ => ev.window_runs.push(Run {
                            node: n,
                            first: off,
                            last: off,
                        }),
                    }
                }
            } else {
                flank_total += 1;
                if hit.is_some() {
                    flank_present += 1;
                }
            }

            match it.get_next_hash_and_bases() {
                Some((next, _, _)) => hc = next,
                None => break,
            }
            pos += 1;
        }

        ev.flank_fraction = if flank_total == 0 {
            1.0
        } else {
            flank_present as f64 / flank_total as f64
        };
        ev
    }
}

/// Read a node's k-mers in the order a walk on `carry` traverses them.
///
/// `abs_ind` is stored in the strand recorded by `innerdir`. Walking the node on a different strand
/// means the stored list runs backwards relative to us, so reverse it. This is the rule
/// `collapser.rs:137-141` already uses, and the clone matters: the reversal must never be written back
/// into the graph.
pub(crate) fn oriented(g: &DbgGraph, n: NodeId, carry: CarryType) -> Vec<u64> {
    let w: &NodeStruct = g.node_weight(n).unwrap();
    let mut hashes = w.abs_ind.clone();
    if let Some(inn) = w.innerdir {
        if carry != inn.get_from_and_to().0 {
            hashes.reverse();
        }
    }
    hashes
}

/// Walk backwards from `start` along a unique path, collecting up to `budget` k-mers.
///
/// Returns them in walk order — i.e. ending adjacent to `start`. Stops at the first branch: an ambiguous
/// extension is no context at all. `truncated` reports whether we ran out of budget rather than out of
/// graph, so the caller can say so instead of silently capping.
///
/// Walking **backwards** from a node on carry `c`, the previous node's carry is the *source* carry of
/// the edge, `get_from_and_to().0`. (Forwards it is the *target* carry, `.1`.) Getting this the wrong
/// way round is the single easiest mistake here.
pub(crate) fn gather_left(
    g: &DbgGraph,
    from: (NodeId, EdgeType),
    budget: usize,
) -> (Vec<u64>, bool, Option<(NodeId, CarryType)>) {
    let mut out: Vec<u64> = Vec::new();
    let mut node = from.0;
    let mut carry = from.1.get_from_and_to().0;
    let mut truncated = false;

    loop {
        let mut hashes = oriented(g, node, carry);
        if out.len() + hashes.len() > budget {
            // Keep the k-mers nearest the bubble: the tail of this node's walk.
            let keep = budget - out.len();
            hashes.drain(..hashes.len() - keep);
            truncated = true;
        }
        hashes.append(&mut out);
        out = hashes;

        if out.len() >= budget {
            return (out, truncated, None);
        }
        let prev = g.in_neighbours_bi(node, carry);
        if prev.len() != 1 {
            // A branch. This is where a collapsed repeat is *entered* from its several genomic loci, so
            // it is exactly the junction repeat resolution has to disambiguate. Hand it back.
            let stop = if prev.is_empty() {
                None
            } else {
                Some((node, carry))
            };
            return (out, truncated, stop);
        }
        node = prev[0].0;
        carry = prev[0].1.get_from_and_to().0;
    }
}

/// Walk forwards from `end` along a unique path, collecting up to `budget` k-mers, in walk order.
///
/// Forwards from a node on carry `c`, the next node's carry is the *target* carry, `get_from_and_to().1`.
pub(crate) fn gather_right(
    g: &DbgGraph,
    from: (NodeId, EdgeType),
    budget: usize,
) -> (Vec<u64>, bool, Option<(NodeId, CarryType)>) {
    let mut out: Vec<u64> = Vec::new();
    let mut node = from.0;
    let mut carry = from.1.get_from_and_to().1;
    let mut truncated = false;

    loop {
        let mut hashes = oriented(g, node, carry);
        if out.len() + hashes.len() > budget {
            hashes.truncate(budget - out.len());
            truncated = true;
        }
        out.extend(hashes);

        if out.len() >= budget {
            return (out, truncated, None);
        }
        let next = g.out_neighbours_bi(node, carry);
        if next.len() != 1 {
            let stop = if next.is_empty() { None } else { Some((node, carry)) };
            return (out, truncated, stop);
        }
        node = next[0].0;
        carry = next[0].1.get_from_and_to().1;
    }
}

/// One branch's reconstructed walk, plus where its divergent middle sits inside it.
pub struct BranchWalk {
    /// The full k-mer walk: left flank, start, mid, end, right flank.
    pub hashes: Vec<u64>,
    /// Number of left-flank k-mers, i.e. the index of `start` in `hashes`.
    pub n_left: usize,
    /// Number of k-mers in the divergent middle.
    pub n_mid: usize,
    /// Whether either flank hit the context budget.
    pub truncated: bool,
}

impl BranchWalk {
    /// Half-open range of k2-mer **start positions** that overlap the divergent middle.
    ///
    /// The k-mer at walk index `j` contributes the base at sequence position `j + k1 - 1`, so `mid`
    /// occupies bases `n_left + k1 ..= n_left + n_mid + k1 - 1`. A k2-mer starting at `p` covers
    /// `[p, p + k2 - 1]`, so it overlaps the middle iff
    /// `p <= n_left + n_mid + k1 - 1` and `p + k2 - 1 >= n_left + k1`.
    ///
    /// This windowing is not a nicety. The flank k2-mers are *shared by both branches* and discriminate
    /// nothing; testing over the whole path dilutes a completely-absent branch to ~77% present, which
    /// reads as "partial" and is never acted on. The feature would be a silent no-op.
    pub fn window(&self, k1: usize, k2: usize) -> (usize, usize) {
        let n_k2 = (self.hashes.len() + k1 - 1).saturating_sub(k2 - 1);
        let first_mid_base = self.n_left + k1;
        let last_mid_base = self.n_left + self.n_mid + k1 - 1;
        let lo = (first_mid_base + 1).saturating_sub(k2);
        let hi = (last_mid_base + 1).min(n_k2);
        (lo.min(hi), hi)
    }
}

/// Reconstruct the k-mer walk for one branch of a bubble, with flanking context on both sides.
///
/// Flanks are **mandatory**: a bubble path on its own spells only 33–63 bases, which yields zero or one
/// k2-mers. Without context there is nothing to look up.
///
/// We owe `spell_path` only the **order** of the hashes, never their strands — it recovers those from the
/// `k-1` overlap. And it self-checks: if the orientation bookkeeping above is wrong, `spell_path` returns
/// `NotAWalk` pointing at the exact k-mer where it broke, which the caller counts separately.
pub fn branch_walk(
    g: &DbgGraph,
    p: &BubbleParts,
    branch: usize,
    flank_budget: usize,
) -> Option<BranchWalk> {
    let left = p.left?;
    let (left_hashes, lt, _) = gather_left(g, left, flank_budget);
    let (right_hashes, rt, _) = gather_right(g, p.right, flank_budget);

    let mid = oriented(g, p.mid[branch].0, p.midct[branch]);
    let n_mid = mid.len();

    let mut hashes = left_hashes;
    let n_left = hashes.len();
    hashes.extend(oriented(g, p.start, CarryType::Min));
    hashes.extend(mid);
    hashes.extend(oriented(g, p.end, p.endct));
    hashes.extend(right_hashes);

    Some(BranchWalk {
        hashes,
        n_left,
        n_mid,
        truncated: lt || rt,
    })
}

/// Spell a branch walk. Errors are the caller's cue that our orientation bookkeeping is wrong.
pub fn spell_branch<IntT>(
    w: &BranchWalk,
    dict: &HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    k1: usize,
) -> Result<Vec<u8>, SpellError>
where
    IntT: for<'a> UInt<'a>,
{
    spell_path(&w.hashes, dict, k1)
}

/// Which predecessor and which successor of the collapsed repeat belong with which branch.
#[derive(Debug, Clone, Copy)]
pub struct Pairing {
    /// `pred_for[i]` is the predecessor the reads pair with branch `i`.
    pub pred_for: [(NodeId, EdgeType); 2],
    /// `succ_for[i]` is the successor the reads pair with branch `i`.
    pub succ_for: [(NodeId, EdgeType); 2],
}

/// Work out, from the evidence graph, which predecessor and successor of the collapsed repeat go with
/// which branch.
///
/// The test falls straight out of what `ResolvableRepeat` already means. For a resolvable bubble each
/// branch's whole path is a **single** k2 unitig — so extending the walk out through a *specific*
/// neighbour and asking "is it *still* a single unitig, with every k2-mer present?" is exactly the
/// pairing test. Only the true neighbour keeps the path unbroken at k2; the wrong one spells a sequence
/// no read ever contained.
///
/// Returns `None` unless the pairing is unambiguous on **both** sides — exactly one neighbour supported
/// per branch, and a different one for each. Guessing here would manufacture a misassembly, which is
/// strictly worse than the repeat collapse it replaces.
pub fn pair_ends<IntT>(
    ev: &EvidenceGraph,
    g: &DbgGraph,
    p: &BubbleParts,
    chain: &SharedChain,
    dict: &HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    k1: usize,
    flank_budget: usize,
) -> Option<Pairing>
where
    IntT: for<'a> UInt<'a>,
{
    let junction = chain.left[0];
    let exit = *chain.right.last().unwrap();
    let preds = g.in_neighbours_bi(junction.0, junction.1);
    let succs = g.out_neighbours_bi(exit.0, exit.1);
    if preds.len() != 2 || succs.len() != 2 {
        return None;
    }

    // The shared chain's own hashes, in walk order. `oriented` handles the strand; we owe `spell_path`
    // only the order.
    let left_shared: Vec<u64> = chain
        .left
        .iter()
        .flat_map(|(n, c)| oriented(g, *n, *c))
        .collect();
    let right_shared: Vec<u64> = chain
        .right
        .iter()
        .flat_map(|(n, c)| oriented(g, *n, *c))
        .collect();

    // Is `[pred .. shared .. mid .. shared'] (+ succ)` a single, fully-present k2 unitig?
    let whole = |hashes: &[u64]| -> bool {
        let Ok(seq) = spell_path(hashes, dict, k1) else {
            return false;
        };
        let e = ev.evidence_for(&seq, (0, 0));
        e.flank_fraction >= 1.0 && e.runs.len() == 1
    };

    let mut pred_sup = [[false; 2]; 2];
    let mut succ_sup = [[false; 2]; 2];

    for bi in 0..2 {
        let mid = oriented(g, p.mid[bi].0, p.midct[bi]);

        for (ni, pred) in preds.iter().enumerate() {
            let (tail, _, _) = gather_left(g, *pred, flank_budget);
            let mut h = tail;
            h.extend(left_shared.iter().copied());
            h.extend(mid.iter().copied());
            h.extend(right_shared.iter().copied());
            pred_sup[bi][ni] = whole(&h);
        }
        for (ni, succ) in succs.iter().enumerate() {
            let (head, _, _) = gather_right(g, *succ, flank_budget);
            let mut h = left_shared.clone();
            h.extend(mid.iter().copied());
            h.extend(right_shared.iter().copied());
            h.extend(head);
            succ_sup[bi][ni] = whole(&h);
        }
    }

    // Demand a clean bijection on both sides.
    let bij = |sup: &[[bool; 2]; 2]| -> Option<[usize; 2]> {
        let a: Vec<usize> = (0..2).filter(|&i| sup[0][i]).collect();
        let b: Vec<usize> = (0..2).filter(|&i| sup[1][i]).collect();
        if a.len() == 1 && b.len() == 1 && a[0] != b[0] {
            Some([a[0], b[0]])
        } else {
            None
        }
    };
    let pi = bij(&pred_sup)?;
    let si = bij(&succ_sup)?;

    Some(Pairing {
        pred_for: [preds[pi[0]], preds[pi[1]]],
        succ_for: [succs[si[0]], succs[si[1]]],
    })
}

/// How many nodes of shared repeat we are willing to duplicate on each side of a bubble.
///
/// Measured on the bench data the shared chain is 2-3 nodes (~44 bases) — short, which is precisely why
/// k2=63 can span it. A long chain would mean a long repeat, which k2 could not have resolved in the
/// first place. So this is a sanity bound, not a tuning knob.
pub(crate) const MAX_SHARED_CHAIN: usize = 8;

/// The shared repeat between the fork point and the bubble, in walk order, each with its traversal carry.
///
/// Left: `[J, .., L, start]`, where `J` is the node the repeat is *entered* at (two predecessors).
/// Right: `[end, R, .., K]`, where `K` is the node it is *exited* at (two successors).
///
/// These are exactly the nodes the genome traverses twice, and so exactly the nodes that must be
/// duplicated for the two loci to become independent.
pub struct SharedChain {
    /// `[J, .., L, start]` — the repeat's entry side.
    pub left: Vec<(NodeId, CarryType)>,
    /// `[end, R, .., K]` — the repeat's exit side.
    pub right: Vec<(NodeId, CarryType)>,
}

fn shared_chain(g: &DbgGraph, p: &BubbleParts) -> Option<SharedChain> {
    // Walk back from `start` through unique predecessors until a node that has two: the fork.
    let mut left = vec![(p.start, CarryType::Min)];
    loop {
        let (cur, carry) = *left.last().unwrap();
        let prev = g.in_neighbours_bi(cur, carry);
        match prev.len() {
            1 => left.push((prev[0].0, prev[0].1.get_from_and_to().0)),
            2 => break, // `cur` is the fork point
            _ => return None,
        }
        if left.len() > MAX_SHARED_CHAIN {
            return None;
        }
    }
    left.reverse(); // [J, .., L, start]

    // Mirror forwards from `end`. Forwards, the next node's carry is the *target* carry.
    let mut right = vec![(p.end, p.endct)];
    loop {
        let (cur, carry) = *right.last().unwrap();
        let next = g.out_neighbours_bi(cur, carry);
        match next.len() {
            1 => right.push((next[0].0, next[0].1.get_from_and_to().1)),
            2 => break, // `cur` is the exit point
            _ => return None,
        }
        if right.len() > MAX_SHARED_CHAIN {
            return None;
        }
    }

    // A node cannot be duplicated into two independent paths if it appears twice in the chain, nor may a
    // branch be part of its own shared context.
    let all: Vec<NodeId> = left.iter().chain(right.iter()).map(|x| x.0).collect();
    let uniq: BTreeSet<NodeId> = all.iter().copied().collect();
    if uniq.len() != all.len() || uniq.contains(&p.mid[0].0) || uniq.contains(&p.mid[1].0) {
        return None;
    }

    Some(SharedChain { left, right })
}

/// Why a planned split did not happen.
///
/// This used to be a bare `bool` folded into a single `split_skipped_stale` counter, which made the
/// number uninterpretable: it conflated *staleness* (the neighbourhood moved between judging and
/// splitting) with *surgery failure* (the neighbourhood was intact but an expected edge was missing).
/// Those have different causes and different fixes, so they are counted apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitOutcome {
    /// The split was applied.
    Applied,
    /// `bubble_parts` no longer recognises a bubble at this start. Usually `shrink` absorbed the
    /// junction into a longer unitig after an earlier split in the same pass.
    StaleNoBubble,
    /// Still a bubble, but with different mid nodes than the ones that were judged.
    StaleMidsChanged,
    /// Intact, but an expected edge in the left shared chain was absent.
    SurgeryFailedLeft,
    /// Intact, but an expected edge in the right shared chain was absent.
    SurgeryFailedRight,
    /// Intact, but the branch's edge to the bubble end was absent.
    SurgeryFailedMid,
}

/// Duplicate the shared repeat so each branch gets its own copy of it, and rewire.
///
/// Before, the genome's two loci are forced through one shared chain, so no contig can walk through and
/// the coverage heuristic deletes one of them:
///
/// ```text
///   P0 -.                                    .- Q0
///        >- [J .. start] -+- A -+- [end .. K] <
///   P1 -'                 `- B -'            `- Q1
/// ```
///
/// After, they are independent, each carrying one real copy. `shrink` then folds each into a single
/// unitig, so contigs run **straight through** the repeat:
///
/// ```text
///   P0 - [J  .. start ] - A - [end  .. K ] - Q0
///   P1 - [J' .. start'] - B - [end' .. K'] - Q1
/// ```
///
/// Every clone is an **exact** copy — same `abs_ind`, same `innerdir` — and every edge between two clones
/// reuses the edge type of the original it mirrors. No orientation is ever recomputed. That is what makes
/// this safe: the bidirected bookkeeping is the easiest thing here to get subtly wrong, so the surgery is
/// arranged so it never has to do any.
///
/// Coverage is halved on both copies: each now carries one locus' worth of reads rather than two.
fn split_repeat(
    g: &mut DbgGraph,
    p: &BubbleParts,
    chain: &SharedChain,
    pred_for: [(NodeId, EdgeType); 2],
    succ_for: [(NodeId, EdgeType); 2],
) -> SplitOutcome {
    // Branch 1 moves onto the clones; branch 0 keeps the originals.
    let b = 1usize;

    let junction = chain.left[0].0;
    let exit = chain.right.last().unwrap().0;

    // Capture every edge type we must reproduce, before anything moves.
    let edge_between = |g: &DbgGraph, a: (NodeId, CarryType), bn: NodeId| -> Option<EdgeType> {
        g.out_neighbours_bi(a.0, a.1)
            .into_iter()
            .find(|(n, _)| *n == bn)
            .map(|(_, t)| t)
    };

    let mut left_edges = Vec::new();
    for w in chain.left.windows(2) {
        match edge_between(g, w[0], w[1].0) {
            Some(t) => left_edges.push(t),
            None => return SplitOutcome::SurgeryFailedLeft,
        }
    }
    let mut right_edges = Vec::new();
    for w in chain.right.windows(2) {
        match edge_between(g, w[0], w[1].0) {
            Some(t) => right_edges.push(t),
            None => return SplitOutcome::SurgeryFailedRight,
        }
    }
    let mid_to_end = match edge_between(g, (p.mid[b].0, p.midct[b]), p.end) {
        Some(t) => t,
        None => return SplitOutcome::SurgeryFailedMid,
    };

    // --- clone the shared nodes ------------------------------------------------------------------
    let mut left_clones = Vec::with_capacity(chain.left.len());
    for (n, _) in &chain.left {
        let mut w = g.node_weight(*n).unwrap().clone();
        w.counts = w.counts.div_ceil(2);
        left_clones.push(g.add_node(w));
    }
    let mut right_clones = Vec::with_capacity(chain.right.len());
    for (n, _) in &chain.right {
        let mut w = g.node_weight(*n).unwrap().clone();
        w.counts = w.counts.div_ceil(2);
        right_clones.push(g.add_node(w));
    }
    // The originals now carry one locus, not two.
    for (n, _) in chain.left.iter().chain(chain.right.iter()) {
        let w = g.node_weight_mut(*n).unwrap();
        w.counts = w.counts.div_ceil(2);
    }

    // --- wire the clone chain, mirroring the originals exactly ------------------------------------
    for (i, t) in left_edges.iter().enumerate() {
        g.add_bi_edge(left_clones[i], left_clones[i + 1], *t);
    }
    for (i, t) in right_edges.iter().enumerate() {
        g.add_bi_edge(right_clones[i], right_clones[i + 1], *t);
    }

    // --- move branch `b` onto the clones ----------------------------------------------------------
    let cut = |g: &mut DbgGraph, a: NodeId, bn: NodeId| {
        for e in g.edges_between(a, bn) {
            g.remove_edge(e);
        }
        for e in g.edges_between(bn, a) {
            g.remove_edge(e);
        }
    };

    cut(g, p.start, p.mid[b].0);
    g.add_bi_edge(*left_clones.last().unwrap(), p.mid[b].0, p.mid[b].1);

    cut(g, p.mid[b].0, p.end);
    g.add_bi_edge(p.mid[b].0, right_clones[0], mid_to_end);

    cut(g, pred_for[b].0, junction);
    g.add_bi_edge(pred_for[b].0, left_clones[0], pred_for[b].1);

    cut(g, exit, succ_for[b].0);
    g.add_bi_edge(*right_clones.last().unwrap(), succ_for[b].0, succ_for[b].1);

    SplitOutcome::Applied
}

/// The fraction of flank k2-mers that must be present before we trust the context at all.
pub(crate) const FLANK_SUPPORT: f64 = 0.9;

/// What the evidence says about a bubble. Every candidate lands in exactly one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// One branch is corroborated at k2 and the other is not present at all: a genuine error bubble.
    /// Collapse onto the survivor. Expect very few of these — `min_count` already removed error k-mers.
    ResolvedError(usize),
    /// Both branches are real, **and** the region does not fork at k2: the larger k spans whatever
    /// collapsed at k1, so this locus can be genuinely *resolved* (see `--multik-resolve-repeats`).
    ResolvableRepeat,
    /// Both branches are real, but k2 forks here too — the repeat is longer than k2 can span. Pop
    /// neither. The contig breaks, but no real sequence is destroyed.
    ProtectedRepeat,
    /// Not enough corroborated flank to build a trustworthy context.
    InconclusiveContext,
    /// Neither branch is corroborated at k2. Coverage at k2 is thinner than at k1, so this is weak
    /// evidence — never act on it.
    InconclusiveAbsent,
    /// Partial support: some discriminating k2-mers present, some not. Do not guess.
    InconclusivePartial,
    /// `spell_path` rejected the walk. **This is our bug, not the data's** — a bubble taken from the
    /// graph is a walk by construction. Counted separately, and it must be zero.
    InconclusiveSpell,
}

/// Ask the evidence graph about one bubble.
///
/// The decision rule is asymmetric on purpose: we require the survivor to be **positively** supported,
/// and never delete a branch merely because the *other* one is absent. The k2 graph is sparser than the
/// k1 graph (fewer k-mers per read, and a lower error-free probability), so absence at k2 is weak
/// evidence; acting on it would silently delete true sequence wherever coverage is thin.
#[allow(clippy::too_many_arguments)]
pub fn judge_bubble<IntT>(
    ev: &EvidenceGraph,
    g: &DbgGraph,
    p: &BubbleParts,
    dict: &HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    k1: usize,
    flank_budget: usize,
    min_evidence: usize,
    max_nodes: usize,
) -> (Verdict, bool)
where
    IntT: for<'a> UInt<'a>,
{
    let mut truncated = false;
    let mut evs: [BranchEvidence; 2] = [BranchEvidence::default(), BranchEvidence::default()];

    for (b, slot) in evs.iter_mut().enumerate() {
        let Some(w) = branch_walk(g, p, b, flank_budget) else {
            return (Verdict::InconclusiveContext, truncated);
        };
        truncated |= w.truncated;

        let seq = match spell_branch(&w, dict, k1) {
            Ok(s) => s,
            Err(_) => return (Verdict::InconclusiveSpell, truncated),
        };

        let window = w.window(k1, ev.k);
        *slot = ev.evidence_for(&seq, window);

        if slot.flank_fraction < FLANK_SUPPORT || slot.n_window < min_evidence {
            return (Verdict::InconclusiveContext, truncated);
        }
    }

    let sup = [
        evs[0].is_supported(min_evidence) && evs[0].window_runs.len() <= max_nodes,
        evs[1].is_supported(min_evidence) && evs[1].window_runs.len() <= max_nodes,
    ];
    let abs = [evs[0].is_absent(), evs[1].is_absent()];

    let verdict = match (sup[0], sup[1]) {
        // Exactly one branch corroborated, the other truly absent: an error bubble.
        (true, false) if abs[1] => Verdict::ResolvedError(0),
        (false, true) if abs[0] => Verdict::ResolvedError(1),

        // Both real. The question is now whether k2 can disentangle them, or only confirm them.
        (true, true) => {
            if evs[0].is_unforked_at_k2() && evs[1].is_unforked_at_k2() {
                Verdict::ResolvableRepeat
            } else {
                Verdict::ProtectedRepeat
            }
        }

        (false, false) if abs[0] && abs[1] => Verdict::InconclusiveAbsent,
        _ => Verdict::InconclusivePartial,
    };

    (verdict, truncated)
}

/// Ask the evidence graph about every bubble, and act on what it says.
///
/// One action: **resolve** (`--multik-resolve-repeats`). Where k2 spans the collapsed repeat *and* the
/// reads pair each predecessor unambiguously with a branch, duplicate the shared repeat so both genomic
/// copies survive and contigs run through. It is the only correction here that throws nothing away.
///
/// There used to be a second action, **protect** — return the judged bubbles so the coverage heuristic
/// would refuse to pop them. It is gone, along with `--multik-protect`. It measured as a regression
/// (a "protected" branch was severed rather than kept, and an isolated node under 100 nt is dropped at
/// `collapser.rs:45-51`, so it lost *both* copies where popping lost one), and it is now unnecessary:
/// `corrector::choose_branch_by_counts` will not touch a bubble whose branches have comparable
/// coverage, which is exactly the case protection existed for.
#[allow(clippy::too_many_arguments)]
pub fn correct_with_evidence<IntT>(
    ev: &EvidenceGraph,
    g: &mut DbgGraph,
    dict: &HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    k1: usize,
    flank_budget: usize,
    min_evidence: usize,
    max_nodes: usize,
    do_resolve: bool,
    stats: &mut MultiKStats,
    prev_skipped: &BTreeSet<NodeId>,
    skipped_out: &mut BTreeSet<NodeId>,
) where
    IntT: for<'a> UInt<'a>,
{
    let candidates: Vec<BubbleParts> = g
        .node_indices()
        .filter_map(|n| crate::algorithms::corrector::bubble_parts(g, n))
        .collect();

    // Splits are collected first and applied afterwards: each one adds nodes and rewires edges, so
    // judging on a graph that is being mutated underneath us would be judging a graph that no longer
    // exists.
    let mut to_split: Vec<(BubbleParts, SharedChain, Pairing)> = Vec::new();

    for p in candidates {
        // Fate of anything the previous round could not split. The retry loop already re-offers these;
        // what we did not know was what happens when it does.
        let revisited = prev_skipped.contains(&p.start);
        if revisited {
            stats.revisit_seen += 1;
        }

        let (v, truncated) =
            judge_bubble::<IntT>(ev, g, &p, dict, k1, flank_budget, min_evidence, max_nodes);
        stats.record(v, truncated);

        if v == Verdict::ResolvableRepeat {
            if revisited {
                stats.revisit_resolvable += 1;
            }
            // k2 spans the repeat, so the reads know which predecessor belongs with which branch.
            // Whether they say so *unambiguously* is what decides if we may split.
            let paired = shared_chain(g, &p).and_then(|c| {
                pair_ends::<IntT>(ev, g, &p, &c, dict, k1, flank_budget).map(|pr| (c, pr))
            });
            match paired {
                Some((c, pr)) => {
                    stats.pairing_ok += 1;
                    if revisited {
                        stats.revisit_paired += 1;
                    }
                    to_split.push((p.clone(), c, pr));
                }
                None => stats.pairing_ambiguous += 1,
            }
        }
    }

    // Anything skipped last round that did not even come back as a candidate. `revisit_seen` counts
    // reappearances, so the remainder vanished — absorbed by `shrink`, or popped.
    stats.revisit_absent = prev_skipped.len().saturating_sub(stats.revisit_seen);

    if do_resolve {
        for (p, c, pr) in &to_split {
            // Re-derive: an earlier split in this same pass may have reshaped this neighbourhood.
            let outcome = match crate::algorithms::corrector::bubble_parts(g, p.start) {
                None => SplitOutcome::StaleNoBubble,
                Some(fresh) if fresh.mid[0].0 != p.mid[0].0 || fresh.mid[1].0 != p.mid[1].0 => {
                    SplitOutcome::StaleMidsChanged
                }
                Some(_) => split_repeat(g, p, c, pr.pred_for, pr.succ_for),
            };
            stats.record_split(outcome);

            if outcome != SplitOutcome::Applied {
                // We *intend* to split this one, we just could not on this pass. Record it so the
                // next round can report what became of it. No veto is needed any more: the popper
                // leaves a comparable-coverage bubble alone, so the locus survives on its own.
                skipped_out.insert(p.start);

                // One line per locus — there are only ~12-25 of these per dataset, so they are worth
                // reading individually rather than inferring from totals. At `info` because `-v` maps
                // to Info and never Debug (`lib.rs`), and this is the whole point of the counter.
                //
                // `still_present` and the degrees are what distinguish the two things a
                // `StaleNoBubble` can mean: the start node destroyed, versus the node alive but no
                // longer a fork because a neighbouring split already separated this locus too. Only
                // the first would be a loss.
                let still_present = g.contains_node(p.start);
                let (od, idg) = if still_present {
                    (g.out_degree(p.start), g.in_degree(p.start))
                } else {
                    (0, 0)
                };
                log::info!(
                    "    split skipped ({outcome:?}): start={:?} present={} out_deg={} in_deg={} \
                     mids={:?}/{:?} chain={}L+{}R preds={:?}/{:?} succs={:?}/{:?}",
                    p.start,
                    still_present,
                    od,
                    idg,
                    p.mid[0].0,
                    p.mid[1].0,
                    c.left.len(),
                    c.right.len(),
                    pr.pred_for[0].0,
                    pr.pred_for[1].0,
                    pr.succ_for[0].0,
                    pr.succ_for[1].0,
                );
            }
        }
    }
}

/// Ask the evidence graph about every bubble in the graph, recording what it said.
///
/// **Read-only on the graph.** This is deliberate: it lets the entire pipeline — flank gathering,
/// orientation bookkeeping, spelling, hashing, lookup, classification — be validated against a real
/// dataset with zero risk to the assembly, and it produces the numbers that decide what acting on the
/// verdicts is actually worth. In particular it proves out `inconclusive_spell == 0`, which is the
/// self-check on the hardest code here.
#[allow(clippy::too_many_arguments)]
pub fn survey_bubbles<IntT>(
    ev: &EvidenceGraph,
    g: &DbgGraph,
    dict: &HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    k1: usize,
    flank_budget: usize,
    min_evidence: usize,
    max_nodes: usize,
    stats: &mut MultiKStats,
) where
    IntT: for<'a> UInt<'a>,
{
    let candidates: Vec<BubbleParts> = g
        .node_indices()
        .filter_map(|n| crate::algorithms::corrector::bubble_parts(g, n))
        .collect();

    for p in &candidates {
        let (v, truncated) =
            judge_bubble::<IntT>(ev, g, p, dict, k1, flank_budget, min_evidence, max_nodes);
        stats.record(v, truncated);
    }
}

/// Counts of what the oracle did. Every bubble candidate lands in exactly one verdict bucket.
#[derive(Debug, Clone, Default)]
pub struct MultiKStats {
    /// Correction rounds run.
    pub rounds: usize,
    /// Bubble candidates examined.
    pub bubbles_seen: usize,

    /// One branch absent at k2 → popped on evidence.
    pub resolved_error: usize,
    /// Both branches real **and** k2 does not fork → the locus can be resolved. *The headline number.*
    pub resolvable_repeat: usize,
    /// Both branches real but k2 forks too → pop neither.
    pub protected_repeat: usize,
    /// Flanks not corroborated at k2, or too little context.
    pub inconclusive_context: usize,
    /// Neither branch corroborated.
    pub inconclusive_absent: usize,
    /// Partial support on one or both.
    pub inconclusive_partial: usize,
    /// `spell_path` rejected the walk. **Must be zero** — it is the self-check on the orientation
    /// bookkeeping, and a non-zero value invalidates every other number here.
    pub inconclusive_spell: usize,

    /// A flank hit the context budget.
    pub flanks_truncated: usize,

    /// Of the `resolvable_repeat` loci, how many yielded a clean, unambiguous predecessor pairing —
    /// exactly one predecessor supported per branch, and a different one for each. **These are the ones
    /// repeat resolution can actually split.**
    pub pairing_ok: usize,
    /// Resolvable, but the pairing was ambiguous (both predecessors supported, or neither), or the
    /// junction was not a clean two-copy collapse. Never split on a guess.
    pub pairing_ambiguous: usize,
    /// Splits actually performed.
    pub split_applied: usize,

    // ── why a planned split did not happen ───────────────────────────────────
    //
    // These five replace a single `split_skipped_stale` counter, which conflated two different
    // failures and so could not be acted on. `stale_*` mean the neighbourhood moved between judging
    // and splitting; `surgery_failed_*` mean it did not, but an expected edge was missing.
    /// `bubble_parts` no longer sees a bubble here — usually `shrink` absorbed the junction.
    pub stale_no_bubble: usize,
    /// Still a bubble, but with different mid nodes than were judged.
    pub stale_mids_changed: usize,
    /// An expected edge in the left shared chain was absent.
    pub surgery_failed_left: usize,
    /// An expected edge in the right shared chain was absent.
    pub surgery_failed_right: usize,
    /// The branch's edge to the bubble end was absent.
    pub surgery_failed_mid: usize,

    // ── fate of the previous round's skipped loci ────────────────────────────
    //
    // The retry loop in `graph_works` re-judges everything after each round with splits, so the
    // question is not "were they retried" (they were) but "what happened when they were". Without
    // this, a locus that silently stops being a candidate is indistinguishable from one that is
    // retried and rejected.
    /// Previously-skipped starts that reappeared as bubble candidates this round.
    pub revisit_seen: usize,
    /// …of those, how many still judged `ResolvableRepeat`.
    pub revisit_resolvable: usize,
    /// …of those, how many also paired unambiguously, i.e. were genuinely splittable this round.
    pub revisit_paired: usize,
    /// Previously-skipped starts that did not reappear as candidates at all.
    pub revisit_absent: usize,
}

impl MultiKStats {
    /// Total planned splits that did not happen, however they failed.
    pub fn split_skipped(&self) -> usize {
        self.stale_no_bubble
            + self.stale_mids_changed
            + self.surgery_failed_left
            + self.surgery_failed_right
            + self.surgery_failed_mid
    }

    /// Record one failed split against its reason.
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
}

impl MultiKStats {
    /// Fold one round's counts into a running total.
    pub fn accumulate(&mut self, r: MultiKStats) {
        self.bubbles_seen += r.bubbles_seen;
        self.resolved_error += r.resolved_error;
        self.resolvable_repeat += r.resolvable_repeat;
        self.protected_repeat += r.protected_repeat;
        self.inconclusive_context += r.inconclusive_context;
        self.inconclusive_absent += r.inconclusive_absent;
        self.inconclusive_partial += r.inconclusive_partial;
        self.inconclusive_spell += r.inconclusive_spell;
        self.flanks_truncated += r.flanks_truncated;
        self.pairing_ok += r.pairing_ok;
        self.pairing_ambiguous += r.pairing_ambiguous;
        self.split_applied += r.split_applied;
        self.stale_no_bubble += r.stale_no_bubble;
        self.stale_mids_changed += r.stale_mids_changed;
        self.surgery_failed_left += r.surgery_failed_left;
        self.surgery_failed_right += r.surgery_failed_right;
        self.surgery_failed_mid += r.surgery_failed_mid;
        self.revisit_seen += r.revisit_seen;
        self.revisit_resolvable += r.revisit_resolvable;
        self.revisit_paired += r.revisit_paired;
        self.revisit_absent += r.revisit_absent;
    }

    /// Record one verdict.
    pub fn record(&mut self, v: Verdict, truncated: bool) {
        self.bubbles_seen += 1;
        if truncated {
            self.flanks_truncated += 1;
        }
        match v {
            Verdict::ResolvedError(_) => self.resolved_error += 1,
            Verdict::ResolvableRepeat => self.resolvable_repeat += 1,
            Verdict::ProtectedRepeat => self.protected_repeat += 1,
            Verdict::InconclusiveContext => self.inconclusive_context += 1,
            Verdict::InconclusiveAbsent => self.inconclusive_absent += 1,
            Verdict::InconclusivePartial => self.inconclusive_partial += 1,
            Verdict::InconclusiveSpell => self.inconclusive_spell += 1,
        }
    }

    /// The six verdict buckets must account for every candidate; nothing may be silently lost.
    pub fn check_accounting(&self) {
        let sum = self.resolved_error
            + self.resolvable_repeat
            + self.protected_repeat
            + self.inconclusive_context
            + self.inconclusive_absent
            + self.inconclusive_partial
            + self.inconclusive_spell;
        assert_eq!(
            sum, self.bubbles_seen,
            "multi-k verdict accounting does not balance: {sum} verdicts for {} bubbles",
            self.bubbles_seen
        );
    }

    /// One TSV row of counters, so parameter sweeps are scriptable without scraping the log.
    pub fn tsv_header() -> &'static str {
        "k\trounds\tbubbles_seen\tresolved_error\tresolvable_repeat\tprotected_repeat\t\
         inconclusive_context\tinconclusive_absent\tinconclusive_partial\tinconclusive_spell\t\
         pairing_ok\tpairing_ambiguous\tsplit_applied\tsplit_skipped\t\
         stale_no_bubble\tstale_mids_changed\tsurgery_failed_left\tsurgery_failed_right\t\
         surgery_failed_mid\trevisit_seen\trevisit_resolvable\trevisit_paired\trevisit_absent\t\
         flanks_truncated"
    }

    /// `label` is the evidence k, or `TOTAL`.
    pub fn tsv_row(&self, label: &str) -> String {
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            label,
            self.rounds,
            self.bubbles_seen,
            self.resolved_error,
            self.resolvable_repeat,
            self.protected_repeat,
            self.inconclusive_context,
            self.inconclusive_absent,
            self.inconclusive_partial,
            self.inconclusive_spell,
            self.pairing_ok,
            self.pairing_ambiguous,
            self.split_applied,
            self.split_skipped(),
            self.stale_no_bubble,
            self.stale_mids_changed,
            self.surgery_failed_left,
            self.surgery_failed_right,
            self.surgery_failed_mid,
            self.revisit_seen,
            self.revisit_resolvable,
            self.revisit_paired,
            self.revisit_absent,
            self.flanks_truncated,
        )
    }

    /// Report at `info`. This is the number that says whether the feature is worth anything, so it is
    /// never hidden behind `-v` or a `debug!`.
    pub fn report(&self, k1: usize, k2: usize) {
        self.check_accounting();
        let n = self.bubbles_seen;
        let pc = |x: usize| if n == 0 { 0.0 } else { 100.0 * x as f64 / n as f64 };

        log::info!("Multi-k correction summary (k1={k1}, k2={k2}), {} rounds:", self.rounds);
        log::info!("  bubbles examined                   {n:5}");
        log::info!("    resolved as error (popped)       {:5}  {:5.1}%", self.resolved_error, pc(self.resolved_error));
        log::info!("    RESOLVABLE repeat (k2 spans it)  {:5}  {:5.1}%", self.resolvable_repeat, pc(self.resolvable_repeat));
        log::info!("    protected repeat (k2 forks too)  {:5}  {:5.1}%", self.protected_repeat, pc(self.protected_repeat));
        log::info!("    inconclusive: no context         {:5}  {:5.1}%", self.inconclusive_context, pc(self.inconclusive_context));
        log::info!("    inconclusive: both absent        {:5}  {:5.1}%", self.inconclusive_absent, pc(self.inconclusive_absent));
        log::info!("    inconclusive: partial            {:5}  {:5.1}%", self.inconclusive_partial, pc(self.inconclusive_partial));
        log::info!("    inconclusive: SPELL FAILED       {:5}  {:5.1}%  <- must be 0", self.inconclusive_spell, pc(self.inconclusive_spell));
        log::info!("  of the resolvable, pairing is:");
        log::info!("    UNAMBIGUOUS (can be split)       {:5}", self.pairing_ok);
        log::info!("    ambiguous (will not guess)       {:5}", self.pairing_ambiguous);
        log::info!("  repeats SPLIT (contigs run through) {:5}", self.split_applied);
        if self.split_skipped() > 0 {
            // Broken out by cause: `stale_*` means the neighbourhood moved between judging and
            // splitting, `surgery_*` means it did not but an expected edge was missing. The single
            // counter these replaced could not distinguish the two, which is why it was never actionable.
            log::info!("    NOT split, by cause              {:5}", self.split_skipped());
            log::info!("      stale: no bubble any more      {:5}", self.stale_no_bubble);
            log::info!("      stale: mid nodes changed       {:5}", self.stale_mids_changed);
            log::info!("      surgery: left chain edge gone  {:5}", self.surgery_failed_left);
            log::info!("      surgery: right chain edge gone {:5}", self.surgery_failed_right);
            log::info!("      surgery: mid->end edge gone    {:5}", self.surgery_failed_mid);
        }
        if self.revisit_seen + self.revisit_absent > 0 {
            log::info!("  fate of previously-skipped loci:");
            log::info!("    came back as a candidate         {:5}", self.revisit_seen);
            log::info!("      still RESOLVABLE               {:5}", self.revisit_resolvable);
            log::info!("      and paired (splittable again)  {:5}", self.revisit_paired);
            log::info!("    never came back                  {:5}", self.revisit_absent);
        }
        log::info!("  flanks truncated at budget         {:5}", self.flanks_truncated);

        if self.inconclusive_spell > 0 {
            log::warn!(
                "{} bubble walks failed to spell. A bubble taken from the graph is a walk by \
                 construction, so this is a bug in the orientation bookkeeping — every other number \
                 above is suspect.",
                self.inconclusive_spell
            );
        }
        if self.resolved_error == 0 && self.resolvable_repeat == 0 && self.protected_repeat == 0 {
            log::warn!(
                "The evidence graph decided nothing: multi-k is a no-op on this data. Check \
                 --multik-flank-context and the k2 min_count fit before drawing conclusions."
            );
        }
    }
}
