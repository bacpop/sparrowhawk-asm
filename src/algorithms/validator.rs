//! Graph-invariant audits used for diagnosing construction/shrink corruption.
use sparrowhawk_graph::DbgGraph;

/// Every directed edge u→v (type t) must have a partner v→u (type t.rev()),
/// and consequently in_degree == out_degree at every node. Any violation means
/// the graph was built from asymmetric neighbour lists (or corrupted afterwards),
/// which is what the shrinker's degree assertions later trip over.
/// Returns the violation count.
pub fn audit_bidirected_pairing(g: &DbgGraph) -> usize {
    let mut violations = 0usize;
    for n in g.node_indices() {
        let (ind, outd) = (g.in_degree(n), g.out_degree(n));
        if ind != outd {
            violations += 1;
            if violations <= 50 {
                log::error!(
                    "[audit] degree imbalance at {n:?}: in {ind} != out {outd}. kmers {:?}",
                    g.node_weight(n).map(|w| &w.abs_ind),
                );
            }
        }
        for (m, t) in g.outgoing_edges(n) {
            if !g
                .outgoing_edges(m)
                .iter()
                .any(|&(back, bt)| back == n && bt == t.rev())
            {
                violations += 1;
                if violations <= 50 {
                    log::error!(
                        "[audit] unpaired edge {n:?} -{t:?}-> {m:?}: no {m:?} -{:?}-> {n:?} partner. \
                         kmers {:?} / {:?}",
                        t.rev(),
                        g.node_weight(n).map(|w| &w.abs_ind),
                        g.node_weight(m).map(|w| &w.abs_ind),
                    );
                }
            }
        }
    }
    if violations == 0 {
        log::info!("[audit] bidirected pairing holds on the built graph");
    } else {
        log::error!("[audit] {violations} pairing violations in the built graph");
    }
    violations
}
