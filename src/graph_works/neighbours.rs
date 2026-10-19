//! Discover predecessor and successor k-mers and construct their indexed edges.

use super::indexed_kmers::{IndexMap, IndexedKmers};
use crate::utils::{bit_encoding::rc_base, nthash};
use rayon::prelude::*;
use sparrowhawk_graph::{EdgeType, IndexedEdge};

/// ntHash's split-rotate tables for `base` at `k`, ORed: the term that drops or adds the far base.
#[inline]
fn ms_tab(base: u8, k: usize) -> u64 {
    let b = base as usize;
    nthash::MS_TAB_5LL[b * 5 + k % 5]
        | nthash::MS_TAB_7L[b * 7 + k % 7]
        | nthash::MS_TAB_9LC[b * 9 + k % 9]
        | nthash::MS_TAB_11CR[b * 11 + k % 11]
        | nthash::MS_TAB_13R[b * 13 + k % 13]
        | nthash::MS_TAB_19RR[b * 19 + k % 19]
}

/// A k-mer's eight candidate predecessor hashes, two per base (from the canonical strand, then from
/// the reverse complement), each with its edge type for a canonical and for a reverse-complement hit.
#[inline]
fn bkg_candidates(hc: u64, hnc: u64, k: usize, bases: u8) -> [(u64, EdgeType, EdgeType); 8] {
    let (canonical_base, reverse_base) = (bases & 3, (bases >> 2) & 3);
    let mut out = [(0, EdgeType::MinToMin, EdgeType::MinToMin); 8];
    for i in 0..4u8 {
        let canonical = nthash::swapbits_18_31_42_51_58_63(
            (hc ^ nthash::HASH_LOOKUP[canonical_base as usize] ^ ms_tab(i, k)).rotate_right(1),
        );
        let reverse = nthash::swapbits_18_31_42_51_58_63(
            (hnc ^ ms_tab(rc_base(i), k) ^ nthash::RC_HASH_LOOKUP[reverse_base as usize])
                .rotate_right(1),
        );
        out[2 * i as usize] = (canonical, EdgeType::MinToMin, EdgeType::MaxToMin);
        out[2 * i as usize + 1] = (reverse, EdgeType::MinToMax, EdgeType::MaxToMax);
    }
    out
}

/// The same for a k-mer's eight candidate successors.
#[inline]
fn fwd_candidates(hc: u64, hnc: u64, k: usize, bases: u8) -> [(u64, EdgeType, EdgeType); 8] {
    let (canonical_base, reverse_base) = ((bases >> 2) & 3, bases & 3);
    let mut out = [(0, EdgeType::MinToMin, EdgeType::MinToMin); 8];
    for i in 0..4u8 {
        let canonical = nthash::swapbits_0_19_32_43_52_59(hc.rotate_left(1))
            ^ nthash::HASH_LOOKUP[i as usize]
            ^ ms_tab(canonical_base, k);
        let reverse = nthash::swapbits_0_19_32_43_52_59(hnc.rotate_left(1))
            ^ nthash::RC_HASH_LOOKUP[i as usize]
            ^ ms_tab(rc_base(reverse_base), k);
        out[2 * i as usize] = (canonical, EdgeType::MinToMin, EdgeType::MinToMax);
        out[2 * i as usize + 1] = (reverse, EdgeType::MaxToMin, EdgeType::MaxToMax);
    }
    out
}

fn indexed_edge(
    hash: u64,
    lookup: &IndexMap,
    canonical_type: EdgeType,
    reverse_type: EdgeType,
) -> Option<IndexedEdge> {
    lookup.get(&hash).copied().map(|entry| IndexedEdge {
        target: entry.index(),
        edge_type: if entry.is_reverse() {
            reverse_type
        } else {
            canonical_type
        },
    })
}

fn check_bkg_indexed(
    hc: u64,
    hnc: u64,
    k: usize,
    bases: u8,
    lookup: &IndexMap,
) -> Vec<IndexedEdge> {
    bkg_candidates(hc, hnc, k, bases)
        .into_iter()
        .filter_map(|(hash, canonical, reverse)| indexed_edge(hash, lookup, canonical, reverse))
        .collect()
}

fn check_fwd_indexed(
    hc: u64,
    hnc: u64,
    k: usize,
    bases: u8,
    lookup: &IndexMap,
) -> Vec<IndexedEdge> {
    fwd_candidates(hc, hnc, k, bases)
        .into_iter()
        .filter_map(|(hash, canonical, reverse)| indexed_edge(hash, lookup, canonical, reverse))
        .collect()
}

pub(crate) fn populate_indexed_neighbours<IntT>(
    k: usize,
    kmers: &mut IndexedKmers<IntT>,
) -> (usize, usize, usize) {
    kmers.initialise_neighbour_slots();

    let canonical_hashes = &kmers.canonical_hashes;
    let reverse_hashes = &kmers.reverse_hashes;
    let boundary_bases = &kmers.boundary_bases;
    let lookup = &kmers.hash_to_index;
    let neighbours = &mut kmers.neighbours;
    let predecessor_counts = &mut kmers.predecessor_counts;

    let (alone, directed_edge_refs) = neighbours
        .par_iter_mut()
        .zip(predecessor_counts.par_iter_mut())
        .enumerate()
        .map(|(index, (slot, predecessor_count))| {
            let mut edges = check_bkg_indexed(
                canonical_hashes[index],
                reverse_hashes[index],
                k,
                boundary_bases[index],
                lookup,
            );
            *predecessor_count =
                u8::try_from(edges.len()).expect("too many predecessor edges for one k-mer");
            edges.extend(check_fwd_indexed(
                canonical_hashes[index],
                reverse_hashes[index],
                k,
                boundary_bases[index],
                lookup,
            ));
            *slot = edges;
            (usize::from(slot.is_empty()), slot.len())
        })
        .reduce(
            || (0, 0),
            |left, right| (left.0 + right.0, left.1 + right.1),
        );

    (canonical_hashes.len(), alone, directed_edge_refs)
}
