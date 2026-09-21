//! Native aligned storage used between k-mer counting and graph construction.

use nohash_hasher::NoHashHasher;
use sparrowhawk_graph::IndexedEdge;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::BuildHasherDefault;

type IndexMap = HashMap<u64, OrientedKmerIndex, BuildHasherDefault<NoHashHasher<u64>>>;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OrientedKmerIndex(usize);

impl OrientedKmerIndex {
    fn new(index: usize, is_reverse: bool) -> Self {
        let encoded = index
            .checked_mul(2)
            .and_then(|value| value.checked_add(usize::from(is_reverse)))
            .expect("too many indexed k-mers to encode their orientation");
        Self(encoded)
    }

    fn canonical(index: usize) -> Self {
        Self::new(index, false)
    }

    fn reverse(index: usize) -> Self {
        Self::new(index, true)
    }

    pub(crate) fn index(self) -> usize {
        self.0 / 2
    }

    pub(crate) fn is_reverse(self) -> bool {
        self.0 & 1 != 0
    }
}

/// K-mer attributes stored once in vectors sharing a stable index.
#[derive(Clone)]
pub struct IndexedKmers<IntT> {
    pub(crate) canonical_hashes: Vec<u64>,
    pub(crate) reverse_hashes: Vec<u64>,
    pub(crate) boundary_bases: Vec<u8>,
    pub(crate) counts: Vec<u32>,
    pub(crate) packed_kmers: Vec<IntT>,
    pub(crate) hash_to_index: IndexMap,
    pub(crate) neighbours: Vec<Vec<IndexedEdge>>,
    pub(crate) predecessor_counts: Vec<u8>,
}

impl<IntT> IndexedKmers<IntT> {
    pub(crate) fn with_capacity(kmers: usize) -> Self {
        let lookup_capacity = kmers
            .checked_mul(2)
            .expect("too many indexed k-mers to allocate their hash lookup");
        Self {
            canonical_hashes: Vec::with_capacity(kmers),
            reverse_hashes: Vec::with_capacity(kmers),
            boundary_bases: Vec::with_capacity(kmers),
            counts: Vec::with_capacity(kmers),
            packed_kmers: Vec::with_capacity(kmers),
            hash_to_index: HashMap::with_capacity_and_hasher(
                lookup_capacity,
                BuildHasherDefault::default(),
            ),
            neighbours: Vec::new(),
            predecessor_counts: Vec::new(),
        }
    }

    pub(crate) fn push(
        &mut self,
        canonical_hash: u64,
        reverse_hash: u64,
        bases: u8,
        count: u32,
        packed: IntT,
    ) {
        let index = self.canonical_hashes.len();
        self.canonical_hashes.push(canonical_hash);
        self.reverse_hashes.push(reverse_hash);
        self.boundary_bases.push(bases);
        self.counts.push(count);
        self.packed_kmers.push(packed);

        // Canonical hits have the same priority as the old `themap.contains_key` check. A later
        // canonical key therefore replaces a colliding reverse key, while reverse keys never replace
        // an existing entry.
        self.hash_to_index
            .insert(canonical_hash, OrientedKmerIndex::canonical(index));
        if let Entry::Vacant(entry) = self.hash_to_index.entry(reverse_hash) {
            entry.insert(OrientedKmerIndex::reverse(index));
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.canonical_hashes.len()
    }

    #[cfg(test)]
    pub(crate) fn lookup_hash(&self, hash: u64) -> Option<(usize, bool)> {
        self.hash_to_index
            .get(&hash)
            .copied()
            .map(|entry| (entry.index(), entry.is_reverse()))
    }

    pub(crate) fn initialise_neighbour_slots(&mut self) {
        let len = self.len();
        self.neighbours.clear();
        self.neighbours.resize_with(len, Vec::new);
        self.predecessor_counts.clear();
        self.predecessor_counts.resize(len, 0);
    }

    pub(crate) fn finish_neighbour_search(&mut self) {
        self.reverse_hashes = Vec::new();
        self.boundary_bases = Vec::new();
        self.hash_to_index.retain(|_, value| !value.is_reverse());
        self.hash_to_index.shrink_to_fit();
    }

    pub(crate) fn take_graph_inputs(
        &mut self,
    ) -> (Vec<u64>, Vec<u32>, Vec<Vec<IndexedEdge>>, Vec<u8>) {
        (
            std::mem::take(&mut self.canonical_hashes),
            std::mem::take(&mut self.counts),
            std::mem::take(&mut self.neighbours),
            std::mem::take(&mut self.predecessor_counts),
        )
    }

    pub(crate) fn get_packed(&self, hash: u64) -> Option<&IntT> {
        let entry = self.hash_to_index.get(&hash)?;
        self.packed_kmers.get(entry.index())
    }
}

impl<IntT> Default for IndexedKmers<IntT> {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_fields_and_orientations_share_one_index() {
        let mut kmers = IndexedKmers::with_capacity(1);
        kmers.push(11, 19, 3, 7, 23_u64);

        assert_eq!(kmers.lookup_hash(11), Some((0, false)));
        assert_eq!(kmers.lookup_hash(19), Some((0, true)));
        assert_eq!(kmers.canonical_hashes, vec![11]);
        assert_eq!(kmers.reverse_hashes, vec![19]);
        assert_eq!(kmers.boundary_bases, vec![3]);
        assert_eq!(kmers.counts, vec![7]);
        assert_eq!(kmers.packed_kmers, vec![23]);
    }

    #[test]
    fn canonical_hash_wins_a_cross_orientation_collision() {
        let mut kmers = IndexedKmers::with_capacity(2);
        kmers.push(11, 19, 0, 1, 1_u64);
        kmers.push(19, 29, 0, 1, 2_u64);

        assert_eq!(kmers.lookup_hash(19), Some((1, false)));
    }

    #[test]
    fn finishing_discards_graph_metadata_and_reverse_lookups() {
        let mut kmers = IndexedKmers::with_capacity(1);
        kmers.push(11, 19, 3, 7, 23_u64);
        kmers.initialise_neighbour_slots();
        kmers.finish_neighbour_search();

        assert_eq!(kmers.lookup_hash(11), Some((0, false)));
        assert_eq!(kmers.lookup_hash(19), None);
        assert!(kmers.reverse_hashes.is_empty());
        assert!(kmers.boundary_bases.is_empty());

        let (hashes, counts, neighbours, predecessor_counts) = kmers.take_graph_inputs();
        assert_eq!(hashes, vec![11]);
        assert_eq!(counts, vec![7]);
        assert_eq!(neighbours, vec![Vec::new()]);
        assert_eq!(predecessor_counts, vec![0]);
        assert_eq!(kmers.get_packed(11), Some(&23));
    }
}
