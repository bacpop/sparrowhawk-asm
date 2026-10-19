//! Batched input parsing, sharded k-mer counting, and carried-contig counting.

use super::bloom_filter::BloomBits;
use super::kmer::{Bases, Kmer};
use super::read_store::{PackedSeq, PackedSeqs, ReadStore, BLOCK_READS};
use super::sketch::SpectrumSketch;
use crate::graph_works::indexed_kmers::IndexedKmers;
use crate::utils::{
    bit_encoding::UInt,
    clock::Instant,
    qual_profile::{window_classes, window_groups, MAX_GROUPS, NONE},
    reads::{MaybeSend, OwnedRecord},
};
use crate::QualOpts;
use nohash_hasher::NoHashHasher;
use rayon::prelude::*;
use std::{borrow::Cow, collections::HashMap, hash::BuildHasherDefault};

/// Never fewer shards than this: below it each map stops being a small fraction of the working set. Heuristically set.
const MIN_COUNTMAP_SHARDS: usize = 64;

/// Nor more than this, or the per-batch split pays for buckets that hold almost nothing.
const MAX_COUNTMAP_SHARDS: usize = 256;

/// Shards per thread. One, not more: oversubscribing to give rayon work to steal was measured and is a
/// bad trade, since every extra shard is another half-empty table. At k=81 on 12 threads, 64 shards
/// cost 17% peak RSS to save 4% wall against 16.
const SHARDS_PER_THREAD: usize = 1;

/// Bloom filter occupancy (%) from which a warning is shown.
const BLOOM_WARN_OCCUPANCY_PCT: f64 = 25.0;

/// Contig bases hashed per round when carrying contigs in; bounds the transient k-mer buffer.
const CARRY_CHUNK_BASES: usize = 1 << 20;

/// Parse `files` into owned batches of records, handing each to `on_batch`, and return how long the
/// whole walk took. A scoped producer fills the next batch while `on_batch` works on the previous one,
/// so the parse overlaps the compute instead of alternating with it.
///
/// Two buffers circulate on a one-deep queue, which bounds memory and cannot deadlock: the consumer
/// returns a buffer after every batch, so the producer's `recv` is always eventually satisfied.
/// One producer and one FIFO means records arrive in exactly the order the serial version used.
#[cfg(not(target_family = "wasm"))]
fn extract_kmers_from_files_batched<F, I>(
    input_iters: &mut [I],
    batch_records: usize,
    mut on_batch: F,
) -> std::time::Duration
where
    F: FnMut(&[OwnedRecord]),
    I: Iterator<Item = (Vec<u8>, Option<Vec<u8>>)> + Send,
{
    let (full_tx, full_rx) = std::sync::mpsc::sync_channel::<Vec<OwnedRecord>>(1);
    let (empty_tx, empty_rx) = std::sync::mpsc::sync_channel::<Vec<OwnedRecord>>(2);
    for _ in 0..2 {
        let _ = empty_tx.send(Vec::with_capacity(batch_records));
    }
    let nfiles = input_iters.len();
    let t0 = Instant::now();
    std::thread::scope(|scope| {
        scope.spawn(move || {
            let mut batch = empty_rx.recv().unwrap_or_default();
            for (idx, records) in input_iters.iter_mut().enumerate() {
                log::info!("Getting kmers from file number {idx}.");
                for record in records {
                    let seq: Vec<u8> = record.0;
                    let qual: Option<Vec<u8>> = record.1;
                    batch.push((seq, qual));
                    if batch.len() == batch_records {
                        // A closed channel means the consumer is gone; there is nothing left to feed.
                        if full_tx.send(batch).is_err() {
                            return;
                        }
                        batch = empty_rx
                            .recv()
                            .unwrap_or_else(|_| Vec::with_capacity(batch_records));
                    }
                }
                log::info!("Finished getting kmers from file number {idx}.");
            }
            if !batch.is_empty() {
                let _ = full_tx.send(batch);
            }
        });
        // The producer holds the only `full_tx`, so this loop ends when the producer does.
        for mut batch in full_rx {
            on_batch(&batch);
            batch.clear();
            let _ = empty_tx.send(batch);
        }
    });
    log::info!("Finished getting kmers from {nfiles} file(s)");
    t0.elapsed()
}

/// The browser's version: no threads, so parse a batch, count it, and tell the page how far it got.
#[cfg(target_family = "wasm")]
fn extract_kmers_from_files_batched<F, I>(
    input_iters: &mut [I],
    batch_records: usize,
    mut on_batch: F,
) -> std::time::Duration
where
    F: FnMut(&[OwnedRecord]),
    I: Iterator<Item = OwnedRecord>,
{
    let t0 = Instant::now();
    let mut batch = Vec::with_capacity(batch_records);
    let mut done = 0usize;
    for records in input_iters.iter_mut() {
        for record in records {
            batch.push(record);
            if batch.len() == batch_records {
                on_batch(&batch);
                done += batch.len();
                batch.clear();
                crate::post_state(&match crate::utils::reads::pass_percent() {
                    Some(percent) => format!("preprocess:bulk:loop:{done}:{percent}"),
                    None => format!("preprocess:bulk:loop:{done}"),
                });
            }
        }
    }
    if !batch.is_empty() {
        on_batch(&batch);
    }
    t0.elapsed()
}

/// The constants of one counting pass: fixed for every record, so they travel together.
#[derive(Clone, Copy)]
pub(super) struct KmerWalk<'a> {
    pub(super) k: usize,
    pub(super) min_qual: u8,
    pub(super) min_class: u8,
    pub(super) floors: Option<&'a [u8]>,
    /// Group index of the strict floor: a k-mer tagged below this does not enter the count-map.
    pub(super) keep: u8,
}

/// What one counting pass reads.
pub enum PassInput<'a, I> {
    /// Reads parsed from the input, as now: the first k (packing them on the way), a native single k,
    /// and every pass in the browser.
    Text(I),
    /// The reads the first k packed, parsed where they lie: every later k of a native ladder, and a
    /// recount. No reader thread, no decoding.
    Stored(&'a ReadStore),
}

/// A sequence the counter walks: a read parsed from the input, or a packed one.
pub(super) trait CountedRead: Sync {
    /// Hand each of its k-mers to `emit` as `(hc, hnc, b, group, packed)`. `groups` is the caller's
    /// scratch, reused per sequence.
    fn walk<IntT, F>(&self, w: KmerWalk, groups: &mut Vec<u8>, emit: F)
    where
        IntT: for<'a> UInt<'a>,
        F: FnMut(u64, u64, u8, u8, Option<IntT>);
}

impl CountedRead for OwnedRecord {
    /// Classify ASCII windows and filter the iterator at the lowest active floor.
    #[inline]
    fn walk<IntT, F>(&self, w: KmerWalk, groups: &mut Vec<u8>, emit: F)
    where
        IntT: for<'a> UInt<'a>,
        F: FnMut(u64, u64, u8, u8, Option<IntT>),
    {
        let (seq, qual) = (self.0.as_slice(), self.1.as_deref());
        // The scratch is shared across records, so a FASTA record must not inherit the previous one's tags.
        groups.clear();
        if let (Some(floors), Some(qual)) = (w.floors, qual) {
            window_groups(seq, qual, w.k, floors, groups);
        }
        // Under a ladder the iterator filters at its lowest floor (0 unless the floor was fixed), so every
        // window it emits clears at least one rung.
        let min_qual = match w.floors {
            Some(floors) if !groups.is_empty() => floors[0],
            _ => w.min_qual,
        };
        walk_kmers(
            Kmer::<IntT>::new(Cow::Borrowed(seq), seq.len(), qual, w.k, min_qual, true),
            groups,
            w.keep,
            emit,
        );
    }
}

impl CountedRead for PackedSeq<'_> {
    #[inline]
    fn walk<IntT, F>(&self, w: KmerWalk, groups: &mut Vec<u8>, emit: F)
    where
        IntT: for<'a> UInt<'a>,
        F: FnMut(u64, u64, u8, u8, Option<IntT>),
    {
        groups.clear();
        if self.is_two_bit() {
            // Reads kept in 2 bits (one floor), or a contig: every stored base clears the floor, so every
            // k-mer is `keep`.
            walk_kmers(
                Kmer::<IntT, _>::from_two_bit(*self, w.k, true),
                groups,
                w.keep,
                emit,
            );
        } else {
            if let Some(floors) = w.floors {
                window_classes(*self, w.k, (floors.len() - 1) as u8, groups);
            }
            walk_kmers(
                Kmer::<IntT, _>::from_four_bit(*self, w.k, w.min_class, true),
                groups,
                w.keep,
                emit,
            );
        }
    }
}

/// Emit each k-mer with its tag: from `groups` under a ladder, `keep` otherwise. The packed k-mer is the
/// costly half at k>=51, so it is built only for what is kept.
#[inline]
fn walk_kmers<IntT, B, F>(kmers: Option<Kmer<'_, IntT, B>>, groups: &[u8], keep: u8, mut emit: F)
where
    IntT: for<'a> UInt<'a>,
    B: Bases,
    F: FnMut(u64, u64, u8, u8, Option<IntT>),
{
    let Some(mut kmer_it) = kmers else { return };
    let (mut hc, mut hnc, mut b) = kmer_it.get_hashes_and_bases();
    loop {
        let g = if groups.is_empty() {
            keep
        } else {
            groups[kmer_it.end_index()]
        };
        debug_assert_ne!(g, NONE, "the iterator emitted a window no floor clears");
        emit(hc, hnc, b, g, (g >= keep).then(|| kmer_it.get_kmer()));
        match kmer_it.get_next_hash_and_bases() {
            Some(next) => (hc, hnc, b) = next,
            None => break,
        }
    }
}

/// One distinct k-mer, accumulated while counting in bulk.
///
/// `hnc`, `b` and `km` describe the *k-mer*, not the *occurrence*, so they are recorded once, on first
/// sight, rather than once per occurrence as the old sorting vector did.
pub(super) struct KmerInfo<IntT> {
    pub(super) count: u32,
    pub(super) hnc: u64,
    pub(super) b: u8,
    pub(super) km: IntT,
}

pub(super) type CountMap<IntT> =
    HashMap<u64, KmerInfo<IntT>, BuildHasherDefault<NoHashHasher<u64>>>;

/// Where the counting phase spends its time, accumulated over every batch. It is bandwidth-bound
/// rather than synchronisation-bound, so wall time alone misattributes it, and a few thousand clock
/// reads per run cost nothing against that.
#[derive(Default)]
struct CountTimings {
    hash_ns: u128,
    sketch_ns: u128,
    split_ns: u128,
    absorb_ns: u128,
    batches: usize,
}

impl CountTimings {
    /// `total` is the whole batched walk, so parse is whatever the timed stages did not account for.
    fn log(&self, total: std::time::Duration) {
        let inner = self.hash_ns + self.sketch_ns + self.split_ns + self.absorb_ns;
        let parse = total.as_nanos().saturating_sub(inner);
        let pct = |n: u128| 100.0 * n as f64 / total.as_nanos().max(1) as f64;
        log::info!(
            "Counting phase: {} batches in {:.1} s | parse {:.1}% serial | hash {:.1}% parallel | \
             sketch {:.1}% serial | split {:.1}% serial | absorb {:.1}% parallel",
            self.batches,
            total.as_secs_f64(),
            pct(parse),
            pct(self.hash_ns),
            pct(self.sketch_ns),
            pct(self.split_ns),
            pct(self.absorb_ns),
        );
    }
}

/// Pick a shard for a canonical hash, mixing first. `shards` must be a power of two.
#[inline(always)]
fn shard_of(hc: u64, shards: usize) -> usize {
    const MIX: u64 = 0x9E37_79B9_7F4A_7C15; // odd, golden-ratio derived
    debug_assert!(
        shards.is_power_of_two(),
        "shard count must be a power of two"
    );
    ((hc.wrapping_mul(MIX) >> (64 - shards.trailing_zeros())) as usize) & (shards - 1)
}

/// Kept k-mers waiting to be absorbed into one shard: hash, non-canonical hash, packed bases, k-mer.
type Bucket<IntT> = Vec<(u64, u64, u8, IntT)>;

/// One rayon task's private scratch, allocated once and reused across every batch.
struct TaskState<IntT> {
    /// Kept k-mers, already split by shard. Absorb drains these, so the capacity carries over.
    buckets: Vec<Bucket<IntT>>,
    /// `window_groups` scratch, reused for every record the task sees.
    groups: Vec<u8>,
    /// Sketch candidates, in record order.
    cand: Vec<(u64, u8)>,
    /// Exact presence of at least one observed k-mer window at each quality-floor rung.
    floor_has_kmers: [bool; MAX_GROUPS],
}

impl<IntT> TaskState<IntT> {
    fn new(n_shards: usize) -> Self {
        Self {
            buckets: (0..n_shards).map(|_| Vec::new()).collect(),
            groups: Vec::new(),
            cand: Vec::new(),
            floor_has_kmers: [false; MAX_GROUPS],
        }
    }
}

/// Hash one batch straight into the tasks' per-shard buckets, then absorb those into the count-map.
/// Nothing is materialised in between: a k-mer is written once into a bucket and once into the map,
/// where the flat-vector version wrote it four times. Shards partition the keys, so absorb needs no lock.
fn count_batch<IntT, R: CountedRead>(
    batch: &[R],
    w: KmerWalk,
    sketch: Option<&mut SpectrumSketch>,
    states: &mut [TaskState<IntT>],
    shards: &mut [CountMap<IntT>],
    blooms: &mut [Option<BloomBits>],
    t: &mut CountTimings,
) where
    IntT: for<'a> UInt<'a>,
{
    let n_shards = shards.len();
    // The filter eats each k-mer's first sighting, so a surviving entry opens at 2 rather than 1 --
    // exactly what a serial Bloom filter in front of a hash map did.
    let base = if blooms.iter().any(Option::is_some) {
        2
    } else {
        1
    };
    // `observe` rejects anything at or above the threshold, and the threshold only ever falls, so
    // filtering against a snapshot taken now admits exactly what it would have accepted.
    let thr = sketch.as_ref().map_or(0, |s| s.threshold);

    let t0 = Instant::now();
    // One chunk per state, so each task owns its scratch outright and the buckets survive the batch.
    let chunk = batch.len().div_ceil(states.len().max(1)).max(1);
    batch
        .par_chunks(chunk)
        .zip(states.par_iter_mut())
        .for_each(|(recs, st)| {
            let TaskState {
                buckets,
                groups,
                cand,
                floor_has_kmers,
            } = st;
            for read in recs {
                read.walk::<IntT, _>(w, groups, |hc, hnc, b, g, km| {
                    if g != NONE {
                        for present in floor_has_kmers.iter_mut().take(usize::from(g) + 1) {
                            *present = true;
                        }
                    }
                    if hc < thr {
                        cand.push((hc, g));
                    }
                    if g >= w.keep {
                        buckets[shard_of(hc, n_shards)].push((
                            hc,
                            hnc,
                            b,
                            km.expect("a kept k-mer carries its bits"),
                        ));
                    }
                });
            }
        });
    t.hash_ns += t0.elapsed().as_nanos();

    // Chunks are contiguous and taken in order, so draining the tasks in order replays the batch in
    // record order: `observe` sees exactly the sequence the single flat pass used to hand it.
    let t1 = Instant::now();
    match sketch {
        Some(sketch) => {
            for st in states.iter_mut() {
                for (hc, g) in st.cand.drain(..) {
                    sketch.observe(hc, g);
                }
            }
        }
        None => {
            for st in states.iter_mut() {
                st.cand.clear();
            }
        }
    }
    t.sketch_ns += t1.elapsed().as_nanos();

    // Absorb wants one thread per shard, but the buckets are task-major. Gathering `&mut` references
    // regroups them shard-major without moving a single k-mer.
    let t2 = Instant::now();
    let mut by_shard: Vec<Vec<&mut Bucket<IntT>>> = (0..n_shards)
        .map(|_| Vec::with_capacity(states.len()))
        .collect();
    for st in states.iter_mut() {
        for (s, bucket) in st.buckets.iter_mut().enumerate() {
            by_shard[s].push(bucket);
        }
    }
    t.split_ns += t2.elapsed().as_nanos();

    let t3 = Instant::now();
    shards
        .par_iter_mut()
        .zip(by_shard.par_iter_mut())
        .zip(blooms.par_iter_mut())
        .for_each(|((map, cols), bloom)| {
            for col in cols.iter_mut() {
                for (hc, hnc, b, km) in col.drain(..) {
                    // A k-mer's first sighting only sets bits; counting starts on the second. The
                    // bits are sharded by the same key as the map, so this stays single-writer.
                    if let Some(bits) = bloom.as_mut() {
                        if !bits.add_and_check(hc) {
                            continue;
                        }
                    }
                    map.entry(hc)
                        .and_modify(|e| e.count = e.count.saturating_add(1))
                        .or_insert(KmerInfo {
                            count: base,
                            hnc,
                            b,
                            km,
                        });
                }
            }
        });
    t.absorb_ns += t3.elapsed().as_nanos();
}

/// One pass over the reads: every kept k-mer counted into the sharded count map, and every
/// k-mer sampled into the per-floor sketch.
pub(super) fn count_kmers<IntT, I>(
    input: PassInput<'_, I>,
    k: usize,
    qual: &QualOpts,
    floors: Option<&[u8]>,
    min_class: u8,
    do_bloom: bool,
    do_fit: bool,
) -> (
    Vec<CountMap<IntT>>,
    Option<SpectrumSketch>,
    [bool; MAX_GROUPS],
)
where
    IntT: for<'a> UInt<'a>,
    I: Iterator<Item = OwnedRecord> + MaybeSend,
{
    // The ladder is ascending, so the last group is the strict floor and `keep` is its index.
    let keep = floors.map_or(0u8, |f| (f.len() - 1) as u8);
    // The sketch only feeds the fit, except under Bloom counting, whose spectrum must come from it.
    let mut sketch = ((floors.is_some() && do_fit) || do_bloom).then(SpectrumSketch::new);
    let n_shards = (rayon::current_num_threads() * SHARDS_PER_THREAD)
        .next_power_of_two()
        .clamp(MIN_COUNTMAP_SHARDS, MAX_COUNTMAP_SHARDS);
    log::info!(
        "Counting k-mers into {n_shards} shards on {} thread(s){}",
        rayon::current_num_threads(),
        if do_bloom {
            ", behind a Bloom filter"
        } else {
            ""
        }
    );
    // One slice of the bit array per shard, so each stays single-writer and the parallel absorb needs
    // no atomics. Splitting the words keeps bits per key exactly as the unsharded filter had them.
    let mut blooms: Vec<Option<BloomBits>> = (0..n_shards)
        .map(|_| {
            do_bloom.then(|| {
                let mut bits = BloomBits::with_words(BloomBits::default_words() / n_shards as u64);
                bits.init();
                bits
            })
        })
        .collect();
    let mut shards: Vec<CountMap<IntT>> = (0..n_shards)
        .map(|_| HashMap::with_hasher(BuildHasherDefault::default()))
        .collect();
    // One scratch per thread, allocated here and reused by every batch, so the counting loop itself
    // allocates nothing.
    let mut states: Vec<TaskState<IntT>> = (0..rayon::current_num_threads().max(1))
        .map(|_| TaskState::new(n_shards))
        .collect();

    let mut t = CountTimings::default();
    crate::post_state("preprocess:bulk:loop:start");
    let w = KmerWalk {
        k,
        min_qual: qual.min_qual,
        min_class,
        floors,
        keep,
    };
    let total = match input {
        PassInput::Text(records) => {
            extract_kmers_from_files_batched(&mut [records], BLOCK_READS, |batch| {
                count_batch(
                    batch,
                    w,
                    sketch.as_mut(),
                    &mut states,
                    &mut shards,
                    &mut blooms,
                    &mut t,
                );
                t.batches += 1;
            })
        }
        PassInput::Stored(store) => {
            let start = Instant::now();
            let mut batch = Vec::new();
            for sequences in store.seqs().batches() {
                batch.clear();
                batch.extend(sequences);
                count_batch(
                    &batch,
                    w,
                    sketch.as_mut(),
                    &mut states,
                    &mut shards,
                    &mut blooms,
                    &mut t,
                );
                t.batches += 1;
            }
            start.elapsed()
        }
    };
    crate::post_state("preprocess:bulk:loop:end");
    t.log(total);

    if do_bloom {
        let (set_bits, capacity_bits) = blooms.iter().flatten().map(BloomBits::occupancy).fold(
            (0u64, 0u64),
            |(set, capacity), (shard_set, shard_capacity)| {
                (set + shard_set, capacity + shard_capacity)
            },
        );
        let occupancy_pct = if capacity_bits == 0 {
            0.0
        } else {
            100.0 * set_bits as f64 / capacity_bits as f64
        };
        log::info!(
            "Bloom filter occupancy: set_bits={set_bits} capacity_bits={capacity_bits} occupancy_pct={occupancy_pct:.3}"
        );
        if occupancy_pct > BLOOM_WARN_OCCUPANCY_PCT {
            log::warn!(
                "The Bloom filter is {occupancy_pct:.1}% full at k={k}: some k-mers seen once will be \
                 admitted at count 2. This matters only if the minimum count ends up at 2; --no-bloom \
                 counts exactly, at the cost of memory for every distinct k-mer."
            );
        }
    }

    let mut floor_has_kmers = [false; MAX_GROUPS];
    for state in &states {
        for (total, present) in floor_has_kmers.iter_mut().zip(state.floor_has_kmers) {
            *total |= present;
        }
    }

    (shards, sketch, floor_has_kmers)
}

/// Consume the count-map into aligned storage, keeping k-mers seen `minc` times.
pub(super) fn countmaps_into_indexed_kmers<IntT>(
    shards: Vec<CountMap<IntT>>,
    minc: u16,
) -> IndexedKmers<IntT>
where
    IntT: for<'a> UInt<'a>,
{
    let minc = u32::from(minc);
    let survivors = shards
        .iter()
        .flat_map(HashMap::values)
        .filter(|info| info.count >= minc)
        .count();
    let mut indexed = IndexedKmers::with_capacity(survivors);

    // Shards are consumed in order. Packed k-mers and metadata move into one aligned slot, so no
    // intermediate sequence, graph-data, or reverse-hash dictionary is materialised.
    for shard in shards {
        for (hc, info) in shard {
            if info.count >= minc {
                indexed.push(hc, info.hnc, info.b, info.count, info.km);
            }
        }
    }

    debug_assert_eq!(indexed.len(), survivors);
    indexed
}

/// Put every k-mer of the previous k's `contigs` into the count-map so the `minc` filter keeps it:
/// each count is raised to `minc`, and counts the reads already clear stay untouched. Returns how many
/// the reads never produced. Each chunk is hashed in parallel, then absorbed one writer per shard.
pub(super) fn carry_contig_kmers<IntT>(
    shards: &mut [CountMap<IntT>],
    contigs: &PackedSeqs,
    k: usize,
    minc: u16,
) -> usize
where
    IntT: for<'a> UInt<'a>,
{
    let n_shards = shards.len();
    let minc = u32::from(minc);
    let before: usize = shards.iter().map(HashMap::len).sum();
    let w = KmerWalk {
        k,
        min_qual: 0,
        min_class: 0,
        floors: None,
        keep: 0,
    };
    let seqs: Vec<_> = contigs.iter().collect();
    let mut rest = seqs.as_slice();
    while !rest.is_empty() {
        let mut take = 1;
        let mut bases = rest[0].len();
        while take < rest.len() && bases + rest[take].len() <= CARRY_CHUNK_BASES {
            bases += rest[take].len();
            take += 1;
        }
        let (chunk, tail) = rest.split_at(take);
        rest = tail;

        let mut kmers: Vec<(u64, u64, u8, IntT)> = chunk
            .par_iter()
            .flat_map_iter(|seq| {
                let mut local = Vec::new();
                let mut groups = Vec::new();
                seq.walk::<IntT, _>(w, &mut groups, |hc, hnc, b, _, km| {
                    local.push((hc, hnc, b, km.expect("a kept k-mer carries its bits")));
                });
                local
            })
            .collect();
        kmers.par_sort_unstable_by_key(|e| shard_of(e.0, n_shards));
        let mut runs: Vec<&[(u64, u64, u8, IntT)]> = Vec::with_capacity(n_shards);
        let mut from = 0;
        for s in 0..n_shards {
            let len = kmers[from..].partition_point(|e| shard_of(e.0, n_shards) == s);
            runs.push(&kmers[from..from + len]);
            from += len;
        }
        shards.par_iter_mut().zip(runs).for_each(|(map, run)| {
            for &(hc, hnc, b, km) in run {
                map.entry(hc)
                    .and_modify(|e| e.count = e.count.max(minc))
                    .or_insert(KmerInfo {
                        count: minc,
                        hnc,
                        b,
                        km,
                    });
            }
        });
    }
    shards.iter().map(HashMap::len).sum::<usize>() - before
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessing::test_support::*;

    #[test]
    fn countmaps_move_survivors_into_aligned_storage() {
        let mut shard: CountMap<u64> = HashMap::with_hasher(BuildHasherDefault::default());
        shard.insert(
            11,
            KmerInfo {
                count: 5,
                hnc: 19,
                b: 3,
                km: 23,
            },
        );
        shard.insert(
            29,
            KmerInfo {
                count: 1,
                hnc: 31,
                b: 4,
                km: 37,
            },
        );

        let indexed = countmaps_into_indexed_kmers(vec![shard], 3);

        assert_eq!(indexed.len(), 1);
        assert_eq!(indexed.canonical_hashes, vec![11]);
        assert_eq!(indexed.reverse_hashes, vec![19]);
        assert_eq!(indexed.boundary_bases, vec![3]);
        assert_eq!(indexed.counts, vec![5]);
        assert_eq!(indexed.packed_kmers, vec![23]);
        assert_eq!(indexed.lookup_hash(11), Some((0, false)));
        assert_eq!(indexed.lookup_hash(19), Some((0, true)));
    }

    /// A fixed floor is a ladder of one rung: counting under it keeps exactly what filtering at that
    /// floor keeps, and no emitted window is tagged NONE.
    #[test]
    fn a_fixed_floor_counts_only_what_clears_it() {
        let reads = counter_test_reads();
        let qual = QualOpts {
            min_count: 2,
            min_qual: 20,
        };
        let counts = |floors: Option<&[u8]>| -> HashMap<u64, u32> {
            count_text::<u64, _>(
                &mut [reads.clone().into_iter()],
                11,
                &qual,
                floors,
                false,
                true,
            )
            .0
            .into_iter()
            .flatten()
            .map(|(hc, info)| (hc, info.count))
            .collect()
        };
        assert_eq!(counts(Some(&[20])), counts(None));
    }

    /// The sharded, threaded counter must agree with the obvious serial one, and every k-mer must sit
    /// in the shard `shard_of` claims. Both sides consume the same `hash_batch` output, so this tests
    /// the sharding and the floor filtering, not the hashing.
    #[test]
    fn the_sharded_counter_agrees_with_a_serial_reference() {
        let reads = counter_test_reads();
        let ladder = [0u8, 11, 25];
        let (k, keep) = (11usize, (ladder.len() - 1) as u8);
        let qual = QualOpts {
            min_count: 2,
            min_qual: 25,
        };

        let mut iters = [reads.clone().into_iter()];
        let (shards, _, _) =
            count_text::<u64, _>(&mut iters, k, &qual, Some(&ladder[..]), false, true);

        let n_shards = shards.len();
        let mut got: HashMap<u64, (u32, u64, u8)> = HashMap::default();
        for (idx, shard) in shards.iter().enumerate() {
            for (hc, info) in shard {
                assert_eq!(
                    shard_of(*hc, n_shards),
                    idx,
                    "k-mer {hc} is in the wrong shard"
                );
                got.insert(*hc, (info.count, info.hnc, info.b));
            }
        }

        let mut want: HashMap<u64, (u32, u64, u8)> = HashMap::default();
        for (hc, hnc, b, g, _) in
            hash_batch::<u64>(&reads, k, qual.min_qual, Some(&ladder[..]), keep)
        {
            if g < keep {
                continue;
            }
            want.entry(hc).or_insert((0, hnc, b)).0 += 1;
        }

        assert!(
            !want.is_empty(),
            "the fixture produced no k-mers above the floor"
        );
        assert_eq!(got, want);
    }

    /// What the sharded filter guarantees: it eats a k-mer's first sighting, so a survivor's count is
    /// its true count, or one more when a false positive let that first sighting through.
    #[test]
    fn the_sharded_bloom_counts_what_the_filter_promises() {
        let reads = counter_test_reads();
        let ladder = [0u8, 11, 25];
        let qual = QualOpts {
            min_count: 2,
            min_qual: 25,
        };
        let want = true_counts(&reads, 11, qual.min_qual, &ladder);
        let got = bloom_shards(reads, 11, &qual, &ladder);

        assert!(!want.is_empty(), "the fixture produced no k-mers");
        for (hc, &n) in &want {
            match got.get(hc) {
                Some(&c) => assert!(
                    c == n || c == n + 1,
                    "k-mer {hc} seen {n} times came back as {c}"
                ),
                // A singleton is invisible unless a false positive admitted it, which is allowed.
                None => assert_eq!(n, 1, "k-mer {hc} seen {n} times is missing from the table"),
            }
        }
        assert!(
            want.values().any(|&n| n >= 2),
            "the fixture has no repeated k-mer to count"
        );
        // Shard assignment is fixed and each shard replays in record order, so scheduling cannot change it.
        assert_eq!(got, bloom_shards(counter_test_reads(), 11, &qual, &ladder));
    }

    /// `shard_of` must spread evenly at every shard count we can pick, since the busiest shard paces
    /// the counting. Sharding on the raw high bits measured a 33x spread, making 16 shards act like 8.
    #[test]
    fn shard_of_is_balanced_at_every_shard_count() {
        for shards in [16usize, 32, 64, 128, 256] {
            let mut sizes = vec![0usize; shards];
            for h in canonical_minimum_hashes(400_000) {
                sizes[shard_of(h, shards)] += 1;
            }
            let ideal = 400_000.0 / shards as f64;
            let worst = *sizes.iter().max().unwrap() as f64 / ideal;
            assert!(worst < 1.1, "{shards} shards: busiest {worst:.2}x ideal");
            assert!(
                sizes.iter().all(|&n| n > 0),
                "{shards} shards: {} were never used",
                sizes.iter().filter(|&&n| n == 0).count()
            );
        }
    }

    /// The store replays the reads exactly: at the ladder, as the first k counts, and at one floor, as
    /// the recount and every later k count.
    #[test]
    fn counting_the_read_store_matches_counting_the_reads() {
        // The reads three times over, so that every width can take a k from its own range.
        let reads: Vec<OwnedRecord> = counter_test_reads()
            .into_iter()
            .map(|(seq, qual)| (seq.repeat(3), qual.map(|q| q.repeat(3))))
            .collect();
        store_counts_match_text::<u64>(&reads, 11);
        store_counts_match_text::<u128>(&reads, 41);
        store_counts_match_text::<crate::utils::bit_encoding::U256>(&reads, 71);
        store_counts_match_text::<crate::utils::bit_encoding::U512>(&reads, 131);
    }

    fn store_counts_match_text<IntT: for<'a> UInt<'a>>(reads: &[OwnedRecord], k: usize) {
        let counts = |shards: Vec<CountMap<IntT>>| -> HashMap<u64, (u32, u64, u8, IntT)> {
            shards
                .into_iter()
                .flatten()
                .map(|(h, i)| (h, (i.count, i.hnc, i.b, i.km)))
                .collect()
        };
        for (ladder, two_bit) in [
            (vec![0, 11, 25], false),
            (vec![20], false),
            (vec![0], false),
            (vec![0], true),
        ] {
            let mut store = ReadStore::new(&ladder, two_bit);
            for (seq, qual) in reads {
                store.push(seq, qual.as_deref());
            }
            store.finish();
            for do_bloom in [false, true] {
                for top in 0..ladder.len() {
                    for classify in [false, true] {
                        let floor = ladder[top];
                        let floors = classify.then_some(&ladder[..=top]);
                        let qual = QualOpts {
                            min_count: 2,
                            min_qual: floor,
                        };
                        let text = count_kmers::<IntT, _>(
                            PassInput::Text(reads.iter().cloned()),
                            k,
                            &qual,
                            floors,
                            0,
                            do_bloom,
                            true,
                        );
                        let packed = count_kmers::<IntT, std::iter::Empty<OwnedRecord>>(
                            PassInput::Stored(&store),
                            k,
                            &qual,
                            floors,
                            if classify { 0 } else { top as u8 },
                            do_bloom,
                            true,
                        );
                        assert_eq!(
                            counts(text.0),
                            counts(packed.0),
                            "k={k} ladder={ladder:?} top={top} two_bit={two_bit} bloom={do_bloom}"
                        );
                        assert_eq!(text.2, packed.2);
                        assert_eq!(
                            text.1.as_ref().map(|s| s.spectra(top + 1)),
                            packed.1.as_ref().map(|s| s.spectra(top + 1))
                        );
                    }
                }
            }
        }
    }
}
