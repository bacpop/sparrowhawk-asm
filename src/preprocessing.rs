//! Assembly preprocessing: this includes k-mer parsing, k-mer filtering, and k-mer counting. We also plot here k-mer spectra and fits to them.

use std::path::PathBuf;
use std::{error::Error, fmt};

use crate::clock::Instant;

use libm::lgamma;
use nohash_hasher::NoHashHasher;
use std::{collections::HashMap, hash::BuildHasherDefault};

use rayon::prelude::*;

#[cfg(not(target_family = "wasm"))]
use plotters::coord::Shift;
#[cfg(not(target_family = "wasm"))]
use plotters::prelude::*;
#[cfg(not(target_family = "wasm"))]
use plotters::style::text_anchor::{HPos, Pos, VPos};

use super::QualOpts;

use crate::bit_encoding::UInt;
use crate::bloom_filter::BloomBits;
use crate::indexed_kmers::IndexedKmers;
use crate::kmer::Kmer;
use crate::qual_profile::{window_groups, MAX_GROUPS, NONE};
use crate::reads::{MaybeSend, OwnedRecord};
use crate::spectrum_fitter::{fit_spectrum, FitAttempt, SpectrumFit};

// Constants

// The histogram

/// Bins of the k-mer spectrum: bin `c - 1` holds the k-mers seen `c` times, and the last bin every
/// count from `MAXSIZEHISTO` up. For the fitting, we'll use actually MAXSIZEHISTO - 1.
const MAXSIZEHISTO: usize = 8000;
/// Window of the k-mer spectrum returned through the WASM preprocessing JSON. It does not limit
/// the current valley or mixture fit.
pub(crate) const LEGACY_HISTO_RANGE: usize = 500;
/// The pinned window has to fit inside the histogram, or the legacy readers would index out of bounds.
const _: () = assert!(LEGACY_HISTO_RANGE <= MAXSIZEHISTO);

// Counting

/// How many records go to the workers at a time: ~1M k-mers at 150 bp and k=31, small enough to stay
/// cache-friendly and large enough to amortise the rayon fork/join.
const BATCH_RECORDS: usize = 8192;
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

// The sketch

/// Sampled hashes held before the threshold halves
const SKETCH_BUDGET: usize = 500_000;
/// The canonical k-mer hash never returns a value below this, so the sample is taken from the interval
/// above it: halving from zero would put the threshold in a range the hash cannot reach, and the whole
/// table would be retained away. Measured over 8M k-mers at k = 21, 31, 41 and 71.
const HASH_FLOOR: u64 = 1 << 52;

// The valley estimator

/// Bins either side of a count in the smoothed spectrum.
const SMOOTH: usize = 2;
/// The run of occupied raw bins that confirms we have left the error lobe rather than hit noise.
const RISE_RUN: usize = 3;
/// How far the genome lobe must stand above the valley. Spectra with a textbook valley and a broad lobe
/// were being rejected at a ratio near 2.4, so a threshold of 3 cut into good data.
const MIN_GP_TO_V_RATIO: f64 = 1.75;
/// Share of k-mer *instances* the lobe above the valley must hold. A handful of noise bins clears 1 %
/// on repeats and adapters alone; a genuine lobe holds 0.75-0.95 of the sequence.
const MIN_CAND_KMER_FRAC: f64 = 0.20;
/// A selected peak's mirrored valley-to-shoulder interval must carry at least this share of all non-singleton bins.
const MIN_PEAK_AREA_FRACTION: f64 = 0.05;
/// Used when the lobes cannot be separated. Not 1: at a genomic peak of 3-4 every singleton error survives and
/// we exhaust memory, which is worse than the ~20 % genome loss cutting at 2 costs there.
const UNRESOLVED_MINCOUNT: u16 = 2;

// The cutoff's guards

/// Genome k-mers the cutoff may delete: this share of those above the valley for the measured guard,
/// and of a Poisson genome for the Poisson one.
const MAX_GENOME_LOSS: f64 = 0.01;
/// Expected single-copy k-mers the cutoff may strand below itself.
const MAX_GENOME_HOLES: f64 = 1.0;

// The floor choice

/// A fit whose error component explains more than this share of observed counts above the applied
/// cutoff is considered too error-heavy when comparing multiple usable quality floors. Provisional;
/// cluster results should be used to recalibrate it.
const MAX_FIT_ERROR_TAIL_FRACTION: f64 = 0.05;
/// A very shallow empirical peak is a reason to inspect the lower floors, not a reason to refuse a fit.
const LOW_COVERAGE_FLOOR_REVIEW: usize = 10;
/// Peak gain a lower floor needs over a starved one (peak below [`LOW_COVERAGE_FLOOR_REVIEW`]) to be
/// stepped down to. Well-covered floors never step down: that cost Neisseria 6-8% auN locally.
const STEP_DOWN_GAIN: f64 = 1.10;

// The plots, native-only

/// Smallest useful horizontal range for a diagnostic plot with no empirical genomic peak.
#[cfg(not(target_family = "wasm"))]
const MIN_PLOT_RANGE: usize = 10;
/// The linear plot's Y axis stops at this multiple of the empirical peak's bar, so the error lobe does
/// not flatten the genome lobe.
#[cfg(not(target_family = "wasm"))]
const PEAK_Y_CAP_MULT: f64 = 3.0;

/// Tuple for name and list of input files
pub type InputFastx = (String, Vec<String>);

/// Everything the preprocessing of one k value produces. Named fields rather than a tuple, since the
/// backends and the public entry point order these differently.
pub struct PreprocessedK<IntT> {
    /// The k this was built with.
    pub k: usize,
    /// K-mer fields stored once in aligned vectors, plus their hash-to-index lookup.
    pub kmers: IndexedKmers<IntT>,
    /// `MAXSIZEHISTO`-bin k-mer spectrum; index `c-1` holds the number of distinct k-mers seen `c` times
    pub histovec: Vec<u32>,
    /// the min-count actually applied (fitted, a fallback, or taken from the CLI)
    pub used_min_count: u16,
    /// the base-quality floor the spectrum asked for. Below `QualOpts::min_qual` when the lobes only
    /// separated at a looser floor, which is the caller's signal to recount.
    pub chosen_min_qual: u8,
    /// Single-copy coverage of the spectrum *this* map was filtered against, as a count. Correction
    /// reads it to tell an error branch from a real one.
    pub genomic_peak: PeakSource,
    /// Whether automatic spectrum selection fell back to the unresolved sentinel without a trusted
    /// peak.
    pub min_count_unresolved: bool,
}

/// A preprocessing or selection failure that prevents a useful assembly step.
#[derive(Debug)]
pub struct PreprocessingError(String);

impl PreprocessingError {
    pub(crate) fn peak_too_low(k: usize, peak: PeakSource) -> Self {
        Self(format!(
            "the single-copy peak at k={k}, {peak:?}, is too low to assemble"
        ))
    }

    pub(crate) fn no_aun_gain(k: usize, new: f64, old: f64) -> Self {
        Self(format!(
            "the unresolved cutoff at k={k} gave an auN of {new:.1}, not above the previous {old:.1}"
        ))
    }
}

impl fmt::Display for PreprocessingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for PreprocessingError {}

#[inline]
fn add_to_histogram(histovec: &mut [u32], count: u32) {
    let idx = if (count as usize) >= MAXSIZEHISTO {
        MAXSIZEHISTO - 1
    } else {
        count.saturating_sub(1) as usize
    };
    histovec[idx] = histovec[idx].saturating_add(1);
}

/// Per-floor k-mer spectra from a fixed-size sample of hash space. Subsampling hash space does not move
/// lambda, as a k-mer's hash does not depend on how often it occurs.
pub struct SpectrumSketch {
    threshold: u64,
    counts: HashMap<u64, [u32; MAX_GROUPS], BuildHasherDefault<NoHashHasher<u64>>>,
    /// Set once the threshold can fall no further without emptying the table.
    saturated: bool,
    /// Times the threshold has halved, for the diagnostic line only.
    shrinks: u32,
}

impl SpectrumSketch {
    /// Allocated once at full size: `shrink` frees entries but not the table, so the high-water mark is
    /// reached anyway, and pre-allocating avoids holding the old and new tables during a resize.
    fn new() -> Self {
        Self {
            threshold: u64::MAX,
            counts: HashMap::with_capacity_and_hasher(SKETCH_BUDGET, BuildHasherDefault::default()),
            saturated: false,
            shrinks: 0,
        }
    }

    /// One compare rejects most occurrences once the threshold has fallen, so this stays cheap in the
    /// hot loop.
    #[inline]
    fn observe(&mut self, hash: u64, group: u8) {
        if hash >= self.threshold || group == NONE {
            return;
        }
        self.counts.entry(hash).or_insert([0; MAX_GROUPS])[group as usize] += 1;
        if !self.saturated && self.counts.len() > SKETCH_BUDGET {
            self.shrink();
        }
    }

    /// Halves the interval above [`HASH_FLOOR`], not the threshold itself, so the threshold converges
    /// on the floor from above and the table always keeps roughly half. Survivors keep the counts they
    /// already had, which is the whole reason the spectrum survives the subsampling.
    fn shrink(&mut self) {
        let (previous, was_populated) = (self.threshold, !self.counts.is_empty());
        self.threshold = HASH_FLOOR + (self.threshold - HASH_FLOOR) / 2;
        let t = self.threshold;
        self.counts.retain(|h, _| *h < t);
        self.shrinks += 1;
        // The floor is a property of the hash, measured rather than derived, so a k whose reachable
        // range starts higher would empty the table here. Stop instead, and keep the sample we have.
        if was_populated && self.counts.is_empty() {
            self.threshold = previous;
            self.saturated = true;
        }
    }

    /// Fraction of k-mers retained, so a sampled spectrum can be rescaled to the whole library. The
    /// hash kept is `min(forward, reverse)`, which is not uniform: `P(min < t) = 1 - (1 - t)^2`, about
    /// twice `t` at the thresholds this reaches. Abundance does not enter it, so the shape is unbiased.
    ///
    /// `t` is measured against the whole of `u64`, not the interval above [`HASH_FLOOR`] that
    /// [`Self::shrink`] halves. That looks inconsistent but is what the library measures: against
    /// `check_sketch_against_table` this predicts 27 484 where 27 291 were sampled, while normalising
    /// by the interval is out by half. The hash is denser just above its floor than uniform.
    fn fraction(&self) -> f64 {
        let p = self.threshold as f64 / u64::MAX as f64;
        1.0 - (1.0 - p) * (1.0 - p)
    }

    /// One spectrum per floor. A k-mer tagged `g` survives floors `0..=g`, so its count at floor `f` is
    /// the total of the groups at or above `f` — hence the reverse accumulation.
    fn spectra(&self, n: usize) -> Vec<Vec<u32>> {
        let mut out = vec![vec![0u32; MAXSIZEHISTO]; n];
        for counts in self.counts.values() {
            let mut acc = 0u32;
            for g in (0..n).rev() {
                acc += counts[g];
                if acc > 0 {
                    add_to_histogram(&mut out[g], acc);
                }
            }
        }
        out
    }

    /// [`Self::spectra`] scaled back to the whole library, since the hole guard budgets in absolute
    /// k-mers. Counts on the x-axis do not move.
    fn rescaled_spectra(&self, n: usize) -> Vec<Vec<u32>> {
        let scale = 1.0 / self.fraction();
        let mut out = self.spectra(n);
        for bin in out.iter_mut().flatten() {
            *bin = ((*bin as f64) * scale).min(u32::MAX as f64) as u32;
        }
        out
    }
}

// =====================================================================================================
// Empirical `min_count` estimator: first locate candidates in the raw spectrum, then refine them in
// the smoothed spectrum without allowing the refinement to escape the raw peak's neighbourhood.
// =====================================================================================================

/// How the raw candidates were turned into the selected empirical valley and peak.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
enum PeakSearch {
    /// No pair was accepted.
    #[default]
    Unresolved,
    /// The raw pair, low and close: both under 15, and under 10 apart.
    RawLowCoverage,
    /// The raw pair refined on the smoothed spectrum.
    SmoothedRefinement,
    /// The raw pair, when the refinement found none.
    RawFallback,
    /// The first raw local valley and the tallest bin after it.
    RawLocalFallback,
}

/// How much of the spectrum a valley-peak pair's lobe holds: the k-mers from the valley to the peak's
/// mirror on its far side, against every non-singleton k-mer.
#[derive(Clone, Copy, Debug, PartialEq)]
struct PeakAreaSupport {
    /// Distinct k-mers from the valley to the mirrored shoulder.
    area_sum: u64,
    /// Distinct k-mers of every count from 2, the saturating last bin included.
    denominator: u64,
}

impl PeakAreaSupport {
    /// Sum the valley-to-mirrored-shoulder interval and all non-singleton bins. The overflow bin is
    /// retained in the denominator but cannot be used as the interval's right endpoint. `None` when the
    /// pair is not valley-then-peak, the mirrored shoulder reaches the saturating last bin, or no k-mer
    /// is seen twice.
    fn measure(histovec: &[u32], valley: usize, peak: usize) -> Option<Self> {
        if valley < 2 || peak <= valley || peak >= histovec.len() {
            return None;
        }
        let mirrored_end = peak.checked_mul(2)?.checked_sub(valley)?;
        let last_unsaturated_count = histovec.len().checked_sub(1)?;
        if mirrored_end > last_unsaturated_count {
            return None;
        }
        let denominator: u64 = histovec[1..].iter().map(|&height| u64::from(height)).sum();
        if denominator == 0 {
            return None;
        }
        let area_sum: u64 = (valley..=mirrored_end)
            .map(|count| u64::from(histovec[count - 1]))
            .sum();
        Some(Self {
            area_sum,
            denominator,
        })
    }

    /// The lobe's share of the non-singleton k-mers.
    fn fraction(self) -> f64 {
        self.area_sum as f64 / self.denominator as f64
    }

    /// Whether that share reaches [`MIN_PEAK_AREA_FRACTION`].
    fn passes(self) -> bool {
        self.fraction() >= MIN_PEAK_AREA_FRACTION
    }
}

/// What the valley estimator concluded about one spectrum, and the figures behind it.
#[derive(Clone, Copy, Default)]
struct SpectrumEstimate {
    /// Where the raw walk left the error lobe: the first raw minimum followed by [`RISE_RUN`] occupied
    /// bins not below it, as a count; 0 when none was found. Plotted.
    valley_seed: usize,
    /// Lowest raw bin from count 2 to the raw genomic peak, as a count. Plotted.
    raw_valley: Option<usize>,
    /// Highest raw bin at or above the seed, as a count. Plotted.
    raw_genomic_peak: Option<usize>,
    /// How the selected pair was found; `Unresolved` when none was accepted.
    peak_search: PeakSearch,
    /// The selected valley, as a count; 0 when unresolved.
    valley: usize,
    /// The selected single-copy peak, as a count; 0 when unresolved.
    genomic_peak: usize,
    /// Distinct k-mers in the valley's bin.
    valley_n: u32,
    /// Distinct k-mers in the peak's bin.
    genomic_peak_n: u32,
    /// `genomic_peak_n / valley_n`, the quantity [`MIN_GP_TO_V_RATIO`] guards.
    gp_to_v_ratio: f64,
    /// Share of k-mer instances at or above the valley, the quantity [`MIN_CAND_KMER_FRAC`] guards.
    cand_kmer_frac: f64,
    /// Largest cutoff whose measured cost is at most [`MAX_GENOME_LOSS`] of the k-mers above the
    /// valley; binds at depth.
    guard_measured: u16,
    /// Largest cutoff deleting at most [`MAX_GENOME_LOSS`] of a Poisson(`genomic_peak`) genome; binds
    /// at low coverage.
    guard_poisson: u16,
    /// The cutoff: the valley, clamped to 2 and to both guards; [`UNRESOLVED_MINCOUNT`] when unresolved.
    min_count: u16,
    /// Variance-to-mean ratio of the genome lobe, from the valley to twice the peak, seeding the fit's
    /// dispersion; NaN without a lobe.
    dispersion: f64,
    /// Whether the lobes separated, and if not, why.
    verdict: Verdict,
}

/// Values needed to explain the spectrum decision in the native PNG. Kept independent of plotting
/// types so the fitting path can carry it without pulling native-only code into WASM, where nothing
/// reads it.
#[derive(Clone, Copy, Default)]
#[cfg_attr(target_family = "wasm", allow(dead_code))]
struct SpectrumPlotDiagnostics {
    valley_seed: Option<usize>,
    valley: Option<usize>,
    empirical_peak: Option<usize>,
    raw_valley: Option<usize>,
    genomic_peak_seed: Option<usize>,
    fit: Option<SpectrumFit>,
    fit_attempted: bool,
    genomic_guard: Option<u16>,
    error_floor: Option<u16>,
}

impl SpectrumPlotDiagnostics {
    fn new(estimate: &SpectrumEstimate, fit: Option<SpectrumFit>, fit_attempted: bool) -> Self {
        Self {
            valley_seed: (estimate.valley_seed > 0).then_some(estimate.valley_seed),
            valley: (estimate.valley > 0).then_some(estimate.valley),
            empirical_peak: (estimate.genomic_peak > 0).then_some(estimate.genomic_peak),
            raw_valley: estimate.raw_valley,
            genomic_peak_seed: estimate.raw_genomic_peak,
            fit,
            fit_attempted,
            genomic_guard: estimate
                .verdict
                .is_ok()
                .then(|| fit.and_then(|fit| fit.hole_cutoff(MAX_GENOME_HOLES)))
                .flatten(),
            error_floor: fit.map(|fit| fit.error_floor_cutoff()),
        }
    }
}

/// Why the estimator accepted or refused a spectrum. The default is a *refusal* on purpose: every path
/// through [`SpectrumEstimate::by_valley`] sets it, so the default is unreachable, and if one ever stops setting
/// it the run should fail closed rather than silently report a resolved spectrum.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
enum Verdict {
    #[default]
    NeverTurnsUp,
    NoLocalValley,
    NoPeakAboveValley,
    PeakTailNotDecreasing,
    PeakAreaNotMeasurable,
    PeakAreaTooSmall,
    TooFewCandidateKmers,
    PeakNotClearOfValley,
    Ok,
}

impl Verdict {
    /// Whether the lobes separated. Everything else is a refusal.
    fn is_ok(self) -> bool {
        matches!(self, Verdict::Ok)
    }

    /// The clause naming this outcome in the terminal warning.
    fn reason(self) -> &'static str {
        match self {
            Verdict::NeverTurnsUp => "the spectrum never turns back up",
            Verdict::NoLocalValley => "the raw histogram has no local valley",
            Verdict::NoPeakAboveValley => "no genomic_peak above the valley",
            Verdict::PeakTailNotDecreasing => {
                "the candidate peak has no confirmed declining right tail"
            }
            Verdict::PeakAreaNotMeasurable => {
                "the peak's mirrored area cannot be measured within the unsaturated histogram"
            }
            Verdict::PeakAreaTooSmall => {
                "the peak's mirrored area is below 5% of non-singleton bins"
            }
            Verdict::TooFewCandidateKmers => {
                "the lobe above the valley holds too little of the sequence"
            }
            Verdict::PeakNotClearOfValley => "the lobe above the valley is not raised clear of it",
            // Reached when the lobes did separate but the loss guard pulled the cutoff to the floor.
            Verdict::Ok => "the cutoff would have cost more genome than the guard allows",
        }
    }
}

/// Where a single-copy coverage figure came from. Correction is more careful with a fallback than
/// with a fitted peak, so the provenance travels with the number rather than being inferred later.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum PeakSource {
    /// The spectrum's lobes separated and `genomic_peak` is trustworthy.
    Fitted(u32),
    /// They did not, so the occurrence-weighted median stands in.
    Fallback(u32),
    /// No spectrum was read at all, as with an empty histogram.
    #[default]
    Unknown,
}

impl PeakSource {
    /// The coverage figure, whatever its provenance.
    pub fn value(self) -> Option<u32> {
        match self {
            Self::Fitted(peak) | Self::Fallback(peak) => Some(peak),
            Self::Unknown => None,
        }
    }
}

/// Single-copy coverage without needing a valley: the smallest count at which k-mers of that count
/// or less hold half of all k-mer *occurrences*. Error k-mers are many but each occurs a few times,
/// so they carry little occurrence mass, and the median lands in the genomic lobe regardless.
fn occurrence_weighted_median(histovec: &[u32]) -> Option<u32> {
    // `histovec[c - 1]` is the count-`c` bin, so the count is the index plus one. u64 is ample: the
    // total is the number of k-mers in the reads, ~1e10 at the very most.
    let occurrences = |(i, n): (usize, &u32)| (i as u64 + 1) * u64::from(*n);
    let total: u64 = histovec.iter().enumerate().map(occurrences).sum();
    if total == 0 {
        return None;
    }

    let mut seen = 0u64;
    for (i, n) in histovec.iter().enumerate() {
        seen += occurrences((i, n));
        if seen * 2 >= total {
            return u32::try_from(i + 1).ok();
        }
    }
    None
}

/// Smooth each histogram bin once so valley and peak searches use the same five-bin view: at each
/// count, the mean of the bins up to [`SMOOTH`] either side, the saturating last bin excluded.
fn smooth_histogram(histovec: &[u32]) -> Vec<u64> {
    (1..=histovec.len())
        .map(|count| {
            let lo = count.saturating_sub(SMOOTH).max(1);
            let hi = (count + SMOOTH).min(histovec.len() - 1);
            (lo..=hi).map(|c| histovec[c - 1] as u64).sum::<u64>() / (hi - lo + 1) as u64
        })
        .collect()
}

/// First raw-histogram minimum followed by [`RISE_RUN`] occupied bins that are not below it, as a
/// **count**. Empty bins are skipped: in a sparse spectrum they are gaps, not evidence of a valley
/// or a rising lobe. Flat occupied bins count as confirmation, matching the historical rule. This
/// only seeds the raw peak search; the valley is subsequently found across the full range below it.
fn find_raw_valley_seed(histovec: &[u32]) -> Option<usize> {
    if histovec.len() <= RISE_RUN + 2 {
        return None;
    }
    let last_unsaturated_count = histovec.len() - 1;
    let mut counts = (2..=last_unsaturated_count).filter(|&count| histovec[count - 1] > 0);
    let mut best_count = counts.next()?;
    let mut best_n = histovec[best_count - 1];
    let mut confirmed = 0usize;
    for count in counts {
        let n = histovec[count - 1];
        if n < best_n {
            best_n = n;
            best_count = count;
            confirmed = 0;
        } else {
            confirmed += 1;
            if confirmed >= RISE_RUN {
                return Some(best_count);
            }
        }
    }
    None
}

/// Highest raw bin at or above the seed, as a **count**. Ties choose the lowest count, and the
/// saturated final bin is excluded.
fn find_raw_genomic_peak(histovec: &[u32], seed: usize) -> Option<usize> {
    let last_unsaturated_count = histovec.len().checked_sub(1)?;
    if seed < 2 || seed > last_unsaturated_count {
        return None;
    }
    (seed..=last_unsaturated_count)
        .min_by_key(|&count| (std::cmp::Reverse(histovec[count - 1]), count))
}

/// Largest cutoff whose *measured* cost is at most [`MAX_GENOME_LOSS`] of the k-mers above the valley. This binds/protects at large coverage.
fn measured_guard(histovec: &[u32], valley: usize, genomic_peak: usize) -> u16 {
    let total: u128 = (valley..histovec.len())
        .map(|c| histovec[c - 1] as u128)
        .sum();
    if total == 0 {
        return UNRESOLVED_MINCOUNT;
    }
    let budget = (total as f64 * MAX_GENOME_LOSS) as u128;
    // `spent` is what cutting at `m` already costs, so we step up only while the next bin still fits;
    // the value returned is therefore the last cutoff inside the budget, never the first one outside.
    let mut spent = 0u128;
    let mut m = valley;
    while m < genomic_peak {
        let next = spent + histovec[m - 1] as u128;
        if next > budget {
            break;
        }
        spent = next;
        m += 1;
    }
    (m as u16).max(2)
}

/// Largest cutoff deleting at most [`MAX_GENOME_LOSS`] of a Poisson(`genomic_peak`) genome. This binds/protects at low coverage.
fn poisson_guard(genomic_peak: usize) -> u16 {
    let lam = genomic_peak as f64;
    let mut cdf = (-lam).exp();
    let mut m = 1usize;
    while m < genomic_peak {
        let next = cdf + (-lam + m as f64 * lam.ln() - lgamma(m as f64 + 1.0)).exp();
        if next > MAX_GENOME_LOSS {
            break;
        }
        cdf = next;
        m += 1;
    }
    (m as u16).max(2)
}

/// Variance-to-mean ratio of the single-copy lobe; 1.0 is Poisson.
/// Stops at twice the genomic peak: beyond that lie the repeat copies and the saturating bin, and including
/// them measures the spread of the whole spectrum rather than of the lobe the fit tries to model.
fn dispersion_above(histovec: &[u32], valley: usize, genomic_peak: usize) -> f64 {
    let top = (2 * genomic_peak).min(histovec.len() - 1);
    let (mut n, mut sx, mut sxx) = (0f64, 0f64, 0f64);
    for count in valley..top {
        let w = histovec[count - 1] as f64;
        let c = count as f64;
        n += w;
        sx += w * c;
        sxx += w * c * c;
    }
    if n <= 0.0 || sx <= 0.0 {
        return f64::NAN;
    }
    let mean = sx / n;
    ((sxx / n - mean * mean).max(0.0)) / mean
}

/// Distinct k-mers at or above `valley`: the genome lobe's breadth, where `genomic_peak` is its depth.
/// A floor that breaks contiguous windows deletes k-mers without moving the peak, so only this sees it.
fn distinct_above(histovec: &[u32], valley: usize) -> u64 {
    histovec[valley.saturating_sub(1)..]
        .iter()
        .map(|&n| n as u64)
        .sum()
}

/// Refine the valley on the smoothed histogram, anchored by the raw candidates. Ties prefer the
/// raw valley, then the lower count.
fn refine_smoothed_valley(smooth: &[u64], raw_valley: usize, raw_peak: usize) -> Option<usize> {
    let last_unsaturated_count = smooth.len().checked_sub(1)?;
    let hi = raw_peak.min(last_unsaturated_count);
    if hi < 2 {
        return None;
    }
    (2..=hi).min_by_key(|&count| (smooth[count - 1], count.abs_diff(raw_valley), count))
}

/// Refine the peak on the smoothed histogram. Its range starts above both the raw and refined
/// valleys, and ends at `floor(1.3 * raw_peak)` or the last unsaturated bin, whichever comes first.
/// Ties prefer the raw peak, then the lower count.
fn refine_smoothed_peak(
    histovec: &[u32],
    smooth: &[u64],
    raw_valley: usize,
    raw_peak: usize,
    refined_valley: usize,
) -> Option<usize> {
    let last_unsaturated_count = smooth.len().checked_sub(1)?;
    let lo = raw_valley.max(refined_valley.saturating_add(1));
    let hi = raw_peak
        .saturating_mul(13)
        .checked_div(10)?
        .min(last_unsaturated_count);
    if lo > hi {
        return None;
    }
    (lo..=hi)
        .filter(|&count| histovec.get(count - 1).is_some_and(|&n| n > 0))
        .min_by_key(|&count| {
            (
                std::cmp::Reverse(smooth[count - 1]),
                count.abs_diff(raw_peak),
                count,
            )
        })
}

/// First raw local minimum from count 2; count 2 uses count 1 as its left neighbour.
fn find_first_raw_local_valley(histovec: &[u32]) -> Option<usize> {
    let last_unsaturated_count = histovec.len().checked_sub(1)?;
    (2..last_unsaturated_count).find(|&count| {
        let height = histovec[count - 1];
        height < histovec[count - 2] && height < histovec[count]
    })
}

/// Tallest raw bin strictly to the right of a local valley. Ties choose the lower count.
fn find_raw_peak_after_valley(histovec: &[u32], valley: usize) -> Option<usize> {
    let last_unsaturated_count = histovec.len().checked_sub(1)?;
    if valley >= last_unsaturated_count {
        return None;
    }
    ((valley + 1)..=last_unsaturated_count)
        .min_by_key(|&count| (std::cmp::Reverse(histovec[count - 1]), count))
}

/// Confirm a peak with two consecutive declines in overlapping averages of adjacent raw bins.
fn confirms_two_bin_right_tail(histovec: &[u32], peak: usize) -> bool {
    let Some(last_unsaturated_count) = histovec.len().checked_sub(1) else {
        return false;
    };
    let Some(last_pair_start) = peak.checked_add(2) else {
        return false;
    };
    if last_pair_start + 1 > last_unsaturated_count {
        return false;
    }
    let pair_sum =
        |left_count: usize| u64::from(histovec[left_count - 1]) + u64::from(histovec[left_count]);
    let first = pair_sum(peak);
    let second = pair_sum(peak + 1);
    let third = pair_sum(peak + 2);
    first > second && second > third
}

impl SpectrumEstimate {
    /// The valley estimate for this spectrum. Falls back to [`UNRESOLVED_MINCOUNT`] with a stated verdict
    /// wherever the lobes are not separated — at 3-5x, trusting it blindly deleted 98-99 % of the genome.
    /// The primary pair comes first; when it is refused, the first raw local valley and the tallest bin
    /// after it. Each attempt is logged where it is decided.
    fn by_valley(histovec: &[u32]) -> Self {
        let mut estimate = Self {
            min_count: UNRESOLVED_MINCOUNT,
            dispersion: f64::NAN,
            ..Self::default()
        };
        match estimate.find_primary_pair(histovec) {
            Ok((valley, peak, method)) => {
                let area = PeakAreaSupport::measure(histovec, valley, peak);
                let mut primary = estimate;
                primary.set_pair(histovec, valley, peak);
                primary.peak_search = method;
                let refusal = match area {
                    None => Some("mirrored area not measurable"),
                    Some(support) if !support.passes() => Some("mirrored area below 5%"),
                    Some(_) => (!primary.verdict.is_ok()).then(|| primary.verdict.reason()),
                };
                log::info!(
                    "Valley search, primary pair: seed={} raw_valley={:?} raw_peak={:?} valley={valley} \
                     peak={peak} method={method:?} area_pct={:?} outcome={}",
                    primary.valley_seed,
                    primary.raw_valley,
                    primary.raw_genomic_peak,
                    area.map(|support| 100.0 * support.fraction()),
                    refusal.unwrap_or("accepted")
                );
                if refusal.is_none() {
                    return primary;
                }
            }
            Err(reason) => log::info!("Valley search, primary pair: none, {reason}"),
        }

        let Some(valley) = find_first_raw_local_valley(histovec) else {
            estimate.verdict = Verdict::NoLocalValley;
            log::info!(
                "Valley search, local fallback: none, {}",
                estimate.verdict.reason()
            );
            return estimate;
        };
        let Some(peak) = find_raw_peak_after_valley(histovec, valley) else {
            estimate.verdict = Verdict::NoPeakAboveValley;
            log::info!(
                "Valley search, local fallback: valley={valley}, {}",
                estimate.verdict.reason()
            );
            return estimate;
        };
        let tail_confirmed = confirms_two_bin_right_tail(histovec, peak);
        let area = PeakAreaSupport::measure(histovec, valley, peak);
        let mut fallback = estimate;
        fallback.set_pair(histovec, valley, peak);
        if !tail_confirmed {
            fallback.verdict = Verdict::PeakTailNotDecreasing;
        } else if area.is_none() {
            fallback.verdict = Verdict::PeakAreaNotMeasurable;
        } else if area.is_some_and(|support| !support.passes()) {
            fallback.verdict = Verdict::PeakAreaTooSmall;
        }
        log::info!(
            "Valley search, local fallback: valley={valley} peak={peak} tail_confirmed={tail_confirmed} \
             area_pct={:?} verdict={:?}",
            area.map(|support| 100.0 * support.fraction()),
            fallback.verdict
        );
        if fallback.verdict.is_ok() {
            fallback.peak_search = PeakSearch::RawLocalFallback;
            return fallback;
        }
        // A refused pair must not escape as an empirical valley or genomic peak to floor arbitration,
        // fitting, or plots.
        fallback.clear_pair();
        fallback
    }

    /// The primary pair, from the raw spectrum: the valley seed, the tallest bin above it, and the
    /// lowest bin from count 2 up to that peak. A low and close pair stands as it is; others are refined
    /// on the smoothed spectrum. Records the seed and the raw pair; the error says where the search
    /// stopped.
    fn find_primary_pair(
        &mut self,
        histovec: &[u32],
    ) -> Result<(usize, usize, PeakSearch), &'static str> {
        let valley_seed = find_raw_valley_seed(histovec).ok_or("no valley seed")?;
        self.valley_seed = valley_seed;
        let genomic_peak_seed =
            find_raw_genomic_peak(histovec, valley_seed).ok_or("no raw peak above the seed")?;
        // The lowest raw bin in `2..=genomic_peak_seed`, as a count; the saturating last bin is excluded.
        let raw_valley = (2..=genomic_peak_seed.min(histovec.len() - 1))
            .min_by_key(|&count| (histovec[count - 1], count))
            .ok_or("no raw valley under the peak")?;
        self.raw_valley = Some(raw_valley);
        self.raw_genomic_peak = Some(genomic_peak_seed);
        if genomic_peak_seed <= raw_valley || histovec[genomic_peak_seed - 1] == 0 {
            return Err("the raw peak does not rise above the raw valley");
        }

        if genomic_peak_seed - raw_valley < 10 && genomic_peak_seed < 15 && raw_valley < 15 {
            return Ok((raw_valley, genomic_peak_seed, PeakSearch::RawLowCoverage));
        }
        let smooth = smooth_histogram(histovec);
        let refined =
            refine_smoothed_valley(&smooth, raw_valley, genomic_peak_seed).and_then(|valley| {
                refine_smoothed_peak(histovec, &smooth, raw_valley, genomic_peak_seed, valley)
                    .filter(|&peak| peak > valley)
                    .map(|peak| (valley, peak))
            });
        Ok(match refined {
            Some((valley, peak)) => (valley, peak, PeakSearch::SmoothedRefinement),
            None => (raw_valley, genomic_peak_seed, PeakSearch::RawFallback),
        })
    }

    /// Populate the metrics and existing separation verdict for a proposed pair.
    fn set_pair(&mut self, histovec: &[u32], valley: usize, genomic_peak: usize) {
        self.valley = valley;
        self.genomic_peak = genomic_peak;
        self.valley_n = histovec[valley - 1];
        self.genomic_peak_n = histovec[genomic_peak - 1];
        self.gp_to_v_ratio = self.genomic_peak_n as f64 / self.valley_n.max(1) as f64;

        let instances_from = |from: usize| -> u128 {
            (from..=histovec.len())
                .map(|count| count as u128 * histovec[count - 1] as u128)
                .sum()
        };
        let total_instances = instances_from(1);
        self.cand_kmer_frac = if total_instances == 0 {
            0.0
        } else {
            instances_from(valley) as f64 / total_instances as f64
        };
        self.guard_measured = measured_guard(histovec, valley, genomic_peak);
        self.guard_poisson = poisson_guard(genomic_peak);
        self.dispersion = dispersion_above(histovec, valley, genomic_peak);

        self.verdict = if genomic_peak <= valley {
            Verdict::NoPeakAboveValley
        } else if self.cand_kmer_frac < MIN_CAND_KMER_FRAC {
            Verdict::TooFewCandidateKmers
        } else if self.gp_to_v_ratio < MIN_GP_TO_V_RATIO {
            Verdict::PeakNotClearOfValley
        } else {
            Verdict::Ok
        };
        if self.verdict.is_ok() {
            self.min_count = (valley as u16).clamp(2, self.guard_measured.min(self.guard_poisson));
        } else {
            self.min_count = UNRESOLVED_MINCOUNT;
        }
    }

    /// Forget a refused pair, keeping the verdict that refused it.
    fn clear_pair(&mut self) {
        self.valley = 0;
        self.genomic_peak = 0;
        self.valley_n = 0;
        self.genomic_peak_n = 0;
        self.gp_to_v_ratio = 0.0;
        self.cand_kmer_frac = 0.0;
        self.guard_measured = 0;
        self.guard_poisson = 0;
        self.min_count = UNRESOLVED_MINCOUNT;
        self.dispersion = f64::NAN;
    }

    /// The whole derivation, as `key=value` pairs so a directory of logs parses into a table. The prose a
    /// user reads is the warning in [`Self::warn_if_unresolved`], not this.
    fn log_spectrum(&self, histovec: &[u32]) {
        // `valley_to_peak_xratio` is logged, never gated on: near 1.0 usually means the valley search
        // found nothing and stopped under the genomic peak, but a tight clean separation reads the same
        // way. `distinct_above` is the lobe's breadth, where `genomic_peak` is its depth. Both are 0
        // without a pair.
        let (valley_to_peak_xratio, breadth) = if self.genomic_peak > 0 {
            (
                self.valley as f64 / self.genomic_peak as f64,
                distinct_above(histovec, self.valley),
            )
        } else {
            (0.0, 0)
        };
        log::info!(
            "K-mer spectrum: seed={} valley={} genomic_peak={} valley_n={} genomic_peak_n={} \
                 gp_to_v_ratio={:.2} valley_to_peak_xratio={:.3} cand_kmer_frac={:.4} \
                 guard_measured={} guard_poisson={} dispersion={:.1} min_count={} distinct_above={} \
                 saturates={} method={:?} verdict={:?}",
            self.valley_seed,
            self.valley,
            self.genomic_peak,
            self.valley_n,
            self.genomic_peak_n,
            self.gp_to_v_ratio,
            valley_to_peak_xratio,
            self.cand_kmer_frac,
            self.guard_measured,
            self.guard_poisson,
            self.dispersion,
            self.min_count,
            breadth,
            histovec[histovec.len() - 1] > 0,
            self.peak_search,
            self.verdict
        );
    }

    /// The fitted peak when the lobes separated, else the occurrence-weighted median.
    ///
    /// `genomic_peak` is an argmax *above a valley*, so on a merged spectrum it names the error lobe.
    /// Refusing it there is what lets every consumer treat `Fitted` as trustworthy.
    fn peak_source(&self, histovec: &[u32]) -> PeakSource {
        if self.verdict.is_ok() {
            PeakSource::Fitted(self.genomic_peak as u32)
        } else {
            occurrence_weighted_median(histovec).map_or(PeakSource::Unknown, PeakSource::Fallback)
        }
    }

    /// The cutoff [`Self::apply_hole_guard`] settles on: the valley capped by the hole guard, then raised
    /// to the 100% error floor.
    fn guarded_min_count(&self, fit: &SpectrumFit) -> u16 {
        self.hole_guarded_min_count(fit)
            .max(fit.error_floor_cutoff())
    }

    /// The valley lowered to what strands at most [`MAX_GENOME_HOLES`] single-copy k-mers, never below 2,
    /// and only for a resolved spectrum.
    fn hole_guarded_min_count(&self, fit: &SpectrumFit) -> u16 {
        match fit.hole_cutoff(MAX_GENOME_HOLES) {
            Some(cutoff) if self.verdict.is_ok() && cutoff < self.min_count => cutoff.max(2),
            _ => self.min_count,
        }
    }

    /// Lower the cutoff to whatever strands at most [`MAX_GENOME_HOLES`] single-copy k-mers.
    ///
    /// A **ceiling** on the valley, never a raise: the valley already removes the errors, and the only
    /// failure measured was cutting too deep. Floored at 2, and skipped by callers when the fit did not
    /// converge, so the worst case is exactly the behaviour without a fit.
    fn apply_hole_guard(&mut self, fit: &SpectrumFit) {
        let hole_guarded = self.hole_guarded_min_count(fit);
        if hole_guarded < self.min_count {
            log::info!(
                "Cutting at {} would strand more than {MAX_GENOME_HOLES} single-copy k-mers; using \
                     {hole_guarded} instead.",
                self.min_count
            );
        }
        let error_floor = fit.error_floor_cutoff();
        self.min_count = hole_guarded.max(error_floor);
        let hole_cutoff = fit.hole_cutoff(MAX_GENOME_HOLES);
        log::info!("Spectrum 100% error floor enforced: floor={error_floor} previous_min_count={hole_guarded} \
                 final_min_count={} hole_cutoff={hole_cutoff:?} conflict={}",
                self.min_count,
                hole_cutoff.is_some_and(|cutoff| error_floor > cutoff));
    }

    /// The warning for a spectrum that yielded no usable cutoff.
    fn warn_if_unresolved(&self) {
        // Keyed on the outcome, not only the verdict: a spectrum can be called `ok` and still have the loss
        // guard collapse the cutoff to `UNRESOLVED_MINCOUNT`, which is the same thin-spectrum situation.
        if !self.verdict.is_ok() || self.min_count == UNRESOLVED_MINCOUNT {
            log::warn!("The k-mer spectrum's error and genome lobes are not separated ({}), so no reliable \
                     minimum count exists and {} will be used — k-mers seen once are discarded and nothing \
                     else is. Expect a fragmented assembly if this is a deep library. This usually means \
                     the spectrum is thin: low coverage, or a large k. Check the k-mer spectrum histogram.",
                    self.verdict.reason(), self.min_count);
        }
    }
}

/// Fit the spectrum and report it. Seeded from the valley estimator's own peak and dispersion: the
/// estimator locates the lobe, the fit refines it and separates the error lobe from it explicitly.
/// `None` when no start converges, which leaves the estimator's answer standing.
/// `sample_scale`: library k-mers per observed one, 1/fraction for a sketch and 1 for an exact table.
fn fit_and_log(
    histovec: &[u32],
    estimate: &SpectrumEstimate,
    sample_scale: f64,
) -> Option<SpectrumFit> {
    if !estimate.verdict.is_ok() {
        log::info!("Spectrum fit skipped because the empirical peak pair is unresolved.");
        return None;
    }
    let started = Instant::now();
    let result = match fit_spectrum(
        histovec,
        estimate.valley,
        estimate.genomic_peak,
        estimate.dispersion,
        sample_scale,
    ) {
        Ok(result) => result,
        Err(error) => {
            log::info!("Spectrum fits could not start ({error}); keeping the valley estimate.");
            return None;
        }
    };
    for attempt in &result.attempts {
        log_fit_attempt(attempt);
    }
    let elapsed = started.elapsed();
    if let Some(fit) = result.selected {
        log::info!("Spectrum fit selected: error_model={} genome_model={} log_likelihood={:.6e} bic={:.6e} deviance={:.6e} \
                 mean={:.1} mode={} repeat_mode={} repeat_ratio={:.3} dispersion={:.2} singleton_probability={:.6} tail_exponent={:.6} weibull_shape={} \
                 shelf={} w=({:.3}/{:.3}/{:.3}) genome_kmers={:.3e} error_kmers={:.3e} crossover={} \
                 hole_cutoff={} best_iterations={} elapsed_ms={} sample_scale={:.1}",
                fit.error_model,
                fit.genome_model,
                fit.log_likelihood,
                fit.bic,
                fit.deviance,
                fit.mean,
                fit.primary_mode(),
                fit.repeat_mode(),
                fit.repeat_ratio,
                fit.dispersion,
                fit.error_params.singleton_probability,
                fit.error_params.tail_exponent,
                fit.error_params.weibull_shape.map_or_else(|| "n/a".to_string(), |shape| format!("{shape:.6}")),
                shelf_summary(&fit),
                fit.w_error,
                fit.w_single,
                fit.w_repeat,
                fit.genome_kmers,
                fit.error_kmers,
                fit.crossover().map_or_else(|| "none".to_string(), |value| value.to_string()),
                fit.hole_cutoff(MAX_GENOME_HOLES)
                    .map_or_else(|| "none".to_string(), |value| value.to_string()),
                fit.best_iterations,
                elapsed.as_millis(),
                sample_scale);
        Some(fit)
    } else {
        log::info!(
            "All spectrum fits were rejected; keeping the valley estimate. elapsed_ms={}",
            elapsed.as_millis()
        );
        None
    }
}

/// The fitted shelf, or `none` for a model without one.
fn shelf_summary(fit: &SpectrumFit) -> String {
    fit.shelf.map_or_else(
        || "none".to_string(),
        |s| {
            format!(
                "(share={:.4} alpha={:.3} beta={:.3})",
                s.share, s.alpha, s.beta
            )
        },
    )
}

fn log_fit_attempt(attempt: &FitAttempt) {
    if let Some(fit) = attempt.candidate {
        let log_window_masses = fit.component_log_window_masses;
        let status = attempt.rejection.map_or_else(
            || "accepted".to_string(),
            |reason| format!("rejected:{reason}"),
        );
        log::info!("Spectrum fit candidate: error_model={} genome_model={} status={} log_likelihood={:.6e} bic={:.6e} \
                 deviance={:.6e} mode={} repeat_mode={} repeat_ratio={:.3} error_mode={} mean={:.1} dispersion={:.2} \
                 singleton_probability={:.6} tail_exponent={:.6} weibull_shape={} shelf={} \
                 component_kmers=({:.3e}/{:.3e}/{:.3e}) component_mass_ratio={:.6e} \
                 log_window_mass=({:.6e}/{:.6e}/{:.6e}) mode_roundtrip_error={:.6e} \
                 total_iterations={} capped_starts={} best_iterations={}",
                attempt.error_model,
                attempt.genome_model,
                status,
                fit.log_likelihood,
                fit.bic,
                fit.deviance,
                fit.primary_mode(),
                fit.repeat_mode(),
                fit.repeat_ratio,
                fit.error_mode(),
                fit.mean,
                fit.dispersion,
                fit.error_params.singleton_probability,
                fit.error_params.tail_exponent,
                fit.error_params.weibull_shape.map_or_else(|| "n/a".to_string(), |shape| format!("{shape:.6}")),
                shelf_summary(&fit),
                fit.error_kmers,
                fit.genome_kmers,
                fit.repeat_kmers,
                fit.component_total_ratio(),
                log_window_masses[0],
                log_window_masses[1],
                log_window_masses[2],
                fit.mode_roundtrip_error(),
                attempt.total_iterations,
                attempt.capped_starts,
                fit.best_iterations);
    } else {
        log::info!("Spectrum fit candidate: error_model={} genome_model={} status=rejected:{} total_iterations={} capped_starts={}",
                attempt.error_model,
                attempt.genome_model,
                attempt
                    .rejection
                    .expect("an absent candidate always has a rejection reason"),
                attempt.total_iterations,
                attempt.capped_starts);
    }
}

// =====================================================================================================
// =====================================================================================================
// =====================================================================================================

/// As [`choose_min_count`], and also the single-copy coverage the same estimate found.
///
/// The valley separates the lobes; the fit then says how much genome cutting there would cost. Those
/// answer different questions, and on deep libraries the valley is far the more aggressive.
/// `sample_scale` as in [`fit_and_log`].
fn choose_min_count_and_peak(
    histovec: &[u32],
    sample_scale: f64,
) -> (u16, PeakSource, SpectrumPlotDiagnostics) {
    let mut estimate = SpectrumEstimate::by_valley(histovec);
    estimate.log_spectrum(histovec);
    let fit_attempted = estimate.verdict.is_ok();
    let fit = fit_attempted
        .then(|| fit_and_log(histovec, &estimate, sample_scale))
        .flatten();
    if let Some(fit) = &fit {
        estimate.apply_hole_guard(fit);
    }
    estimate.warn_if_unresolved();
    (
        estimate.min_count,
        estimate.peak_source(histovec),
        SpectrumPlotDiagnostics::new(&estimate, fit, fit_attempted),
    )
}

#[derive(Clone, Copy)]
struct FloorCandidate {
    floor: u8,
    estimate: SpectrumEstimate,
    fit: Option<SpectrumFit>,
    fit_attempted: bool,
    error_tail_fraction: Option<f64>,
}

struct FloorDiagnostic {
    floor: u8,
    spectrum: Vec<u32>,
    estimate: SpectrumEstimate,
    fit: Option<SpectrumFit>,
    fit_attempted: bool,
}

/// Share of observed counts the fitted error component explains, from the applied cutoff `start`
/// through six times the fitted primary mode: the error share among the k-mers the cutoff keeps.
fn fit_error_tail_fraction(histovec: &[u32], fit: &SpectrumFit, start: u16) -> Option<f64> {
    let start = usize::from(start).max(1);
    let end = fit
        .primary_mode()
        .checked_mul(6)?
        .min(fit.fit_window_end)
        .min(histovec.len().saturating_sub(1));
    if start > end {
        return None;
    }
    let observed: u64 = (start..=end)
        .map(|count| u64::from(histovec[count - 1]))
        .sum();
    if observed == 0 {
        return None;
    }
    let fitted_error: f64 = (start..=end)
        .map(|count| fit.component_heights(count)[0])
        .sum();
    let fraction = fitted_error / observed as f64;
    fraction.is_finite().then_some(fraction)
}

fn evaluate_floor<F>(floor: u8, histovec: &[u32], sample_scale: f64, fit: &mut F) -> FloorCandidate
where
    F: FnMut(&[u32], &SpectrumEstimate, f64) -> Option<SpectrumFit>,
{
    let estimate = SpectrumEstimate::by_valley(histovec);
    let fit_attempted = estimate.verdict.is_ok();
    let fitted = fit_attempted
        .then(|| fit(histovec, &estimate, sample_scale))
        .flatten();
    let error_tail_fraction = fitted
        .and_then(|fit| fit_error_tail_fraction(histovec, &fit, estimate.guarded_min_count(&fit)));
    log::info!(
        "Quality-floor fit assessment: floor={floor} empirical_peak={} empirical_verdict={:?} fit_usable={} error_tail_fraction={:?}",
        estimate.genomic_peak,
        estimate.verdict,
        fitted.is_some(),
        error_tail_fraction
    );
    FloorCandidate {
        floor,
        estimate,
        fit: fitted,
        fit_attempted,
        error_tail_fraction,
    }
}

fn select_usable_fit(candidates: &[FloorCandidate]) -> Option<usize> {
    let usable: Vec<usize> = candidates
        .iter()
        .enumerate()
        .filter_map(|(index, candidate)| candidate.fit.map(|_| index))
        .collect();
    match usable.len() {
        0 => None,
        1 => Some(usable[0]),
        _ => {
            let acceptable: Vec<usize> = usable
                .iter()
                .copied()
                .filter(|&index| {
                    candidates[index]
                        .error_tail_fraction
                        .is_some_and(|fraction| fraction <= MAX_FIT_ERROR_TAIL_FRACTION)
                })
                .collect();
            match acceptable.len() {
                1 => Some(acceptable[0]),
                2.. => acceptable
                    .into_iter()
                    .max_by_key(|&index| candidates[index].floor),
                _ => usable.into_iter().max_by_key(|&index| {
                    (
                        candidates[index].fit.expect("usable fit").primary_mode(),
                        candidates[index].floor,
                    )
                }),
            }
        }
    }
}

/// MAIN METHOD FOR SETTING MIN_COUNT AND FLOOR
fn choose_min_count_and_floor_with_debug<F>(
    histovec: &[u32],
    strict_scale: f64,
    sketch: &SpectrumSketch,
    floors: &[u8],
    mut fit: F,
    floor_has_kmers: &[bool],
    debug_all_floors: bool,
) -> (
    u16,
    u8,
    PeakSource,
    Option<SpectrumPlotDiagnostics>,
    Vec<FloorDiagnostic>,
)
where
    F: FnMut(&[u32], &SpectrumEstimate, f64) -> Option<SpectrumFit>,
{
    let strict_index = floors.len() - 1;
    let strict_floor = floors[strict_index];
    let strict_present = floor_has_kmers[strict_index];
    let strict = evaluate_floor(strict_floor, histovec, strict_scale, &mut fit);
    strict.estimate.log_spectrum(histovec);
    let strict_needs_review = !strict_present
        || strict.fit.is_none()
        || strict
            .error_tail_fraction
            .is_none_or(|fraction| fraction > MAX_FIT_ERROR_TAIL_FRACTION)
        || strict.estimate.genomic_peak < LOW_COVERAGE_FLOOR_REVIEW;

    let evaluate_other_floors = strict_needs_review || debug_all_floors;
    if strict_needs_review {
        log::info!("Strict quality floor needs review (unusable fit, excessive fitted error tail, or peak below 10); checking lower quality floors.");
    } else if debug_all_floors {
        log::info!("Debug quality-floor mode: evaluating other floors for diagnostics only; the selected floor will be decided from the strict-floor result.");
    }

    // At the whole library's scale, as the strict spectrum already is: fitted on the raw sample, the
    // hole guard would strand 1/fraction genome k-mers instead of one.
    let spectra = (strict_index > 0).then(|| sketch.rescaled_spectra(floors.len()));
    // Lower floors exist only in the sketch: each observed k-mer stands for 1/fraction of them.
    let sketch_scale = 1.0 / sketch.fraction();
    let mut evaluated: Vec<Option<FloorCandidate>> = vec![None; floors.len()];
    let mut candidates = vec![strict];
    let mut diagnostic_floors = Vec::new();
    if debug_all_floors {
        diagnostic_floors.push(FloorDiagnostic {
            floor: strict_floor,
            spectrum: histovec.to_vec(),
            estimate: strict.estimate,
            fit: strict.fit,
            fit_attempted: strict.fit_attempted,
        });
    }

    if let Some(spectra) = spectra.as_ref().filter(|_| evaluate_other_floors) {
        for index in (0..strict_index).rev() {
            if !floor_has_kmers[index] {
                log::info!(
                    "No k-mer windows passed the candidate base-quality floor of {}; skipping it.",
                    floors[index]
                );
                continue;
            }
            let candidate = evaluate_floor(floors[index], &spectra[index], sketch_scale, &mut fit);
            evaluated[index] = Some(candidate);
            let estimate = &candidate.estimate;
            log::info!("Sketch at a base-quality floor of {}: valley={} genomic_peak={} gp_to_v_ratio={:.2} \
                     min_count={} distinct_above={} method={:?} verdict={:?}",
                    candidate.floor,
                    estimate.valley,
                    estimate.genomic_peak,
                    estimate.gp_to_v_ratio,
                    estimate.min_count,
                    if estimate.genomic_peak > 0 { distinct_above(&spectra[index], estimate.valley) } else { 0 },
                    estimate.peak_search,
                    estimate.verdict);
            if debug_all_floors {
                diagnostic_floors.push(FloorDiagnostic {
                    floor: candidate.floor,
                    spectrum: spectra[index].clone(),
                    estimate: candidate.estimate,
                    fit: candidate.fit,
                    fit_attempted: candidate.fit_attempted,
                });
            }
            if strict_needs_review {
                candidates.push(candidate);
            }
        }
    }

    let selected_index = if let Some(index) = select_usable_fit(&candidates) {
        Some(index)
    } else {
        // No fit survived. Prefer accepted primary empirical valley/peak pairs, then accepted local
        // fallbacks; in either class the largest peak on the count axis is the least starved option.
        let primary = candidates
            .iter()
            .enumerate()
            .filter(|(_, candidate)| {
                candidate.estimate.verdict.is_ok()
                    && candidate.estimate.peak_search != PeakSearch::RawLocalFallback
            })
            .max_by_key(|(_, candidate)| (candidate.estimate.genomic_peak, candidate.floor))
            .map(|(index, _)| index);
        primary.or_else(|| {
            candidates
                .iter()
                .enumerate()
                .filter(|(_, candidate)| {
                    candidate.estimate.verdict.is_ok()
                        && candidate.estimate.peak_search == PeakSearch::RawLocalFallback
                })
                .max_by_key(|(_, candidate)| (candidate.estimate.genomic_peak, candidate.floor))
                .map(|(index, _)| index)
        })
    };

    let Some(selected_index) = selected_index else {
        let (min_count, floor, peak, diagnostics) = unresolved(floors, floor_has_kmers, histovec);
        return (min_count, floor, peak, diagnostics, diagnostic_floors);
    };
    let mut selected = candidates[selected_index];
    // Step down while the current peak is starved and the next lower floor's peak clears
    // [`STEP_DOWN_GAIN`], never trading a usable fit for none.
    if let Some(spectra) = &spectra {
        let mut index = floors
            .iter()
            .position(|&floor| floor == selected.floor)
            .unwrap_or(0);
        while index > 0
            && floor_has_kmers[index - 1]
            && selected.estimate.genomic_peak < LOW_COVERAGE_FLOOR_REVIEW
        {
            let lower = index - 1;
            let peak = evaluated[lower].map_or_else(
                || SpectrumEstimate::by_valley(&spectra[lower]),
                |candidate| candidate.estimate,
            );
            if !peak.verdict.is_ok()
                || (peak.genomic_peak as f64)
                    < STEP_DOWN_GAIN * selected.estimate.genomic_peak as f64
            {
                break;
            }
            let candidate = evaluated[lower].unwrap_or_else(|| {
                evaluate_floor(floors[lower], &spectra[lower], sketch_scale, &mut fit)
            });
            if candidate.fit.is_none() && selected.fit.is_some() {
                break;
            }
            log::info!(
                "Stepping down from base-quality floor {} (peak {}) to {} (peak {}): at least \
                     {STEP_DOWN_GAIN:.2}x was required.",
                selected.floor,
                selected.estimate.genomic_peak,
                candidate.floor,
                candidate.estimate.genomic_peak
            );
            selected = candidate;
            index = lower;
        }
    }
    if let Some(selected_fit) = selected.fit {
        selected.estimate.apply_hole_guard(&selected_fit);
    }
    let peak = selected.fit.map_or_else(
        || selected.estimate.peak_source(histovec),
        |selected_fit| PeakSource::Fitted(selected_fit.primary_mode() as u32),
    );
    let diagnostics = (selected.floor == strict_floor).then(|| {
        SpectrumPlotDiagnostics::new(&selected.estimate, selected.fit, selected.fit_attempted)
    });
    log::info!("Quality floor selected: floor={} min_count={} peak={} fit_usable={} error_tail_fraction={:?} decision_candidates={} debug_only_floor_records={}",
            selected.floor,
            selected.estimate.min_count,
            peak.value().unwrap_or_default(),
            selected.fit.is_some(),
            selected.error_tail_fraction,
            candidates.len(),
            if debug_all_floors && !strict_needs_review {
                diagnostic_floors.len().saturating_sub(1)
            } else {
                0
            });
    (
        selected.estimate.min_count,
        selected.floor,
        peak,
        diagnostics,
        diagnostic_floors,
    )
}

/// Nothing separated at any floor: drop the filter, which is the most depth available, and warn, because
/// the assembly will be fragmented whatever is chosen.
fn unresolved(
    floors: &[u8],
    floor_has_kmers: &[bool],
    histovec: &[u32],
) -> (u16, u8, PeakSource, Option<SpectrumPlotDiagnostics>) {
    let loosest_index = floor_has_kmers
        .iter()
        .take(floors.len())
        .position(|&present| present)
        .expect("empty floor ladders are rejected before quality selection");
    let loosest = floors[loosest_index];
    log::warn!("The k-mer spectrum does not separate at any candidate base-quality floor ({floors:?}), so \
             the floor will be dropped to {loosest} and {UNRESOLVED_MINCOUNT} used as the minimum \
             count. Expect a fragmented assembly. Check the k-mer spectrum histogram.");
    // Nothing separated, so there is no fitted peak to report — only the median standing in.
    (
        UNRESOLVED_MINCOUNT,
        loosest,
        occurrence_weighted_median(histovec).map_or(PeakSource::Unknown, PeakSource::Fallback),
        None,
    )
}

/// The sketch's own spectrum at the floor in force, rescaled, against the table built beside it. Free on
/// every run and the strongest check available that the sampling is unbiased.
fn check_sketch_against_table(histovec: &[u32], sketch: &SpectrumSketch, keep: usize) {
    let sampled = &sketch.spectra(keep + 1)[keep];
    let (table, sample): (u64, u64) = (
        histovec.iter().map(|&n| n as u64).sum(),
        sampled.iter().map(|&n| n as u64).sum(),
    );
    let expected = table as f64 * sketch.fraction();
    // The table size is reported alongside the strict-group total: a sketch that has emptied itself
    // used to be indistinguishable here from a library with no strict-floor k-mers.
    log::info!(
        "Sketch: {sample} distinct k-mers sampled from {table} at a hash fraction of {:.3e} \
             (expected {expected:.0}; table holds {} after {} halvings{})",
        sketch.fraction(),
        sketch.counts.len(),
        sketch.shrinks,
        if sketch.saturated { ", saturated" } else { "" }
    );
}

/// The sketch's strict-floor spectrum, scaled back to the whole library. The hole guard budgets in
/// absolute k-mers, so bin heights matter to it; the counts on the x-axis are scale-invariant.
fn rescaled_strict_spectrum(sketch: &SpectrumSketch, keep: u8) -> Vec<u32> {
    sketch
        .rescaled_spectra(keep as usize + 1)
        .pop()
        .unwrap_or_default()
}

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
                crate::post_state(&match crate::reads::pass_percent() {
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

/// Writes the full spectrum beside its plots as `count<TAB>distinct`, so a run's cutoff decision can
/// be replayed offline. The plots show the fitting window and cannot be read back.
fn write_kmer_spectrum_tsv(histovec: &[u32], out_path: &std::path::Path) {
    use std::io::Write;
    let path = out_path.with_extension("spectrum.tsv");
    let Ok(file) = std::fs::File::create(&path) else {
        log::warn!("Could not write the k-mer spectrum to {}", path.display());
        return;
    };
    let mut w = std::io::BufWriter::new(file);
    let _ = writeln!(w, "count\tdistinct");
    // `histovec[c - 1]` is the count-`c` bin. Trailing empty bins are skipped, but the saturating top
    // bin is kept when occupied: it is the only sign that counts ran off the end of the histogram.
    let last = histovec.iter().rposition(|&n| n > 0).unwrap_or(0);
    for (i, &n) in histovec[..=last].iter().enumerate() {
        let _ = writeln!(w, "{}\t{}", i + 1, n);
    }
}

/// `path` with `suffix` appended to its file stem and `extension` set; dots already in the stem stay.
fn suffixed_path(path: &std::path::Path, suffix: &str, extension: &str) -> PathBuf {
    let mut name = path.file_stem().unwrap_or_default().to_os_string();
    name.push(format!("{suffix}.{extension}"));
    path.with_file_name(name)
}

#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MarkerLineStyle {
    Solid,
    Dashed,
}

#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlotYAxis {
    /// Linear, topped at [`PEAK_Y_CAP_MULT`] times the empirical peak's bar, or at the tallest bar or
    /// curve without a peak.
    PeakCapped,
    /// Logarithmic, up to the tallest bar or curve.
    Log,
}

#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Debug)]
struct PlotMarker {
    x: f64,
    colour: RGBColor,
    label: String,
    line_style: MarkerLineStyle,
}

#[cfg(not(target_family = "wasm"))]
fn plot_markers(diagnostics: &SpectrumPlotDiagnostics, used_min_count: u16) -> Vec<PlotMarker> {
    let mut markers = Vec::new();
    if let Some(seed) = diagnostics.valley_seed {
        markers.push(PlotMarker {
            x: seed as f64,
            colour: RGBColor(125, 125, 125),
            label: format!("Valley seed = {seed}"),
            line_style: MarkerLineStyle::Dashed,
        });
    }
    if let Some(valley) = diagnostics.raw_valley {
        markers.push(PlotMarker {
            x: valley as f64,
            colour: RGBColor(125, 125, 125),
            label: format!("Raw valley candidate = {valley}"),
            line_style: MarkerLineStyle::Dashed,
        });
    }
    if let Some(peak) = diagnostics.genomic_peak_seed {
        markers.push(PlotMarker {
            x: peak as f64,
            colour: RGBColor(125, 125, 125),
            label: format!("Genomic peak seed = {peak}"),
            line_style: MarkerLineStyle::Dashed,
        });
    }
    if let Some(valley) = diagnostics.valley {
        markers.push(PlotMarker {
            x: valley as f64,
            colour: RGBColor(0, 150, 110),
            label: format!("Empirical valley = {valley}"),
            line_style: MarkerLineStyle::Solid,
        });
    }
    if let Some(peak) = diagnostics.empirical_peak {
        markers.push(PlotMarker {
            x: peak as f64,
            colour: RGBColor(0, 150, 200),
            label: format!("Empirical peak = {peak}"),
            line_style: MarkerLineStyle::Solid,
        });
    }
    if let Some(guard) = diagnostics.genomic_guard {
        markers.push(PlotMarker {
            x: f64::from(guard),
            colour: RGBColor(20, 135, 65),
            label: format!("Genomic hole guard = {guard}"),
            line_style: MarkerLineStyle::Dashed,
        });
    }
    if let Some(fit) = diagnostics.fit {
        markers.push(PlotMarker {
            x: fit.mean,
            colour: RGBColor(30, 90, 220),
            label: format!("Fitted mean = {:.1}", fit.mean),
            line_style: MarkerLineStyle::Solid,
        });
        let mode = fit.primary_mode();
        markers.push(PlotMarker {
            x: mode as f64,
            colour: RGBColor(210, 70, 170),
            label: format!("Fitted mode = {mode}"),
            line_style: MarkerLineStyle::Solid,
        });
        if let Some(crossover) = fit.crossover() {
            markers.push(PlotMarker {
                x: f64::from(crossover),
                colour: RGBColor(120, 70, 200),
                label: format!("Error/genome crossover = {crossover}"),
                line_style: MarkerLineStyle::Solid,
            });
        }
    }
    if let Some(error_floor) = diagnostics.error_floor {
        markers.push(PlotMarker {
            x: f64::from(error_floor),
            colour: RGBColor(105, 105, 105),
            label: format!("Enforced 100% error floor = {error_floor}"),
            line_style: MarkerLineStyle::Dashed,
        });
    }
    // Draw this last: the cutoff is the operative decision and must remain visible when it coincides
    // with an empirical or diagnostic marker.
    markers.push(PlotMarker {
        x: f64::from(used_min_count),
        colour: BLACK,
        label: format!("Minimum count used = {used_min_count}"),
        line_style: MarkerLineStyle::Solid,
    });
    markers
}

#[cfg(not(target_family = "wasm"))]
fn plot_x_end(
    spectrum_len: usize,
    diagnostics: &SpectrumPlotDiagnostics,
    used_min_count: u16,
) -> usize {
    let histogram_end = spectrum_len.saturating_sub(1).max(1);
    let fit_end = diagnostics
        .fit
        .map_or_else(
            || {
                diagnostics
                    .empirical_peak
                    .map(|peak| crate::spectrum_fitter::fit_window_end(spectrum_len, peak))
            },
            |fit| Some(fit.fit_window_end),
        )
        .unwrap_or(MIN_PLOT_RANGE);
    let marker_end = plot_markers(diagnostics, used_min_count)
        .iter()
        .map(|marker| marker.x.ceil() as usize)
        .max()
        .unwrap_or(0);
    fit_end
        .max(marker_end)
        .max(MIN_PLOT_RANGE)
        .min(histogram_end)
}

#[cfg(not(target_family = "wasm"))]
fn plot_y_max(
    decision_spectrum: &[u32],
    raw_bloom_spectrum: Option<&[u32]>,
    diagnostics: &SpectrumPlotDiagnostics,
    plot_end: usize,
) -> f64 {
    let decision_max = decision_spectrum
        .iter()
        .take(plot_end)
        .copied()
        .max()
        .unwrap_or(0);
    let raw_max = raw_bloom_spectrum
        .into_iter()
        .flat_map(|spectrum| spectrum.iter().take(plot_end))
        .copied()
        .max()
        .unwrap_or(0);
    let fit_max = diagnostics.fit.map_or(0.0, |fit| {
        (1..=plot_end.min(fit.fit_window_end))
            .map(|count| fit.component_heights(count).iter().sum::<f64>())
            .fold(0.0, f64::max)
    });

    f64::from(decision_max).max(f64::from(raw_max)).max(fit_max)
}

#[cfg(not(target_family = "wasm"))]
fn transform_plot_y(value: f64, y_axis: PlotYAxis) -> f64 {
    match y_axis {
        PlotYAxis::PeakCapped => value,
        // A half-count baseline keeps count-1 bins visible. Zero-height bars remain invisible at
        // that baseline, and zero-valued line points are omitted from logarithmic series.
        PlotYAxis::Log => value.max(0.5).log10(),
    }
}

#[cfg(not(target_family = "wasm"))]
fn plot_axis_y_min(y_axis: PlotYAxis) -> f64 {
    match y_axis {
        PlotYAxis::PeakCapped => 0.0,
        PlotYAxis::Log => 0.5f64.log10(),
    }
}

#[cfg(not(target_family = "wasm"))]
fn plot_axis_y_max(
    y_axis: PlotYAxis,
    decision_spectrum: &[u32],
    raw_bloom_spectrum: Option<&[u32]>,
    diagnostics: &SpectrumPlotDiagnostics,
    plot_end: usize,
) -> f64 {
    let tallest = plot_y_max(decision_spectrum, raw_bloom_spectrum, diagnostics, plot_end) * 1.05;
    match y_axis {
        PlotYAxis::PeakCapped => diagnostics
            .empirical_peak
            .and_then(|peak| decision_spectrum.get(peak - 1))
            .map_or(tallest, |&height| {
                tallest.min(PEAK_Y_CAP_MULT * f64::from(height))
            })
            .max(1.0),
        PlotYAxis::Log => transform_plot_y(tallest.max(10.0), y_axis),
    }
}

/// Draw the spectrum used for the decision and all diagnostics available from its mixture fit. Bloom
/// counting additionally shows its biased raw count map; the principal histogram remains the rescaled
/// sketch that the estimator and fit actually read.
#[cfg(not(target_family = "wasm"))]
fn draw_kmer_histogram<DB: DrawingBackend>(
    root: DrawingArea<DB, Shift>,
    decision_spectrum: &[u32],
    raw_bloom_spectrum: Option<&[u32]>,
    diagnostics: &SpectrumPlotDiagnostics,
    used_min_count: u16,
    y_axis: PlotYAxis,
) where
    DB::ErrorType: 'static,
{
    let markers = plot_markers(diagnostics, used_min_count);
    let plot_end = plot_x_end(decision_spectrum.len(), diagnostics, used_min_count);
    let y_max = plot_axis_y_max(
        y_axis,
        decision_spectrum,
        raw_bloom_spectrum,
        diagnostics,
        plot_end,
    );
    let mut notes: Vec<String> = markers
        .iter()
        .filter(|marker| marker.x > plot_end as f64)
        .map(|marker| format!("{} (off-scale)", marker.label))
        .collect();
    if diagnostics.fit_attempted && diagnostics.fit.is_none() {
        notes.push("mixture fit unavailable".to_string());
    }
    let plot_kind = match y_axis {
        PlotYAxis::PeakCapped => "k-mer spectrum",
        PlotYAxis::Log => "k-mer spectrum (logarithmic Y)",
    };
    let caption = if notes.is_empty() {
        plot_kind.to_string()
    } else {
        format!("{plot_kind} — {}", notes.join(", "))
    };

    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(35)
        .y_label_area_size(65)
        .margin(5)
        .caption(caption, 24.0)
        .build_cartesian_2d(0.0f64..plot_end as f64, plot_axis_y_min(y_axis)..y_max)
        .unwrap();
    chart
        .configure_mesh()
        .disable_x_mesh()
        .x_labels(15)
        .y_labels(11)
        .max_light_lines(2)
        .light_line_style(RGBColor(220, 220, 220).mix(0.35))
        .bold_line_style(RGBColor(180, 180, 180).mix(0.4))
        .y_label_formatter(&|y| match y_axis {
            PlotYAxis::PeakCapped => format!("{y:.0}"),
            PlotYAxis::Log => {
                let count = 10.0f64.powf(*y);
                if count < 1.0 {
                    "0.5".to_string()
                } else {
                    format!("{count:.0}")
                }
            }
        })
        .draw()
        .unwrap();

    let shown = &decision_spectrum[..plot_end.min(decision_spectrum.len())];
    chart
        .draw_series(shown.iter().enumerate().map(|(i, &height)| {
            let count = (i + 1) as f64;
            let height = transform_plot_y(f64::from(height), y_axis);
            Rectangle::new(
                [
                    (count - 0.5, plot_axis_y_min(y_axis)),
                    (count + 0.5, height),
                ],
                RED.mix(0.42).filled(),
            )
        }))
        .unwrap()
        .label(if raw_bloom_spectrum.is_some() {
            "Rescaled sketch used for fitting"
        } else {
            "Observed spectrum"
        })
        .legend(|(x, y)| Rectangle::new([(x, y - 4), (x + 16, y + 4)], RED.mix(0.42).filled()));

    if let Some(raw) = raw_bloom_spectrum {
        chart
            .draw_series(LineSeries::new(
                raw.iter()
                    .take(plot_end)
                    .enumerate()
                    .filter(|&(_, &height)| height > 0)
                    .map(|(i, &height)| {
                        ((i + 1) as f64, transform_plot_y(f64::from(height), y_axis))
                    }),
                RGBColor(95, 95, 95).stroke_width(1),
            ))
            .unwrap()
            .label("Raw Bloom count map")
            .legend(|(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 16, y)],
                    RGBColor(95, 95, 95).stroke_width(1),
                )
            });
    }

    if let Some(fit) = diagnostics.fit {
        let components: Vec<(f64, [f64; 3])> = (1..=plot_end.min(fit.fit_window_end))
            .map(|count| (count as f64, fit.component_heights(count)))
            .collect();

        chart
            .draw_series(LineSeries::new(
                components
                    .iter()
                    .filter(|(_, values)| values.iter().sum::<f64>() > 0.0)
                    .map(|(count, values)| (*count, transform_plot_y(values.iter().sum(), y_axis))),
                BLACK.stroke_width(2),
            ))
            .unwrap()
            .label("Fitted mixture")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 16, y)], BLACK.stroke_width(2)));
        chart
            .draw_series(DashedLineSeries::new(
                components
                    .iter()
                    .filter(|(_, values)| values[0] > 0.0)
                    .map(|(count, values)| (*count, transform_plot_y(values[0], y_axis))),
                6,
                5,
                RGBColor(235, 145, 0).stroke_width(1),
            ))
            .unwrap()
            .label(format!("{} error component", fit.error_model))
            .legend(|(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 16, y)],
                    RGBColor(235, 145, 0).stroke_width(1),
                )
            });
        chart
            .draw_series(DashedLineSeries::new(
                components
                    .iter()
                    .filter(|(_, values)| values[1] > 0.0)
                    .map(|(count, values)| (*count, transform_plot_y(values[1], y_axis))),
                6,
                5,
                RGBColor(30, 90, 220).stroke_width(1),
            ))
            .unwrap()
            .label(format!("Single-copy ({})", fit.genome_model))
            .legend(|(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 16, y)],
                    RGBColor(30, 90, 220).stroke_width(1),
                )
            });
        if fit.shelf.is_some() {
            chart
                .draw_series(DashedLineSeries::new(
                    (1..=plot_end.min(fit.fit_window_end))
                        .filter_map(|count| Some((count as f64, fit.shelf_height(count)?)))
                        .filter(|&(_, height)| height > 0.0)
                        .map(|(count, height)| (count, transform_plot_y(height, y_axis))),
                    3,
                    3,
                    RGBColor(30, 90, 220).stroke_width(1),
                ))
                .unwrap()
                .label("Shelf (beta-binomial)")
                .legend(|(x, y)| {
                    PathElement::new(
                        vec![(x, y), (x + 16, y)],
                        RGBColor(30, 90, 220).stroke_width(1),
                    )
                });
        }
        chart
            .draw_series(DashedLineSeries::new(
                components
                    .iter()
                    .filter(|(_, values)| values[2] > 0.0)
                    .map(|(count, values)| (*count, transform_plot_y(values[2], y_axis))),
                2,
                5,
                RGBColor(100, 100, 100).stroke_width(1),
            ))
            .unwrap()
            .label(format!(
                "Two-copy ({})",
                crate::spectrum_fitter::GenomeModel::NegativeBinomial
            ))
            .legend(|(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 16, y)],
                    RGBColor(100, 100, 100).stroke_width(1),
                )
            });
    }

    for marker in markers {
        if marker.x > plot_end as f64 {
            continue;
        }
        let annotation = match marker.line_style {
            MarkerLineStyle::Solid => chart.draw_series(LineSeries::new(
                [(marker.x, plot_axis_y_min(y_axis)), (marker.x, y_max)],
                marker.colour.stroke_width(1),
            )),
            MarkerLineStyle::Dashed => chart.draw_series(DashedLineSeries::new(
                [(marker.x, plot_axis_y_min(y_axis)), (marker.x, y_max)],
                6,
                5,
                marker.colour.stroke_width(1),
            )),
        }
        .unwrap();
        let colour = marker.colour;
        annotation.label(marker.label).legend(move |(x, y)| {
            PathElement::new(vec![(x, y), (x + 16, y)], colour.stroke_width(1))
        });
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.82))
        .border_style(BLACK)
        .draw()
        .unwrap();

    let (plot_x, plot_y) = chart.plotting_area().get_pixel_range();
    drop(chart);
    let axis_style = 15.0.into_text_style(&root);
    let y_title_y = plot_y.start + (plot_y.end - plot_y.start) / 4;
    root.draw(&Text::new(
        "Counts",
        (plot_x.start - 52, y_title_y),
        axis_style
            .clone()
            .transform(FontTransform::Rotate270)
            .pos(Pos::new(HPos::Center, VPos::Center)),
    ))
    .unwrap();
    root.draw(&Text::new(
        "k-mer frequency",
        (plot_x.end - 1, plot_y.end + 30),
        axis_style.pos(Pos::new(HPos::Right, VPos::Top)),
    ))
    .unwrap();
    root.present()
        .expect("Unable to write result to file. Does the output folder exist?");
}

/// The browser plots the spectrum on the page; `out_path` is always `None` there.
#[cfg(target_family = "wasm")]
fn plot_kmer_histogram(
    _decision_spectrum: &[u32],
    _raw_bloom_spectrum: Option<&[u32]>,
    _diagnostics: &SpectrumPlotDiagnostics,
    _used_min_count: u16,
    _out_path: &std::path::Path,
) {
}

#[cfg(not(target_family = "wasm"))]
fn plot_kmer_histogram(
    decision_spectrum: &[u32],
    raw_bloom_spectrum: Option<&[u32]>,
    diagnostics: &SpectrumPlotDiagnostics,
    used_min_count: u16,
    out_path: &std::path::Path,
) {
    // Register before constructing any text style; with Plotters' `ttf` feature disabled,
    // `ab_glyph` reads this bundled font instead of querying system fonts.
    static REGISTER_FONT: std::sync::Once = std::sync::Once::new();
    REGISTER_FONT.call_once(|| {
        assert!(
            plotters::style::register_font(
                "sans-serif",
                FontStyle::Normal,
                include_bytes!("../assets/fonts/IBMPlexSans-Regular.ttf"),
            )
            .is_ok(),
            "Bundled IBM Plex Sans font is invalid"
        );
    });

    // `out_path` is the linear plot; the logarithmic one takes a `_log` suffix. Each is written as PNG
    // and as SVG.
    for (y_axis, png_path) in [
        (PlotYAxis::PeakCapped, out_path.to_path_buf()),
        (PlotYAxis::Log, suffixed_path(out_path, "_log", "png")),
    ] {
        draw_kmer_histogram(
            BitMapBackend::new(&png_path, (1280, 960)).into_drawing_area(),
            decision_spectrum,
            raw_bloom_spectrum,
            diagnostics,
            used_min_count,
            y_axis,
        );
        let svg_path = png_path.with_extension("svg");
        draw_kmer_histogram(
            SVGBackend::new(&svg_path, (1280, 960)).into_drawing_area(),
            decision_spectrum,
            raw_bloom_spectrum,
            diagnostics,
            used_min_count,
            y_axis,
        );
    }
}

/// The constants of one counting pass: fixed for every record, so they travel together.
#[derive(Clone, Copy)]
struct KmerWalk<'a> {
    k: usize,
    min_qual: u8,
    floors: Option<&'a [u8]>,
    /// Group index of the strict floor: a k-mer tagged below this does not enter the count-map.
    keep: u8,
}

/// Walk one record's k-mers, handing each to `emit` as `(hc, hnc, b, group, packed)`. `groups` is the
/// caller's scratch, reused per record. `floors = Some(ladder)` tags each k-mer with the strictest
/// floor it clears; `None` filters at the iterator and tags everything `keep`.
#[inline]
fn for_each_kmer<IntT, F>(
    seq: &[u8],
    qual_bytes: Option<&[u8]>,
    w: KmerWalk,
    groups: &mut Vec<u8>,
    mut emit: F,
) where
    IntT: for<'a> UInt<'a>,
    F: FnMut(u64, u64, u8, u8, Option<IntT>),
{
    // The scratch is shared across records, so a FASTA record must not inherit the previous one's tags.
    groups.clear();
    if let (Some(floors), Some(qual)) = (w.floors, qual_bytes) {
        window_groups(seq, qual, w.k, floors, groups);
    }
    // Every emitted window must clear at least one rung. Automatic ladders start at zero, while an
    // explicit quality floor is a single rung that must already apply on this first pass.
    let min_qual = match w.floors {
        Some(floors) if !groups.is_empty() => floors[0],
        _ => w.min_qual,
    };
    let kmer_opt = Kmer::<IntT>::new(
        std::borrow::Cow::Borrowed(seq),
        seq.len(),
        qual_bytes,
        w.k,
        min_qual,
        true,
    );
    if let Some(mut kmer_it) = kmer_opt {
        let (mut hc, mut hnc, mut b) = kmer_it.get_hashes_and_bases();
        loop {
            // A record with no qualities (FASTA) has nothing to classify: keep every k-mer.
            let g = if groups.is_empty() {
                w.keep
            } else {
                groups[kmer_it.end_index()]
            };
            debug_assert_ne!(g, NONE, "the iterator emitted a window no floor clears");
            // The packed k-mer is the costly half at k>=51, so build it only for what is kept.
            emit(hc, hnc, b, g, (g >= w.keep).then(|| kmer_it.get_kmer()));
            match kmer_it.get_next_hash_and_bases() {
                Some(next) => (hc, hnc, b) = next,
                None => break,
            }
        }
    }
}

/// One distinct k-mer, accumulated while counting in bulk.
///
/// `hnc`, `b` and `km` describe the *k-mer*, not the *occurrence*, so they are recorded once, on first
/// sight, rather than once per occurrence as the old sorting vector did.
struct KmerInfo<IntT> {
    count: u32,
    hnc: u64,
    b: u8,
    km: IntT,
}

type CountMap<IntT> = HashMap<u64, KmerInfo<IntT>, BuildHasherDefault<NoHashHasher<u64>>>;

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
#[allow(clippy::too_many_arguments)]
fn count_batch<IntT>(
    batch: &[OwnedRecord],
    k: usize,
    qual: &QualOpts,
    floors: Option<&[u8]>,
    keep: u8,
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
    let w = KmerWalk {
        k,
        min_qual: qual.min_qual,
        floors,
        keep,
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
            for (seq, qual_bytes) in recs {
                for_each_kmer::<IntT, _>(
                    seq,
                    qual_bytes.as_deref(),
                    w,
                    groups,
                    |hc, hnc, b, g, km| {
                        if g != NONE {
                            for present in floor_has_kmers.iter_mut().take(usize::from(g) + 1) {
                                *present = true;
                            }
                        }
                        if hc < thr {
                            cand.push((hc, g));
                        }
                        if g >= keep {
                            buckets[shard_of(hc, n_shards)].push((
                                hc,
                                hnc,
                                b,
                                km.expect("a kept k-mer carries its bits"),
                            ));
                        }
                    },
                );
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
fn count_kmers<IntT, I>(
    input_iters: &mut [I],
    k: usize,
    qual: &QualOpts,
    floors: Option<&[u8]>,
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
    let total = extract_kmers_from_files_batched(input_iters, BATCH_RECORDS, |batch| {
        count_batch(
            batch,
            k,
            qual,
            floors,
            keep,
            sketch.as_mut(),
            &mut states,
            &mut shards,
            &mut blooms,
            &mut t,
        );
        t.batches += 1;
    });
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
fn countmaps_into_indexed_kmers<IntT>(shards: Vec<CountMap<IntT>>, minc: u16) -> IndexedKmers<IntT>
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
fn carry_contig_kmers<IntT>(
    shards: &mut [CountMap<IntT>],
    contigs: &[Vec<u8>],
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
        floors: None,
        keep: 0,
    };
    let mut rest = contigs;
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
                for_each_kmer::<IntT, _>(seq, None, w, &mut groups, |hc, hnc, b, _, km| {
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

/// A recount refits the whole spectrum at the floor the sketch chose. The refit wins unless the
/// sketch's lobes separated there and the refit's did not, so a recount never loses a resolved cutoff.
fn keep_refit_or_prior(
    refit: (u16, PeakSource),
    prior: Option<(u16, PeakSource)>,
) -> (u16, PeakSource) {
    let Some(prior) = prior else { return refit };
    let separated = |peak: PeakSource| matches!(peak, PeakSource::Fitted(_));
    if separated(prior.1) && !separated(refit.1) {
        log::warn!(
            "The refit at the selected floor did not separate the lobes; keeping the sketch's \
             minimum count {} and peak {:?}.",
            prior.0,
            prior.1
        );
        return prior;
    }
    log::info!(
        "Refit at the selected floor: minimum count {} -> {}, peak {:?} -> {:?}",
        prior.0,
        refit.0,
        prior.1,
        refit.1
    );
    refit
}

/// Choose, filter and plot from a finished sharded count-map.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn choose_cutoff_and_filter<IntT>(
    mut shards: Vec<CountMap<IntT>>,
    k: usize,
    qual: &QualOpts,
    floors: Option<&[u8]>,
    floor_has_kmers: &[bool; MAX_GROUPS],
    sketch: Option<SpectrumSketch>,
    do_fit: bool,
    do_bloom: bool,
    carried: Option<&[Vec<u8>]>,
    out_path: &mut Option<PathBuf>,
    debug_quality_floors: bool,
    prior: Option<(u16, PeakSource)>,
) -> Result<(IndexedKmers<IntT>, Vec<u32>, u16, u8, PeakSource), PreprocessingError>
where
    IntT: for<'a> UInt<'a>,
{
    let checked_floors = floors.map_or_else(|| vec![qual.min_qual], <[u8]>::to_vec);
    let floor_count = checked_floors.len();
    let any_floor_has_kmers = floor_has_kmers
        .iter()
        .take(floor_count)
        .any(|&present| present);
    if !any_floor_has_kmers {
        return Err(PreprocessingError(format!(
            "no k-mer windows passed any candidate base-quality floor {checked_floors:?} at k={k}"
        )));
    }
    if !do_fit {
        let strict_index = floors.map_or(0, |values| values.len() - 1);
        if !floor_has_kmers[strict_index] {
            return Err(PreprocessingError(format!(
                "no k-mer windows passed the requested base-quality floor {} at k={k}",
                checked_floors[strict_index]
            )));
        }
    }

    // The slowest shard paces the parallel counting, so report the balance rather than assume it: an
    // imbalanced `shard_of` costs parallelism silently, which is what a naive high-bit selector did.
    let sizes: Vec<usize> = shards.iter().map(|s| s.len()).collect();
    let total: usize = sizes.iter().sum();
    log::debug!("Number of distinct kmers BEFORE cleaning: {total:?}");
    if total > 0 {
        let worst = *sizes.iter().max().unwrap() as f64 / (total as f64 / sizes.len() as f64);
        log::info!("Count-map shard balance: busiest shard {worst:.2}x ideal");
        log::debug!("Count-map shard sizes: {sizes:?}");
        if worst > 1.5 {
            log::warn!(
                "Count-map shards are badly imbalanced (busiest {worst:.2}x ideal). Parallel \
                 counting is limited by the busiest shard, so this costs speed."
            );
        }
    }

    // Every distinct k-mer is held with its exact count, singletons included, so this is the same
    // spectrum the old sort counter produced: count everything, histogram, choose, then filter.
    let mut histovec = vec![0_u32; MAXSIZEHISTO];
    for shard in &shards {
        for info in shard.values() {
            add_to_histogram(&mut histovec, info.count);
        }
    }

    // A Bloom table has no count-1 bin and false positives inflate the rest, so its own histogram is
    // only drawn; every decision is read from the sketch, which is exact on its subsample.
    let sketch_spectrum;
    // Library k-mers per observed one in `spectrum`, so the fit counts the sample it really has.
    let spectrum_scale;
    let spectrum: &[u32] = if do_bloom {
        let keep = floors.map_or(0u8, |f| (f.len() - 1) as u8);
        let bloom_sketch = sketch.as_ref().expect("the Bloom path always sketches");
        sketch_spectrum = rescaled_strict_spectrum(bloom_sketch, keep);
        spectrum_scale = 1.0 / bloom_sketch.fraction();
        &sketch_spectrum
    } else {
        spectrum_scale = 1.0;
        &histovec
    };

    let minc;
    // The floor the spectrum asks for. Only the fitting path can ask for a looser one.
    let mut chosen_min_qual = qual.min_qual;
    // Single-copy coverage of the spectrum this very map is filtered against.
    let genomic_peak;
    let plot_diagnostics;
    let mut debug_floor_diagnostics = Vec::new();
    if do_fit {
        log::info!("Counting finished. Choosing the minimum count...");
        crate::post_state("preprocess:bulk:fitting");
        match (&sketch, floors) {
            (Some(sketch), Some(floors)) => {
                // The table is a bin short under Bloom counting, so the comparison would be noise.
                if !do_bloom {
                    check_sketch_against_table(&histovec, sketch, floors.len() - 1);
                }
                (
                    minc,
                    chosen_min_qual,
                    genomic_peak,
                    plot_diagnostics,
                    debug_floor_diagnostics,
                ) = choose_min_count_and_floor_with_debug(
                    spectrum,
                    spectrum_scale,
                    sketch,
                    floors,
                    fit_and_log,
                    floor_has_kmers,
                    debug_quality_floors,
                );
            }
            _ => {
                let (refit_minc, refit_peak, diagnostics) =
                    choose_min_count_and_peak(spectrum, spectrum_scale);
                (minc, genomic_peak) = keep_refit_or_prior((refit_minc, refit_peak), prior);
                plot_diagnostics = Some(diagnostics);
            }
        }
        if let Some(base_path) = out_path.as_deref() {
            for diagnostic in &debug_floor_diagnostics {
                let mut estimate = diagnostic.estimate;
                if let Some(fit) = diagnostic.fit {
                    estimate.apply_hole_guard(&fit);
                }
                let diagnostics = SpectrumPlotDiagnostics::new(
                    &estimate,
                    diagnostic.fit,
                    diagnostic.fit_attempted,
                );
                let path = suffixed_path(base_path, &format!("_q{}", diagnostic.floor), "png");
                plot_kmer_histogram(
                    &diagnostic.spectrum,
                    do_bloom.then_some(histovec.as_slice()),
                    &diagnostics,
                    estimate.min_count,
                    &path,
                );
                write_kmer_spectrum_tsv(&diagnostic.spectrum, &path);
            }
        }

        if chosen_min_qual < qual.min_qual {
            // The caller recounts at the looser floor, so building the maps here is wasted work and
            // wasted memory. Dropping `shards` on the way out is the whole saving.
            return Ok((
                IndexedKmers::default(),
                histovec,
                minc,
                chosen_min_qual,
                genomic_peak,
            ));
        }
        log::info!("Minimum count chosen: {minc}. Starting filtering...");
    } else {
        minc = qual.min_count;
        // `--min-count` fixes the cutoff, not the coverage. The spectrum above spans every distinct
        // k-mer including singletons, so it is the same one the fitting path reads.
        let estimate = SpectrumEstimate::by_valley(spectrum);
        genomic_peak = estimate.peak_source(spectrum);
        plot_diagnostics = Some(SpectrumPlotDiagnostics::new(&estimate, None, false));
    }
    log::info!("Single-copy coverage read from the spectrum: {genomic_peak:?}");

    // Only now: the reads alone set the cutoff, and the carried k-mers must not have to meet it unaided.
    if let Some(carried) = carried {
        let absent = carry_contig_kmers(&mut shards, carried, k, minc);
        log::info!(
            "Carried {} contig(s) into k={k}; {absent} of their k-mers were not in the reads",
            carried.len()
        );
    }
    crate::post_state("preprocess:bulk:filtering");
    let kmers = countmaps_into_indexed_kmers::<IntT>(shards, minc);

    if let Some(p) = out_path {
        let diagnostics = plot_diagnostics.unwrap_or_default();
        plot_kmer_histogram(
            spectrum,
            do_bloom.then_some(histovec.as_slice()),
            &diagnostics,
            minc,
            p.as_path(),
        );
        write_kmer_spectrum_tsv(&histovec, p.as_path());
    }
    // After the plot and TSV, so a k that leaves nothing still says why.
    if kmers.len() == 0 {
        return Err(PreprocessingError(format!(
            "no k-mers survived filtering at k={k}, base-quality floor {chosen_min_qual}, and minimum count {minc}"
        )));
    }
    Ok((kmers, histovec, minc, chosen_min_qual, genomic_peak))
}

/// Read the records, count their k-mers, choose the cutoff and filter by it. `carried`: a multi-k
/// step's previous contigs, put in once the cutoff is chosen. `debug_quality_floors` also evaluates
/// and plots every floor; `prior`: a recount's selection from the sketch, see [`keep_refit_or_prior`].
#[allow(clippy::too_many_arguments)]
pub fn preprocessing<IntT, I>(
    input_iters: &mut [I],
    k: usize,
    qual: &QualOpts,
    floors: Option<&[u8]>,
    timevec: &mut Option<&mut Vec<Instant>>,
    out_path: &mut Option<PathBuf>,
    do_bloom: bool,
    do_fit: bool,
    carried: Option<&[Vec<u8>]>,
    debug_quality_floors: bool,
    prior: Option<(u16, PeakSource)>,
) -> Result<PreprocessedK<IntT>, PreprocessingError>
where
    IntT: for<'a> UInt<'a>,
    I: Iterator<Item = OwnedRecord> + MaybeSend,
{
    crate::post_state("preprocess:bulk:start");
    log::info!("Starting preprocessing with k = {k}");
    log::info!("Counting k-mers into a hash map, without sorting");
    let (shards, sketch, floor_has_kmers) =
        count_kmers::<IntT, _>(input_iters, k, qual, floors, do_bloom, do_fit);
    let (kmers, histovec, used_min_count, chosen_min_qual, genomic_peak) =
        choose_cutoff_and_filter::<IntT>(
            shards,
            k,
            qual,
            floors,
            &floor_has_kmers,
            sketch,
            do_fit,
            do_bloom,
            carried,
            out_path,
            debug_quality_floors,
            prior,
        )?;

    if let Some(timevec) = timevec.as_mut() {
        timevec.push(Instant::now());
        log::info!(
            "k-mers extracted, counted and filtered in {} s",
            timevec
                .last()
                .unwrap()
                .duration_since(*timevec.get(timevec.len().wrapping_sub(2)).unwrap())
                .as_secs()
        );
    }
    log::info!("Minimum count per k-mer used: {used_min_count}");

    let min_count_unresolved = do_fit
        && used_min_count == UNRESOLVED_MINCOUNT
        && !matches!(genomic_peak, PeakSource::Fitted(_));
    Ok(PreprocessedK {
        k,
        kmers,
        histovec,
        used_min_count,
        chosen_min_qual,
        genomic_peak,
        min_count_unresolved,
    })
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
    #[cfg(not(target_family = "wasm"))]
    use crate::spectrum_fitter::{ErrorModel, ErrorParams, GenomeModel};
    use std::{collections::HashMap, hash::BuildHasherDefault};

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn plot_paths_keep_dots_in_the_output_prefix() {
        let base = std::path::Path::new("out/s.v2_kmerspectrum_k21.png");
        assert_eq!(
            suffixed_path(base, "_q20", "png"),
            std::path::Path::new("out/s.v2_kmerspectrum_k21_q20.png")
        );
        assert_eq!(
            suffixed_path(base, "_log", "png"),
            std::path::Path::new("out/s.v2_kmerspectrum_k21_log.png")
        );
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn native_diagnostic_plot_renders_exact_and_bloom_spectra() {
        let mut decision = vec![0u32; MAXSIZEHISTO];
        for count in 1..=570 {
            decision[count - 1] = if count < 20 {
                100_000 / count as u32
            } else {
                let distance = count.abs_diff(95) as f64;
                (8_000.0 * (-distance * distance / 450.0).exp()) as u32
            };
        }
        let mut raw_bloom = decision.clone();
        raw_bloom[0] = 0;
        let estimate = SpectrumEstimate {
            valley: 20,
            genomic_peak: 95,
            min_count: 12,
            ..SpectrumEstimate::default()
        };
        let fit = SpectrumFit::for_test(
            ErrorModel::SingletonPareto,
            GenomeModel::NegativeBinomial,
            [0.7, 0.29, 0.01],
            95.0,
            6.6,
            ErrorParams {
                singleton_probability: 0.85,
                tail_exponent: 2.0,
                weibull_shape: None,
            },
            10.0e6,
            570,
        );
        let diagnostics = SpectrumPlotDiagnostics::new(&estimate, Some(fit), true);

        assert_eq!(
            plot_axis_y_max(PlotYAxis::PeakCapped, &decision, None, &diagnostics, 570),
            PEAK_Y_CAP_MULT * f64::from(decision[94]),
            "the linear plot is capped at three times the empirical peak's bar"
        );
        let no_peak = SpectrumPlotDiagnostics::default();
        assert_eq!(
            plot_axis_y_max(PlotYAxis::PeakCapped, &decision, None, &no_peak, 570),
            plot_y_max(&decision, None, &no_peak, 570) * 1.05,
            "without an empirical peak the linear plot is unbounded"
        );

        for (suffix, raw) in [("exact", None), ("bloom", Some(raw_bloom.as_slice()))] {
            let path = std::env::temp_dir().join(format!(
                "sparrowhawk-spectrum-{suffix}-{}.png",
                std::process::id()
            ));
            let svg_path = path.with_extension("svg");
            let log_path = suffixed_path(&path, "_log", "png");
            let log_svg_path = log_path.with_extension("svg");
            plot_kmer_histogram(&decision, raw, &diagnostics, 12, &path);
            for written in [&path, &svg_path, &log_path, &log_svg_path] {
                assert!(std::fs::metadata(written).is_ok_and(|metadata| metadata.len() > 0));
            }
            let svg = std::fs::read_to_string(&svg_path).expect("read spectrum SVG");
            assert!(svg.contains("<svg"));
            assert!(svg.contains("Counts"));
            assert!(svg.contains("k-mer frequency"));
            assert!(svg.contains("k-mer spectrum"));
            assert!(!svg.contains("logarithmic"));
            assert!(svg.contains("Minimum count used = 12"));
            let log_svg = std::fs::read_to_string(&log_svg_path).expect("read logarithmic SVG");
            assert!(log_svg.contains("k-mer spectrum (logarithmic Y)"));
            for written in [path, svg_path, log_path, log_svg_path] {
                std::fs::remove_file(written).expect("remove temporary spectrum plot");
            }
        }
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn active_error_floor_remains_a_hard_minimum() {
        let fit = SpectrumFit::for_test(
            ErrorModel::SingletonPareto,
            GenomeModel::NegativeBinomial,
            [0.9, 0.09, 0.01],
            100.0,
            6.0,
            ErrorParams {
                singleton_probability: 0.85,
                tail_exponent: 2.0,
                weibull_shape: None,
            },
            10.0e6,
            570,
        );
        let mut estimate = SpectrumEstimate {
            min_count: 2,
            verdict: Verdict::NoPeakAboveValley,
            ..SpectrumEstimate::default()
        };
        estimate.apply_hole_guard(&fit);
        assert_eq!(
            estimate.min_count,
            fit.error_floor_cutoff().max(2),
            "the full error floor is a hard minimum"
        );
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn empty_counts_at_every_candidate_floor_are_an_error() {
        let qual = QualOpts {
            min_count: 2,
            min_qual: 25,
        };
        let floors = [0u8, 11, 25];
        let mut iters = [Vec::<OwnedRecord>::new().into_iter()];
        let (shards, sketch, floor_has_kmers) =
            count_kmers::<u64, _>(&mut iters, 31, &qual, Some(&floors), false, true);
        let error = choose_cutoff_and_filter::<u64>(
            shards,
            31,
            &qual,
            Some(&floors),
            &floor_has_kmers,
            sketch,
            true,
            false,
            None,
            &mut None,
            false,
            None,
        )
        .err()
        .expect("empty input must stop before assembly");
        assert!(error.to_string().contains("no k-mer windows passed any"));
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn explicit_min_count_does_not_relax_an_empty_strict_floor() {
        let qual = QualOpts {
            min_count: 2,
            min_qual: 25,
        };
        let floors = [0u8, 11, 25];
        let reads = deep_library(2, 20);
        let mut iters = [reads.into_iter()];
        let (shards, sketch, floor_has_kmers) =
            count_kmers::<u64, _>(&mut iters, 31, &qual, Some(&floors), false, true);
        assert!(!floor_has_kmers[2]);
        assert!(floor_has_kmers[1]);
        let error = choose_cutoff_and_filter::<u64>(
            shards,
            31,
            &qual,
            Some(&floors),
            &floor_has_kmers,
            sketch,
            false,
            false,
            None,
            &mut None,
            false,
            None,
        )
        .err()
        .expect("explicit quality floor must not silently loosen");
        assert!(error
            .to_string()
            .contains("requested base-quality floor 25"));
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn an_empty_sketch_does_not_override_exact_floor_presence() {
        let qual = QualOpts {
            min_count: 1,
            min_qual: 25,
        };
        let floors = [0u8, 11, 25];
        let mut shard: CountMap<u64> = HashMap::with_hasher(BuildHasherDefault::default());
        shard.insert(
            11,
            KmerInfo {
                count: 3,
                hnc: 19,
                b: 3,
                km: 23,
            },
        );
        let (kmers, _, _, _, _) = choose_cutoff_and_filter::<u64>(
            vec![shard],
            31,
            &qual,
            Some(&floors),
            &[true; MAX_GROUPS],
            Some(SpectrumSketch::new()),
            false,
            false,
            None,
            &mut None,
            false,
            None,
        )
        .expect("the sketch sample being empty is not proof that the floor has no k-mers");
        assert_eq!(kmers.len(), 1);
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn no_kmers_surviving_the_count_cutoff_is_an_error() {
        let qual = QualOpts {
            min_count: 2,
            min_qual: 20,
        };
        let mut shard: CountMap<u64> = HashMap::with_hasher(BuildHasherDefault::default());
        shard.insert(
            11,
            KmerInfo {
                count: 1,
                hnc: 19,
                b: 3,
                km: 23,
            },
        );
        let error = choose_cutoff_and_filter::<u64>(
            vec![shard],
            31,
            &qual,
            None,
            &[true; MAX_GROUPS],
            None,
            false,
            false,
            None,
            &mut None,
            false,
            None,
        )
        .err()
        .expect("do not continue to graph assembly without surviving k-mers");
        assert!(error.to_string().contains("no k-mers survived filtering"));
    }

    #[cfg(not(target_family = "wasm"))]
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

    /// Reads for the counter tests. Rotating the backbone keeps most k-mers shared across replicates
    /// while making the seam k-mers unique, so the spectrum has a singleton lobe as well as a deep one.
    /// Each read is mostly Q37 with one dip through Q11 to Q2, moved along by replicate: the dip has to
    /// be short, or no window of k consecutive bases ever clears the strict floor.
    #[cfg(not(target_family = "wasm"))]
    fn counter_test_reads() -> Vec<OwnedRecord> {
        const BACKBONE: &[u8] = b"ACGTACGTAACGGTTACGATCGATTACGGCATCAGGTACAGGTTACAGGATCAGGTACA";
        (0..8u8)
            .map(|rep| {
                let mut seq = BACKBONE.to_vec();
                seq.rotate_left(rep as usize * 3);
                let dip = (rep as usize * 7) % seq.len();
                let qual: Vec<u8> = (0..seq.len())
                    .map(|i| {
                        33 + if (dip..dip + 2).contains(&i) {
                            2u8
                        } else if (dip + 2..dip + 5).contains(&i) {
                            11
                        } else {
                            37
                        }
                    })
                    .collect();
                (seq, Some(qual))
            })
            .collect()
    }

    /// A fixed floor is a ladder of one rung: counting under it keeps exactly what filtering at that
    /// floor keeps, and no emitted window is tagged NONE.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn a_fixed_floor_counts_only_what_clears_it() {
        let reads = counter_test_reads();
        let qual = QualOpts {
            min_count: 2,
            min_qual: 20,
        };
        let counts = |floors: Option<&[u8]>| -> HashMap<u64, u32> {
            count_kmers::<u64, _>(
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
    #[cfg(not(target_family = "wasm"))]
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
            count_kmers::<u64, _>(&mut iters, k, &qual, Some(&ladder[..]), false, true);

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

    /// True occurrence count per k-mer above the strict floor, from the independent reference hasher.
    #[cfg(not(target_family = "wasm"))]
    fn true_counts(
        reads: &[OwnedRecord],
        k: usize,
        min_qual: u8,
        ladder: &[u8],
    ) -> HashMap<u64, u32> {
        let keep = (ladder.len() - 1) as u8;
        let mut want: HashMap<u64, u32> = HashMap::default();
        for (hc, _, _, g, _) in hash_batch::<u64>(reads, k, min_qual, Some(ladder), keep) {
            if g >= keep {
                *want.entry(hc).or_insert(0) += 1;
            }
        }
        want
    }

    #[cfg(not(target_family = "wasm"))]
    fn bloom_shards(
        reads: Vec<OwnedRecord>,
        k: usize,
        qual: &QualOpts,
        ladder: &[u8],
    ) -> HashMap<u64, u32> {
        let mut iters = [reads.into_iter()];
        let (shards, _, _) = count_kmers::<u64, _>(&mut iters, k, qual, Some(ladder), true, true);
        let mut got: HashMap<u64, u32> = HashMap::default();
        for shard in &shards {
            for (hc, info) in shard {
                got.insert(*hc, info.count);
            }
        }
        got
    }

    /// What the sharded filter guarantees: it eats a k-mer's first sighting, so a survivor's count is
    /// its true count, or one more when a false positive let that first sighting through.
    #[cfg(not(target_family = "wasm"))]
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

    /// Canonical-minimum hashes for the sharding tests: `min(fwd, rc)` has a triangular density, which
    /// is exactly the bias `shard_of` has to mix away.
    #[cfg(not(target_family = "wasm"))]
    fn canonical_minimum_hashes(n: u64) -> impl Iterator<Item = u64> {
        (0..n).map(|i| {
            let f = i.wrapping_mul(0x2545_F491_4F6C_DD1D);
            let r = (!i).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            f.min(r)
        })
    }

    /// `shard_of` must spread evenly at every shard count we can pick, since the busiest shard paces
    /// the counting. Sharding on the raw high bits measured a 33x spread, making 16 shards act like 8.
    #[cfg(not(target_family = "wasm"))]
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

    // ---------------------------------------------------------------------------------------------
    // The valley estimator. Spectra are built from the same model the plan was simulated with: a
    // genome lobe at Poisson(lambda) over 4.6 Mb, errors at Poisson(lambda * p / 3) over 3k slots.
    // ---------------------------------------------------------------------------------------------

    fn poisson(k: usize, lam: f64) -> f64 {
        (-lam + k as f64 * lam.ln() - libm::lgamma(k as f64 + 1.0)).exp()
    }

    /// A synthetic spectrum at k-mer coverage `lam`, with a 1 % per-base error rate at k = 51.
    fn synthetic_spectrum(lam: f64) -> Vec<u32> {
        const GENOME: f64 = 4_600_000.0;
        const K: f64 = 51.0;
        let mut h = vec![0u32; MAXSIZEHISTO];
        let lam_err = lam * 0.01 / 3.0;
        let err_slots = GENOME * 3.0 * K;
        for c in 1..MAXSIZEHISTO {
            let n = GENOME * poisson(c, lam)
                + if c < 60 {
                    err_slots * poisson(c, lam_err)
                } else {
                    0.0
                };
            h[c - 1] = n as u32;
        }
        h
    }

    #[test]
    fn raw_first_search_finds_a_known_error_valley_and_genome_peak() {
        let mut histogram = vec![0u32; 128];
        for count in 1usize..=9 {
            histogram[count - 1] = 1_000 / count as u32;
        }
        histogram[9] = 1; // count 10: the valley
        histogram[10] = 2;
        histogram[11] = 3;
        histogram[12] = 4; // three bins not below the minimum confirm the seed
        for count in 14..=65 {
            histogram[count - 1] = 5;
        }
        for count in 31usize..=49 {
            let distance = count.abs_diff(40) as f64;
            histogram[count - 1] = (200.0 * (-0.5 * distance * distance / 25.0).exp()) as u32;
        }

        let estimate = SpectrumEstimate::by_valley(&histogram);
        assert_eq!(estimate.valley_seed, 10);
        assert_eq!(estimate.raw_valley, Some(10));
        assert_eq!(estimate.raw_genomic_peak, Some(40));
        assert_eq!(estimate.peak_search, PeakSearch::SmoothedRefinement);
        assert!((10..=13).contains(&estimate.valley));
        assert!((38..=42).contains(&estimate.genomic_peak));
        assert!(estimate.verdict.is_ok());
        assert_eq!(estimate.valley_n, histogram[estimate.valley - 1]);
        assert_eq!(
            estimate.genomic_peak_n,
            histogram[estimate.genomic_peak - 1]
        );
    }

    #[test]
    fn peak_refinement_cannot_jump_to_a_broad_mode_beyond_the_raw_anchor() {
        let mut histogram = vec![0u32; 128];
        for count in 1usize..=9 {
            histogram[count - 1] = 1_000 / count as u32;
        }
        histogram[9] = 1;
        histogram[10] = 2;
        histogram[11] = 3;
        histogram[12] = 4;
        for count in 14..=70 {
            histogram[count - 1] = 5;
        }
        for count in 31usize..=49 {
            let distance = count.abs_diff(40) as f64;
            histogram[count - 1] = (200.0 * (-0.5 * distance * distance / 25.0).exp()) as u32;
        }
        // This broad later shoulder dominates smoothing, but no individual bin exceeds the raw peak.
        for count in 58..=68 {
            histogram[count - 1] = 199;
        }

        let smooth = smooth_histogram(&histogram);
        let global_smooth_peak = (20..histogram.len() - 1)
            .max_by_key(|&count| smooth[count - 1])
            .expect("non-empty search range");
        assert!(
            global_smooth_peak > 52,
            "fixture should fool a global smooth search"
        );

        let estimate = SpectrumEstimate::by_valley(&histogram);
        assert_eq!(estimate.raw_genomic_peak, Some(40));
        assert!(
            estimate.genomic_peak <= 52,
            "selected peak escaped the 1.3× bound"
        );
        assert!((38..=42).contains(&estimate.genomic_peak));
    }

    #[test]
    fn srr26931854_20x_k81_rejects_the_weak_empirical_area() {
        // Captured from this real sample's exact-count (no-Bloom) spectrum. Raw-first candidates and
        // the anchored refinement stay at the earlier shoulder, which still correctly fails the
        // support guard, so the spectrum is not trusted.
        let observed = [
            683_875, 350_296, 360_089, 310_003, 235_672, 163_282, 105_683, 64_315, 38_723, 21_767,
            12_405, 7_342, 4_149, 2_454, 1_551, 1_014, 785, 584, 484, 395, 342, 305, 279, 249, 256,
            233, 199, 141, 86, 98, 60, 71, 60, 63, 45, 36, 20, 13, 16, 17, 16, 19, 14, 12, 11, 11,
            6, 5, 4, 3, 5, 1, 2, 5, 1, 3, 0, 4, 2, 4, 4, 4, 4, 2, 0, 1, 4, 1,
        ];
        let mut histogram = vec![0u32; MAXSIZEHISTO];
        histogram[..observed.len()].copy_from_slice(&observed);

        let estimate = SpectrumEstimate::by_valley(&histogram);
        assert_eq!(estimate.raw_valley, Some(31));
        assert_eq!(estimate.raw_genomic_peak, Some(32));
        assert_eq!(
            SpectrumEstimate::default().find_primary_pair(&histogram),
            Ok((32, 33, PeakSearch::SmoothedRefinement))
        );
        assert!(
            PeakAreaSupport::measure(&histogram, 32, 33).is_some_and(|support| !support.passes())
        );
        assert_ne!(estimate.verdict, Verdict::Ok);
        assert_eq!((estimate.valley, estimate.genomic_peak), (0, 0));
        assert_eq!(estimate.min_count, UNRESOLVED_MINCOUNT);
    }

    fn spectrum_with_valley_and_peak(valley: usize, peak: usize, len: usize) -> Vec<u32> {
        let mut histogram = vec![0u32; len];
        histogram[0] = 20_000;
        for count in 2..valley {
            histogram[count - 1] = 100 + (valley - count) as u32;
        }
        histogram[valley - 1] = 1;
        histogram[valley] = 2;
        histogram[valley + 1] = 3;
        histogram[valley + 2] = 4;
        for count in (valley + 4)..=peak {
            histogram[count - 1] = 10;
        }
        histogram[peak - 1] = 1_000;
        histogram
    }

    #[test]
    fn raw_low_coverage_bypass_has_strict_boundaries() {
        let nine_count_gap = spectrum_with_valley_and_peak(5, 14, 64);
        let estimate = SpectrumEstimate::by_valley(&nine_count_gap);
        assert_eq!(estimate.raw_valley, Some(5));
        assert_eq!(estimate.raw_genomic_peak, Some(14));
        assert_eq!(estimate.peak_search, PeakSearch::RawLowCoverage);
        assert_eq!((estimate.valley, estimate.genomic_peak), (5, 14));

        // Exactly ten counts apart does not bypass smoothing.
        let ten_count_gap = spectrum_with_valley_and_peak(5, 15, 64);
        let estimate = SpectrumEstimate::by_valley(&ten_count_gap);
        assert_eq!(estimate.raw_genomic_peak, Some(15));
        assert_eq!(estimate.peak_search, PeakSearch::SmoothedRefinement);

        // Both positions must be below 15; equality at either boundary takes the refinement path.
        let peak_at_15 = spectrum_with_valley_and_peak(5, 15, 64);
        assert_eq!(
            SpectrumEstimate::by_valley(&peak_at_15).peak_search,
            PeakSearch::SmoothedRefinement
        );
        let valley_at_15 = spectrum_with_valley_and_peak(15, 24, 64);
        assert_eq!(
            SpectrumEstimate::by_valley(&valley_at_15).peak_search,
            PeakSearch::SmoothedRefinement
        );
    }

    #[test]
    fn refinement_uses_anchored_inclusive_ranges_and_deterministic_ties() {
        let mut smooth = vec![100u64; 128];
        let mut observed = vec![0u32; 128];
        smooth[19..=25].fill(10);
        smooth[26..=38].fill(30);
        observed[37..=39].fill(1);
        smooth[38] = 100; // count 39 = floor(1.3 × 30)
        smooth[39] = 1_000; // count 40 is outside the inclusive upper bound
        assert_eq!(refine_smoothed_valley(&smooth, 22, 30), Some(22));
        assert_eq!(
            refine_smoothed_peak(&observed, &smooth, 20, 30, 22),
            Some(39)
        );

        smooth[37] = 100; // count 38 ties, but is closer to the raw peak
        assert_eq!(
            refine_smoothed_peak(&observed, &smooth, 20, 30, 22),
            Some(38),
            "equally high bins choose the one nearest the raw peak"
        );
    }

    #[test]
    fn failed_refinement_rejects_an_unmeasurable_mirrored_area() {
        // The raw peak is near the histogram end, so its mirrored area reaches past the final
        // unsaturated bin. The estimator must fail closed instead of treating a clipped area as support.
        let mut histogram = vec![99u32; 19];
        histogram[0] = 20_000;
        histogram[1..6].copy_from_slice(&[150, 130, 120, 110, 100]);
        histogram[6] = 1; // raw valley at 7
        histogram[15] = 2;
        histogram[16] = 2;
        histogram[17] = 100; // raw peak at 18; bin 19 is the saturated bin

        let estimate = SpectrumEstimate::by_valley(&histogram);
        assert_eq!(estimate.raw_valley, Some(7));
        assert_eq!(estimate.raw_genomic_peak, Some(18));
        assert_eq!(PeakAreaSupport::measure(&histogram, 7, 18), None);
        assert_eq!(estimate.peak_search, PeakSearch::Unresolved);
        assert_eq!(estimate.min_count, UNRESOLVED_MINCOUNT);
    }

    #[test]
    fn raw_search_rejects_a_monotone_spectrum_and_saturated_only_peak() {
        let mut monotone = vec![0u32; 64];
        for count in 2..64 {
            monotone[count - 1] = (64 - count) as u32;
        }
        assert_eq!(find_raw_valley_seed(&monotone), None);
        let estimate = SpectrumEstimate::by_valley(&monotone);
        assert_eq!(estimate.peak_search, PeakSearch::Unresolved);
        assert_eq!(estimate.min_count, UNRESOLVED_MINCOUNT);

        let mut saturated = spectrum_with_valley_and_peak(10, 20, 64);
        saturated[63] = 1_000_000;
        assert_eq!(find_raw_genomic_peak(&saturated, 10), Some(20));
    }

    #[test]
    fn mirrored_peak_area_excludes_singletons_and_accepts_exactly_five_percent() {
        let mut histogram = vec![0u32; 12];
        histogram[0] = 1_000_000; // singletons are excluded from the denominator
        histogram[1] = 95; // count 2, outside the valley-to-mirrored-shoulder area
        histogram[2..7].fill(1); // counts 3..=7: exactly 5 of 100 non-singleton bins

        let support = PeakAreaSupport::measure(&histogram, 3, 5).unwrap();
        assert_eq!(support.area_sum, 5);
        assert_eq!(support.denominator, 100);
        assert_eq!(support.fraction(), MIN_PEAK_AREA_FRACTION);
        assert!(support.passes(), "the boundary is inclusive");

        histogram[1] += 1;
        let support = PeakAreaSupport::measure(&histogram, 3, 5).unwrap();
        assert_eq!(support.area_sum, 5);
        assert_eq!(support.denominator, 101);
        assert!(!support.passes(), "just below 5 % must fail");
    }

    #[test]
    fn mirrored_peak_area_rejects_an_endpoint_in_the_saturated_bin() {
        let histogram = vec![1, 4, 3, 2, 8, 9, 5, 2, 1];
        assert_eq!(
            PeakAreaSupport::measure(&histogram, 2, 6),
            None,
            "the right endpoint is count 10, beyond the last unsaturated count 8"
        );
    }

    #[test]
    fn raw_local_fallback_allows_count_two_and_checks_two_right_tail_declines() {
        let histogram = vec![10, 1, 2, 3, 20, 10, 5, 1, 0];
        assert_eq!(find_first_raw_local_valley(&histogram), Some(2));
        let peak = find_raw_peak_after_valley(&histogram, 2).unwrap();
        assert_eq!(peak, 5);
        assert!(confirms_two_bin_right_tail(&histogram, peak));
        assert!(PeakAreaSupport::measure(&histogram, 2, peak)
            .unwrap()
            .passes());

        let mut non_decreasing_tail = histogram;
        non_decreasing_tail[7] = 10; // makes the final overlapping average equal to the previous
        assert!(!confirms_two_bin_right_tail(&non_decreasing_tail, peak));
    }

    #[test]
    fn fallback_with_a_non_decreasing_tail_is_unresolved_and_publishes_no_pair() {
        let mut histogram = vec![0u32; 12];
        histogram[..4].copy_from_slice(&[10, 2, 3, 20]);
        histogram[5] = 1; // count 6 makes the last two-bin average stop decreasing

        assert_eq!(find_raw_valley_seed(&histogram), None);
        assert_eq!(find_first_raw_local_valley(&histogram), Some(2));
        assert_eq!(find_raw_peak_after_valley(&histogram, 2), Some(4));
        assert!(!confirms_two_bin_right_tail(&histogram, 4));
        let estimate = SpectrumEstimate::by_valley(&histogram);
        assert_eq!(estimate.verdict, Verdict::PeakTailNotDecreasing);
        assert_eq!((estimate.valley, estimate.genomic_peak), (0, 0));
        assert_eq!(estimate.min_count, UNRESOLVED_MINCOUNT);
    }

    #[test]
    fn supported_raw_local_fallback_resolves_when_no_seed_was_found() {
        let mut histogram = vec![0u32; 12];
        histogram[..5].copy_from_slice(&[10, 2, 3, 20, 1]);

        assert_eq!(find_raw_valley_seed(&histogram), None);
        assert_eq!(find_first_raw_local_valley(&histogram), Some(2));
        assert_eq!(find_raw_peak_after_valley(&histogram, 2), Some(4));
        assert!(confirms_two_bin_right_tail(&histogram, 4));
        assert!(PeakAreaSupport::measure(&histogram, 2, 4).is_some_and(PeakAreaSupport::passes));
        let estimate = SpectrumEstimate::by_valley(&histogram);
        assert_eq!(estimate.peak_search, PeakSearch::RawLocalFallback);
        assert_eq!((estimate.valley, estimate.genomic_peak), (2, 4));
        assert!(estimate.verdict.is_ok());
    }

    /// Fraction of a Poisson(`lam`) genome deleted by cutting below `min_count`.
    fn genome_loss(lam: f64, min_count: u16) -> f64 {
        (0..min_count as usize).map(|c| poisson(c, lam)).sum()
    }

    /// At extreme depth the genome lobe is real but under 5 % of the non-singleton bins, so the area rule
    /// refuses it: no pair is published, nothing is fitted, and the cutoff is the sentinel.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn deep_error_dominated_spectra_are_refused_by_the_area_guard() {
        let deepest = synthetic_spectrum(500.0);
        let distinct_total: u64 = deepest.iter().map(|&n| u64::from(n)).sum();
        let genome_distinct: u64 = deepest[100..].iter().map(|&n| u64::from(n)).sum();
        assert!(
            genome_distinct * 50 < distinct_total,
            "the genome lobe is under 2 % of distinct-bin mass"
        );

        for lam in [250.0, 500.0] {
            let h = synthetic_spectrum(lam);
            let e = SpectrumEstimate::by_valley(&h);
            assert_eq!(e.verdict, Verdict::NoLocalValley, "lambda {lam}");
            assert_eq!(
                (e.valley, e.genomic_peak, e.min_count),
                (0, 0, UNRESOLVED_MINCOUNT),
                "lambda {lam}"
            );

            let (min_count, peak, diagnostics) = choose_min_count_and_peak(&h, 1.0);
            assert_eq!(min_count, UNRESOLVED_MINCOUNT, "lambda {lam}");
            assert!(matches!(peak, PeakSource::Fallback(_)), "lambda {lam}");
            assert!(
                !diagnostics.fit_attempted,
                "lambda {lam}: an unresolved pair must not be fitted"
            );
        }
    }

    /// Up to 5x the lobes merge and the cutoff stays at the sentinel; at 6-7x they resolve at the observed
    /// modal count; at 8x the cutoff still must not move; by 9x it resolves inside the genome-loss budget.
    #[test]
    fn low_coverage_spectra_resolve_only_once_the_lobes_separate() {
        for lam in [3.0, 4.0, 5.0] {
            let e = SpectrumEstimate::by_valley(&synthetic_spectrum(lam));
            assert_ne!(e.verdict, Verdict::Ok, "lambda {lam}");
            assert_eq!(e.min_count, UNRESOLVED_MINCOUNT, "lambda {lam}");
        }
        for lam in [6.0, 7.0] {
            let e = SpectrumEstimate::by_valley(&synthetic_spectrum(lam));
            assert_eq!(e.verdict, Verdict::Ok, "lambda {lam}");
            assert!(
                e.genomic_peak.abs_diff(lam as usize) <= 1,
                "lambda {lam}: peak {}",
                e.genomic_peak
            );
        }
        assert_eq!(
            SpectrumEstimate::by_valley(&synthetic_spectrum(8.0)).min_count,
            UNRESOLVED_MINCOUNT
        );
        let e = SpectrumEstimate::by_valley(&synthetic_spectrum(9.0));
        assert_eq!(
            e.verdict,
            Verdict::Ok,
            "smoothing resolves the shallow 9x lobe"
        );
        assert!(genome_loss(9.0, e.min_count) < MAX_GENOME_LOSS);
    }

    /// A monotone spectrum has no genome lobe, so the walk runs off into the tail and locks onto a noise
    /// fluctuation. The depth of the valley is what gives it away: the two bins are within ~2 % of each
    /// other, nowhere near [`MIN_GP_TO_V_RATIO`]. Position cannot be used for this — a clean pair of narrow
    /// lobes a few counts apart is equally crowded and perfectly resolvable.
    #[test]
    fn a_monotone_spectrum_is_refused_for_having_no_lobe() {
        let mut h = vec![0u32; MAXSIZEHISTO];
        for c in 1..MAXSIZEHISTO {
            // Power-law decay with a deterministic ripple standing in for counting noise.
            let ripple = 1.0 + 0.01 * ((c % 7) as f64 - 3.0) / 3.0;
            h[c - 1] = (200_000_000.0 / (c as f64).powf(1.5) * ripple) as u32;
        }
        // Either refusal is right here — the walk may stop on the crest it locked onto, making genomic peak and
        // valley equal — but the depth is what rules it out either way.
        let e = SpectrumEstimate::by_valley(&h);
        assert_ne!(e.verdict, Verdict::Ok);
        assert!(
            e.gp_to_v_ratio < MIN_GP_TO_V_RATIO,
            "valley {} ({}) genomic_peak {} ({}) gave gp_to_v_ratio {:.3}",
            e.valley,
            e.valley_n,
            e.genomic_peak,
            e.genomic_peak_n,
            e.gp_to_v_ratio
        );
        assert_eq!(e.min_count, UNRESOLVED_MINCOUNT);
    }

    /// Two narrow lobes a few counts apart are crowded in position but perfectly separated in depth, so
    /// they must resolve. This is the case a positional guard would have thrown away.
    #[test]
    fn a_tight_but_clean_separation_still_resolves() {
        let mut h = vec![0u32; MAXSIZEHISTO];
        h[0] = 100_000; // error spike at count 1
        h[5..10].fill(50_000); // a narrow but smoothed genome lobe, with a clear empty valley
        let e = SpectrumEstimate::by_valley(&h);
        assert_eq!(
            e.verdict,
            Verdict::Ok,
            "valley {} genomic_peak {} gp_to_v_ratio {:.1} valley_to_peak_xratio {:.3}",
            e.valley,
            e.genomic_peak,
            e.gp_to_v_ratio,
            e.valley as f64 / e.genomic_peak as f64
        );
        assert!(
            e.valley as f64 / e.genomic_peak as f64 > 0.3,
            "the point of the case is that it is crowded"
        );
    }

    /// Through moderate depth the lobes separate inside the genome-loss budget, from raw candidates, and
    /// the reported heights are the observed bins rather than their five-bin averages.
    #[test]
    fn the_working_range_resolves_from_raw_bins_within_the_loss_budget() {
        let mut spike = vec![0u32; 40];
        spike[19] = 100;
        let smooth = smooth_histogram(&spike);
        assert_eq!(
            (smooth[17], smooth[19], smooth[21]),
            (20, 20, 20),
            "a spike is averaged over five bins"
        );
        assert_eq!(
            SpectrumEstimate::by_valley(&synthetic_spectrum(30.0)).peak_search,
            PeakSearch::SmoothedRefinement
        );

        for lam in [10.0, 15.0, 20.0, 50.0, 100.0] {
            let e = SpectrumEstimate::by_valley(&synthetic_spectrum(lam));
            assert_eq!(e.verdict, Verdict::Ok, "lambda {lam}");
            assert!(
                e.min_count >= 2 && genome_loss(lam, e.min_count) < MAX_GENOME_LOSS,
                "lambda {lam} cut at {}",
                e.min_count
            );
        }
        for lam in [20.0, 30.0, 60.0, 150.0] {
            let h = synthetic_spectrum(lam);
            let e = SpectrumEstimate::by_valley(&h);
            assert!(
                e.raw_valley.is_some() && e.raw_genomic_peak.is_some(),
                "lambda {lam}"
            );
            assert!(e.valley < e.genomic_peak, "lambda {lam}");
            if e.verdict.is_ok() {
                assert_eq!(
                    (e.valley_n, e.genomic_peak_n),
                    (h[e.valley - 1], h[e.genomic_peak - 1]),
                    "lambda {lam}: heights must be the observed bins"
                );
            }
        }
    }

    /// A peak beyond the old 500-bin ceiling is measurable when the histogram also extends far
    /// enough to contain its full mirrored shoulder.
    #[test]
    fn valley_estimator_finds_a_peak_beyond_the_old_ceiling() {
        let mut h = vec![0u32; 20_000];
        for c in 1..40 {
            h[c - 1] = 1_000_000 / (c as u32 * c as u32); // a decaying error lobe
        }
        for c in 5500..6500 {
            h[c - 1] = 20_000; // a genome lobe far outside LEGACY_HISTO_RANGE
        }
        let e = SpectrumEstimate::by_valley(&h);
        assert_eq!(e.verdict, Verdict::Ok);
        assert!(
            e.genomic_peak >= LEGACY_HISTO_RANGE,
            "genomic_peak was {}",
            e.genomic_peak
        );
    }

    /// What cutting at `m` costs, as a fraction of the k-mers above `valley` — the quantity the guard
    /// budgets, measured the same way from the same histogram.
    fn measured_loss(histovec: &[u32], valley: usize, m: usize) -> f64 {
        let total: f64 = (valley..histovec.len())
            .map(|c| histovec[c - 1] as f64)
            .sum();
        let cut: f64 = (valley..m).map(|c| histovec[c - 1] as f64).sum();
        if total > 0.0 {
            cut / total
        } else {
            0.0
        }
    }

    /// The measured half of the guard must spend its whole budget and no more: one bin higher has to
    /// break it.
    #[test]
    fn measured_guard_spends_its_budget_and_stops() {
        for lam in [20.0, 50.0, 100.0] {
            let h = synthetic_spectrum(lam);
            let estimate = SpectrumEstimate::by_valley(&h);
            let valley = estimate.valley;
            let genomic_peak = estimate.genomic_peak;
            let m = measured_guard(&h, valley, genomic_peak) as usize;
            assert!(m >= 2, "lambda {lam}");
            assert!(
                measured_loss(&h, valley, m) <= MAX_GENOME_LOSS,
                "lambda {lam} guard {m} cost {:.4}",
                measured_loss(&h, valley, m)
            );
            assert!(
                m + 1 >= genomic_peak || measured_loss(&h, valley, m + 1) > MAX_GENOME_LOSS,
                "lambda {lam} guard {m} could have gone higher"
            );
        }
    }

    /// Neither half of the guard is safe alone: the Poisson bound binds at low coverage, where errors above
    /// the valley loosen the measured one, and the measured bound at depth, where a lobe as wide as the
    /// real ones (mean 200, sd 45) would let a Poisson tail cut near 168.
    #[test]
    fn the_loss_guard_takes_the_tighter_of_its_two_bounds() {
        let h = synthetic_spectrum(10.0);
        let e = SpectrumEstimate::by_valley(&h);
        assert!(poisson_guard(e.genomic_peak) < measured_guard(&h, e.valley, e.genomic_peak));

        let mut wide = vec![0u32; MAXSIZEHISTO];
        for c in 1..MAXSIZEHISTO {
            let z = (c as f64 - 200.0) / 45.0;
            wide[c - 1] = (1_000_000.0 * (-0.5 * z * z).exp()) as u32;
        }
        let measured = measured_guard(&wide, 60, 200);
        assert!(measured < poisson_guard(200));
        assert!(
            measured < 150,
            "guard {measured} is no tighter than a Poisson tail"
        );
        let cost = measured_loss(&wide, 60, usize::from(measured));
        assert!(cost <= MAX_GENOME_LOSS, "guard {measured} cost {cost:.4}");
    }

    /// Dispersion is what tells us whether the mixture fit's Poisson genome component is defensible,
    /// so it has to read ~1 on a Poisson lobe and clearly above it on a wide one.
    #[test]
    fn dispersion_reads_one_on_poisson_and_more_on_a_wide_lobe() {
        let mut poisson_lobe = vec![0u32; MAXSIZEHISTO];
        for c in 1..MAXSIZEHISTO {
            poisson_lobe[c - 1] = (1e9 * poisson(c, 200.0)) as u32;
        }
        let d = dispersion_above(&poisson_lobe, 100, 200);
        assert!((0.8..1.3).contains(&d), "Poisson lobe read {d:.2}");

        let mut wide = vec![0u32; MAXSIZEHISTO];
        for c in 1..MAXSIZEHISTO {
            let z = (c as f64 - 200.0) / 45.0;
            wide[c - 1] = (1_000_000.0 * (-0.5 * z * z).exp()) as u32;
        }
        assert!(
            dispersion_above(&wide, 60, 200) > 5.0,
            "wide lobe read {:.2}",
            dispersion_above(&wide, 60, 200)
        );
    }

    // ---- the sketch -------------------------------------------------------------------------------

    /// A modest bimodal spectrum as (count, number of distinct k-mers) pairs: an error lobe decaying
    /// from count 1 and a genome lobe around `genomic_peak`.
    fn bimodal(genomic_peak: usize) -> Vec<(u32, usize)> {
        bimodal_with_genome_size(genomic_peak, 3_000.0)
    }

    fn bimodal_with_genome_size(genomic_peak: usize, genome_size: f64) -> Vec<(u32, usize)> {
        let mut out: Vec<(u32, usize)> = (1..=5).map(|c| (c as u32, 4000 / c)).collect();
        for c in (genomic_peak - 12)..=(genomic_peak + 12) {
            let d = (c as f64 - genomic_peak as f64) / 5.0;
            out.push((c as u32, (genome_size * (-0.5 * d * d).exp()) as usize));
        }
        out
    }

    /// Put a spectrum into one group of a sketch, one hash per distinct k-mer.
    fn sketch_with(group: usize, spectrum: &[(u32, usize)]) -> SpectrumSketch {
        let mut s = SpectrumSketch::new();
        let mut h = 1u64;
        for &(count, n) in spectrum {
            for _ in 0..n {
                let mut c = [0u32; MAX_GROUPS];
                c[group] = count;
                s.counts.insert(h, c);
                h += 1;
            }
        }
        s
    }

    /// The floor the sketch's subsampling rests on. If a future hash change lowers it, `HASH_FLOOR`
    /// is wrong and the sample would be silently truncated rather than emptied.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn the_canonical_hash_never_falls_below_the_floor() {
        use crate::nthash::NtHashIterator;
        const BASES: [u8; 4] = *b"ACGT";
        let mut st = 0x2545_F491_4F6C_DD1Du64;
        for k in [21usize, 31, 41, 71] {
            let mut lowest = u64::MAX;
            for _ in 0..20_000 {
                let seq: Vec<u8> = (0..k)
                    .map(|_| BASES[(xorshift(&mut st) % 4) as usize])
                    .collect();
                lowest = lowest.min(NtHashIterator::new(&seq, k, true).canonical_hash());
            }
            assert!(
                lowest >= HASH_FLOOR,
                "k={k}: saw {lowest}, below HASH_FLOOR {HASH_FLOOR}"
            );
        }
    }

    /// The bug this guards: halving the threshold itself walks it below the hash's floor, so `retain`
    /// drops every entry and, both gates being `hash < threshold`, nothing is ever admitted again.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn shrinking_halves_the_sample_and_keeps_survivors_exact() {
        let mut st = 0x9E37_79B9_7F4A_7C15u64;
        let mut sketch = SpectrumSketch::new();
        // One halving above the floor: the old code stepped from here to exactly `HASH_FLOOR`, which
        // no hash is below, so its `retain` took the whole table with it.
        sketch.threshold = 2 * HASH_FLOOR;
        for _ in 0..40_000 {
            // Drawn from the support the real hash has, which `the_canonical_hash_never_falls_below
            // _the_floor` pins down: uniform above the floor, never under it.
            let hash = HASH_FLOOR + xorshift(&mut st) % HASH_FLOOR;
            sketch.observe(hash, (hash % MAX_GROUPS as u64) as u8);
        }
        let before = sketch.counts.len();
        assert!(before > 1_000, "fixture admitted only {before}");

        sketch.shrink();
        assert!(
            !sketch.counts.is_empty(),
            "shrink erased a populated table: {before} -> 0, threshold now {}",
            sketch.threshold
        );
        let kept = sketch.counts.len() as f64 / before as f64;
        assert!(
            (0.3..0.7).contains(&kept),
            "kept {kept:.2} of the table, expected about half"
        );
        assert!(
            sketch.fraction() > 0.0,
            "a zero fraction divides by zero when rescaling"
        );
        // Survivors keep the counts they had, which is why the spectrum survives the subsampling.
        let mut exact = SpectrumSketch::new();
        let (low, high) = (HASH_FLOOR + 1, u64::MAX - 1);
        for _ in 0..5 {
            exact.observe(low, 0);
        }
        for _ in 0..9 {
            exact.observe(high, 0);
        }
        exact.shrink();
        assert_eq!(exact.counts[&low][0], 5, "a survivor must keep its count");
        assert!(
            !exact.counts.contains_key(&high),
            "a hash above the threshold must go"
        );
    }
    /// The core invariant: a sampled spectrum has the same shape as the full one, because subsampling
    /// hash space does not touch the counts of the k-mers it keeps.
    #[test]
    fn sketch_matches_a_full_count() {
        let mut full = vec![0u32; MAXSIZEHISTO];
        let mut sketch = SpectrumSketch::new();
        // Canonical hashes, i.e. min of a forward and a reverse value, so the test exercises the same
        // non-uniform distribution `fraction` has to correct for.
        for i in 0..60_000u64 {
            let fwd = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let rev = (i ^ 0x5DEE_CE66_D1B7_1234).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            let hash = fwd.min(rev);
            let count = 10 + (i % 40) as u32;
            add_to_histogram(&mut full, count);
            for _ in 0..count {
                sketch.observe(hash, 0);
            }
        }
        sketch.shrink();
        sketch.shrink();
        let sampled = &sketch.spectra(1)[0];
        let (ft, st) = (
            full[9..60].iter().sum::<u32>(),
            sampled[9..60].iter().sum::<u32>(),
        );
        assert!(st > 0, "the sample is empty");
        // The whole spectrum lives in counts 10..49 in both, and the valley/genomic peak structure is identical.
        assert_eq!(ft, 60_000);
        let rescaled = st as f64 / sketch.fraction();
        assert!(
            (rescaled / ft as f64 - 1.0).abs() < 0.1,
            "rescaled {rescaled:.0} against {ft}"
        );
    }

    /// Rescaling moves bin heights, never the counts they sit at: the hole guard reads the heights,
    /// while everything the valley estimator gates on is a ratio.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn rescaled_strict_spectrum_preserves_counts() {
        let mut sketch = sketch_with(0, &bimodal(40));
        // `sketch_with` inserts past the threshold, so halve it twice for a fraction worth scaling by.
        sketch.shrink();
        sketch.shrink();
        let raw = sketch.spectra(1).pop().unwrap();
        let scaled = rescaled_strict_spectrum(&sketch, 0);

        let (r, s) = (
            SpectrumEstimate::by_valley(&raw),
            SpectrumEstimate::by_valley(&scaled),
        );
        assert_eq!(r.genomic_peak, s.genomic_peak, "the peak moved");
        assert_eq!(r.valley, s.valley, "the valley moved");
        assert_eq!(r.min_count, s.min_count, "the cutoff moved");

        let scale = 1.0 / sketch.fraction();
        assert!(scale > 1.5, "the fraction did not fall: scale {scale}");
        for (i, (&a, &b)) in raw.iter().zip(scaled.iter()).enumerate() {
            assert_eq!(b, (a as f64 * scale).min(u32::MAX as f64) as u32, "bin {i}");
        }
        assert!(scaled[39] > raw[39], "heights must actually scale");
    }

    /// Lower floors exist only in the sketch, so they must be fitted at the whole library's scale, as
    /// the strict floor is: the hole guard budgets in absolute k-mers. The fit is told that scale, so
    /// its likelihood counts the sample rather than the rescaled bins.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn lower_floors_are_fitted_at_the_whole_library_scale() {
        let mut sketch = sketch_with(1, &bimodal(40));
        sketch.shrink();
        sketch.shrink();
        let scale = 1.0 / sketch.fraction();
        assert!(scale > 1.5, "the fraction did not fall: scale {scale}");
        let raw = sketch.spectra(3)[1].clone();
        let seen = std::cell::RefCell::new(Vec::new());
        // Every fit fails, so the strict floor is reviewed and both lower floors are fitted.
        choose_min_count_and_floor_with_fit(
            &histo(&bimodal(30)),
            &sketch,
            &[0u8, 11, 25],
            |histogram, _, sample_scale| {
                seen.borrow_mut().push((histogram.to_vec(), sample_scale));
                None
            },
            &[true; MAX_GROUPS],
        );
        let seen = seen.borrow(); // [strict, floor 11, floor 0]
        assert_eq!(seen[0].1, 1.0, "the strict spectrum here is exact");
        let (fitted, fitted_scale) = &seen[1];
        assert!(
            (fitted_scale - scale).abs() < 1e-12,
            "{fitted_scale} against {scale}"
        );
        for (i, (&a, &b)) in raw.iter().zip(fitted.iter()).enumerate() {
            assert_eq!(b, (a as f64 * scale).min(u32::MAX as f64) as u32, "bin {i}");
        }
    }

    // ---- the Bloom path ---------------------------------------------------------------------------

    #[cfg(not(target_family = "wasm"))]
    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// A deterministic genome sequenced at `depth` with 100 bp reads at a flat quality, plus a scatter
    /// of single-base errors, so the spectrum has a singleton lobe as well as a deep one.
    #[cfg(not(target_family = "wasm"))]
    fn deep_library(depth: usize, phred: u8) -> Vec<OwnedRecord> {
        const BASES: [u8; 4] = *b"ACGT";
        const GENOME: usize = 3000;
        const READ: usize = 100;
        let mut st = 0x2545_F491_4F6C_DD1Du64;
        let genome: Vec<u8> = (0..GENOME)
            .map(|_| BASES[(xorshift(&mut st) % 4) as usize])
            .collect();

        (0..GENOME * depth / READ)
            .map(|i| {
                let start = (xorshift(&mut st) as usize) % (GENOME - READ);
                let mut seq = genome[start..start + READ].to_vec();
                // One read in eight carries a substitution, which makes k-mers seen once or twice.
                if i % 8 == 0 {
                    let at = (xorshift(&mut st) as usize) % READ;
                    seq[at] = BASES[(xorshift(&mut st) % 4) as usize];
                }
                (seq, Some(vec![33 + phred; READ]))
            })
            .collect()
    }

    /// Drive the real counting pair the pipeline uses, so these tests exercise the dispatch rather
    /// than a shape of their own.
    #[cfg(not(target_family = "wasm"))]
    fn count_and_finish(
        reads: Vec<OwnedRecord>,
        k: usize,
        qual: &QualOpts,
        ladder: &[u8],
        do_bloom: bool,
    ) -> (IndexedKmers<u64>, Vec<u32>, u16, u8, PeakSource) {
        let mut iters = [reads.into_iter()];
        let (shards, sketch, floor_has_kmers) =
            count_kmers::<u64, _>(&mut iters, k, qual, Some(ladder), do_bloom, true);
        choose_cutoff_and_filter::<u64>(
            shards,
            k,
            qual,
            Some(ladder),
            &floor_has_kmers,
            sketch,
            true,
            do_bloom,
            None,
            &mut None,
            false,
            None,
        )
        .expect("fixture should leave k-mers after filtering")
    }

    /// The point of reading the spectrum from the sketch: the Bloom path now reports a usable peak,
    /// even though its own histogram has no count-1 bin to read one from.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn bloom_path_reads_its_peak_from_the_sketch() {
        let reads = deep_library(50, 37);
        let ladder = [0u8, 11, 25];
        let qual = QualOpts {
            min_count: 2,
            min_qual: 25,
        };
        let (kmers, histovec, minc, chosen_min_qual, peak) =
            count_and_finish(reads, 31, &qual, &ladder, true);

        assert_eq!(histovec[0], 0, "a Bloom histogram cannot hold singletons");
        assert!(
            matches!(peak, PeakSource::Fitted(_)),
            "the sketch separates, so the peak is fitted, not {peak:?}"
        );
        // Every base is Q37, so the strict floor keeps everything and nothing asks to loosen.
        assert_eq!(chosen_min_qual, 25);
        // A band, not a point: reads start at random offsets and the sketch subsamples, so the lobe
        // lands near the library's depth rather than on it.
        let p = peak.value().unwrap();
        assert!((25..=50).contains(&p), "peak {p} is not the library's");
        assert!(
            minc >= 2 && (minc as u32) < p,
            "cutoff {minc} against peak {p}"
        );
        assert!(kmers.len() > 2000, "only {} k-mers survived", kmers.len());
    }

    /// The ladder now reaches this path: a library whose strict floor clears nothing asks the caller
    /// to recount, and hands back no k-mers rather than the ones it would have thrown away.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn bloom_path_asks_for_a_recount_when_the_strict_floor_starves() {
        // Q20 clears the middle rung and not the strict one, so no window is tagged at floor 25.
        let reads = deep_library(50, 20);
        let ladder = [0u8, 11, 25];
        let qual = QualOpts {
            min_count: 2,
            min_qual: 25,
        };
        let (kmers, _, _, chosen_min_qual, peak) =
            count_and_finish(reads, 31, &qual, &ladder, true);

        assert_eq!(
            chosen_min_qual, 11,
            "the loose rung is the one that separates"
        );
        assert_eq!(kmers.len(), 0, "a recount pass must not build the table");
        assert!(matches!(peak, PeakSource::Fitted(_)), "got {peak:?}");
    }

    /// A k-mer moves between count bins as the floor drops; it does not appear in two bins at once.
    #[test]
    fn a_kmer_moves_between_count_bins_when_the_floor_drops() {
        let mut s = SpectrumSketch::new();
        s.counts.insert(1, [10, 0, 40]);
        let spectra = s.spectra(3);
        assert_eq!(spectra[2][39], 1, "count 40 at the strict floor");
        assert_eq!(spectra[2].iter().sum::<u32>(), 1);
        assert_eq!(spectra[0][49], 1, "count 50 with no filter");
        assert_eq!(spectra[0].iter().sum::<u32>(), 1);
    }

    // ---- the decision -----------------------------------------------------------------------------

    /// The main table does not resolve but a looser floor does, so the looser floor is returned — and
    /// the *strictest* of the ones that resolve, not the loosest.
    #[test]
    fn the_strictest_resolving_floor_wins() {
        let flat = vec![1000u32; MAXSIZEHISTO];
        // Mass in group 1 only, so floors 1 and 0 both see the bimodal spectrum and floor 2 sees nothing.
        let sketch = sketch_with(1, &bimodal(40));
        let floors = [0u8, 11, 25];
        let (minc, floor, _, _) = choose_min_count_and_floor(&flat, &sketch, &floors);
        assert_eq!(floor, 11, "should loosen to B, not all the way to C");
        assert!(
            minc > 2,
            "a resolving floor must yield a real cutoff, got {minc}"
        );
    }

    /// Build a histogram from a `(count, distinct)` spectrum, as the main table would.
    fn histo(spectrum: &[(u32, usize)]) -> Vec<u32> {
        let mut out = vec![0u32; MAXSIZEHISTO];
        for &(count, n) in spectrum {
            for _ in 0..n {
                add_to_histogram(&mut out, count);
            }
        }
        out
    }

    #[cfg(not(target_family = "wasm"))]
    fn test_fit(mean: usize, error_weight: f64) -> SpectrumFit {
        let remainder = 1.0 - error_weight;
        SpectrumFit::for_test(
            ErrorModel::SingletonPareto,
            GenomeModel::NegativeBinomial,
            [error_weight, remainder * 0.99, remainder * 0.01],
            mean as f64,
            3.0,
            ErrorParams {
                singleton_probability: 0.85,
                tail_exponent: 2.0,
                weibull_shape: None,
            },
            1.0e6,
            (mean * 6).min(MAXSIZEHISTO - 1),
        )
    }

    /// Doubling every bin doubles the distinct k-mers without moving the mode. `distinct_above` must see
    /// that and `genomic_peak` must not — the difference a depth criterion is blind to.
    #[test]
    fn distinct_above_sees_breadth_that_the_peak_does_not() {
        let narrow = histo(&bimodal(30));
        let wide: Vec<u32> = narrow.iter().map(|n| n * 2).collect();
        let (a, b) = (
            SpectrumEstimate::by_valley(&narrow),
            SpectrumEstimate::by_valley(&wide),
        );
        assert_eq!(a.genomic_peak, b.genomic_peak, "depth is unchanged");
        assert_eq!(
            distinct_above(&wide, b.valley),
            2 * distinct_above(&narrow, a.valley),
            "breadth is not"
        );
    }

    /// Floors only step down below [`LOW_COVERAGE_FLOOR_REVIEW`]: a resolved strict floor at 14 or 40 stays,
    /// however deep a lower floor is, and costs no extra fit.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn a_resolved_strict_floor_at_or_above_the_review_peak_does_not_step_down() {
        assert_eq!(
            SpectrumEstimate::by_valley(&histo(&bimodal(14))).genomic_peak,
            14
        );
        // (strict floor's peak, lower floor's peak)
        for (strict_peak, lower_peak) in [(14, 40), (40, 60), (40, 120)] {
            let strict = histo(&bimodal(strict_peak));
            assert!(
                SpectrumEstimate::by_valley(&strict).verdict.is_ok(),
                "strict peak {strict_peak}"
            );
            let calls = std::cell::Cell::new(0);
            let (_, floor, _, _) = choose_min_count_and_floor_with_fit(
                &strict,
                &sketch_with(1, &bimodal(lower_peak)),
                &[0, 11, 25],
                |_, estimate, _| {
                    calls.set(calls.get() + 1);
                    Some(test_fit(estimate.genomic_peak, 0.0))
                },
                &[true; MAX_GROUPS],
            );
            assert_eq!(
                (floor, calls.get()),
                (25, 1),
                "strict {strict_peak} against lower {lower_peak}"
            );
        }
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn a_failed_strict_floor_fit_checks_lower_quality_floors() {
        let strict_histogram = histo(&bimodal(30));
        let strict = SpectrumEstimate::by_valley(&strict_histogram);
        assert!(strict.verdict.is_ok());

        let sketch = sketch_with(1, &bimodal(40));
        let calls = std::cell::Cell::new(0);
        let (min_count, floor, _, diagnostics) = choose_min_count_and_floor_with_fit(
            &strict_histogram,
            &sketch,
            &[0u8, 11, 25],
            |_, estimate, _| {
                let call = calls.get();
                calls.set(call + 1);
                (call == 1).then(|| test_fit(estimate.genomic_peak, 0.0))
            },
            &[true; MAX_GROUPS],
        );

        assert_eq!(calls.get(), 3);
        assert_eq!(floor, 11);
        assert_ne!(min_count, UNRESOLVED_MINCOUNT);
        assert!(
            diagnostics.is_none(),
            "the winning candidate is a lower floor"
        );
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn peak_below_ten_checks_other_floors_even_if_its_fit_converges() {
        let histogram = synthetic_spectrum(9.0);
        let estimate = SpectrumEstimate::by_valley(&histogram);
        assert!(
            estimate.verdict.is_ok(),
            "fixture must yield an accepted pair"
        );
        assert!(estimate.genomic_peak < LOW_COVERAGE_FLOOR_REVIEW);
        let sketch = sketch_with(1, &bimodal(20));
        let calls = std::cell::Cell::new(0);
        let (_, floor, _, _) = choose_min_count_and_floor_with_fit(
            &histogram,
            &sketch,
            &[0, 11, 25],
            |_, estimate, _| {
                calls.set(calls.get() + 1);
                Some(test_fit(estimate.genomic_peak, 0.0))
            },
            &[true; MAX_GROUPS],
        );
        assert_eq!(calls.get(), 3, "a sub-10 peak itself triggers floor review");
        assert_eq!(floor, 11, "a sub-10 peak steps down to a deeper floor");
    }

    /// The tail is measured from the cutoff the floor would actually apply, so error k-mers that
    /// cutoff removes anyway do not count against it.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn the_error_tail_starts_at_the_applied_cutoff() {
        let histogram = synthetic_spectrum(30.0);
        let estimate = SpectrumEstimate::by_valley(&histogram);
        let fit = fit_and_log(&histogram, &estimate, 1.0).expect("the fixture fits");
        let cutoff = estimate.guarded_min_count(&fit);
        assert!(cutoff > 2, "the fixture must cut above 2, got {cutoff}");
        let applied = fit_error_tail_fraction(&histogram, &fit, cutoff).unwrap();
        let from_two = fit_error_tail_fraction(&histogram, &fit, 2).unwrap();
        assert!(applied < from_two, "{applied} should be below {from_two}");
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn stepping_down_never_trades_a_usable_fit_for_none() {
        let calls = std::cell::Cell::new(0);
        // Peak 9 against 20 would step down, but only the strict floor's fit converges.
        let (_, floor, _, _) = choose_min_count_and_floor_with_fit(
            &synthetic_spectrum(9.0),
            &sketch_with(1, &bimodal(20)),
            &[0u8, 11, 25],
            |_, estimate, _| {
                let call = calls.get();
                calls.set(call + 1);
                (call == 0).then(|| test_fit(estimate.genomic_peak, 0.0))
            },
            &[true; MAX_GROUPS],
        );
        assert_eq!(floor, 25);
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn a_recount_keeps_its_refit_unless_only_the_sketch_separated() {
        let (sep, merged) = (PeakSource::Fitted(40), PeakSource::Fallback(12));
        assert_eq!(keep_refit_or_prior((9, sep), Some((14, sep))), (9, sep));
        assert_eq!(keep_refit_or_prior((2, merged), Some((14, sep))), (14, sep));
        assert_eq!(
            keep_refit_or_prior((2, merged), Some((2, merged))),
            (2, merged)
        );
        assert_eq!(keep_refit_or_prior((9, sep), Some((2, merged))), (9, sep));
        assert_eq!(keep_refit_or_prior((9, sep), None), (9, sep));
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn diagnostic_all_floor_fits_do_not_enter_the_decision() {
        let histogram = histo(&bimodal(35));
        let sketch = sketch_with(1, &bimodal(45));
        let regular_calls = std::cell::Cell::new(0);
        let regular = choose_min_count_and_floor_with_debug(
            &histogram,
            1.0,
            &sketch,
            &[0, 11, 25],
            |_, estimate, _| {
                regular_calls.set(regular_calls.get() + 1);
                Some(test_fit(estimate.genomic_peak, 0.0))
            },
            &[true; MAX_GROUPS],
            false,
        );
        let debug_calls = std::cell::Cell::new(0);
        let debug = choose_min_count_and_floor_with_debug(
            &histogram,
            1.0,
            &sketch,
            &[0, 11, 25],
            |_, estimate, _| {
                debug_calls.set(debug_calls.get() + 1);
                Some(test_fit(estimate.genomic_peak, 0.0))
            },
            &[true; MAX_GROUPS],
            true,
        );
        assert_eq!(
            (regular.0, regular.1, regular.2.value()),
            (debug.0, debug.1, debug.2.value())
        );
        assert_eq!(
            regular_calls.get(),
            1,
            "normal mode fits only the strict floor"
        );
        assert_eq!(
            debug_calls.get(),
            3,
            "diagnostic mode fits all non-empty floors"
        );
        assert_eq!(debug.4.len(), 3, "one diagnostic plot/TSV record per floor");
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn multiple_fit_selection_applies_the_error_area_and_fallback_rules() {
        let estimate = SpectrumEstimate::by_valley(&histo(&bimodal(30)));
        let candidate = |floor, mean, fraction| FloorCandidate {
            floor,
            estimate,
            fit: Some(test_fit(mean, 0.1)),
            fit_attempted: true,
            error_tail_fraction: fraction,
        };
        let multiple = [
            candidate(25, 30, Some(0.08)),
            candidate(11, 31, Some(0.02)),
            candidate(0, 40, Some(0.04)),
        ];
        assert_eq!(select_usable_fit(&multiple), Some(1));

        let at_area_boundary = [
            candidate(20, 30, Some(MAX_FIT_ERROR_TAIL_FRACTION)),
            candidate(0, 40, Some(MAX_FIT_ERROR_TAIL_FRACTION + 0.000_001)),
        ];
        assert_eq!(select_usable_fit(&at_area_boundary), Some(0));

        let only_one = [
            FloorCandidate {
                fit: None,
                ..multiple[0]
            },
            FloorCandidate {
                error_tail_fraction: Some(0.95),
                ..multiple[1]
            },
            FloorCandidate {
                fit: None,
                ..multiple[2]
            },
        ];
        assert_eq!(select_usable_fit(&only_one), Some(1));

        let no_area_passes = [
            FloorCandidate {
                error_tail_fraction: Some(0.08),
                ..multiple[0]
            },
            FloorCandidate {
                error_tail_fraction: Some(0.07),
                ..multiple[1]
            },
            FloorCandidate {
                error_tail_fraction: Some(0.06),
                ..multiple[2]
            },
        ];
        assert_eq!(select_usable_fit(&no_area_passes), Some(2));
    }

    /// Nothing resolves anywhere, so the quality filter is dropped entirely rather than trusting an
    /// estimate that by definition did not resolve.
    #[test]
    fn nothing_resolving_drops_the_floor_to_zero() {
        let flat = vec![1000u32; MAXSIZEHISTO];
        let sketch = SpectrumSketch::new();
        let floors = [0u8, 11, 25];
        let (minc, floor, _, _) = choose_min_count_and_floor(&flat, &sketch, &floors);
        assert_eq!(
            floor, 0,
            "nothing resolved, so the floor goes to the bottom"
        );
        assert_eq!(minc, UNRESOLVED_MINCOUNT);
    }

    // ---- Single-copy coverage: fitted, and the fallback when the lobes merge ----

    /// A clean deep library must report a *fitted* peak near the true coverage.
    #[test]
    fn a_deep_library_reports_a_fitted_peak() {
        let h = synthetic_spectrum(120.0);
        match SpectrumEstimate::by_valley(&h).peak_source(&h) {
            PeakSource::Fitted(p) => assert!(
                (100..=140).contains(&p),
                "fitted peak {p} should sit near the true 120x"
            ),
            other => panic!("expected a fitted peak, got {other:?}"),
        }
    }

    /// The occurrence-weighted median names the genomic lobe even when error k-mers outnumber it, reads
    /// counts rather than indices, and is what a merged spectrum falls back to; an empty one claims nothing.
    #[test]
    fn the_occurrence_weighted_median_stands_in_for_an_unresolved_peak() {
        let mut lobes = vec![0u32; MAXSIZEHISTO];
        lobes[0] = 5_000_000; // many singleton errors, one occurrence each
        lobes[39] = 1_000_000; // fewer genomic k-mers, 40 occurrences each
        let mut one_bin = vec![0u32; MAXSIZEHISTO];
        one_bin[6] = 10;
        let empty = vec![0u32; MAXSIZEHISTO];
        // (spectrum, median count)
        for (h, want) in [(&lobes, Some(40)), (&one_bin, Some(7)), (&empty, None)] {
            assert_eq!(occurrence_weighted_median(h), want);
        }
        assert_eq!(
            SpectrumEstimate::by_valley(&empty).peak_source(&empty),
            PeakSource::Unknown
        );

        let flat = vec![1000u32; MAXSIZEHISTO];
        let estimate = SpectrumEstimate::by_valley(&flat);
        assert!(!estimate.verdict.is_ok(), "fixture must not resolve");
        assert!(matches!(
            estimate.peak_source(&flat),
            PeakSource::Fallback(_)
        ));
    }

    /// The store replays the reads exactly: at the ladder, as the first k counts, and at one floor, as
    /// the recount and every later k count.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn counting_the_read_store_matches_counting_the_reads() {
        let reads = counter_test_reads();
        let ladder = [0u8, 11, 25];
        let mut store = crate::read_store::ReadStore::new(&ladder);
        for (seq, qual) in &reads {
            store.push(seq, qual.as_deref());
        }
        store.finish();
        let counts = |shards: Vec<CountMap<u64>>| -> HashMap<u64, (u32, u64, u8)> {
            shards
                .into_iter()
                .flatten()
                .map(|(hc, i)| (hc, (i.count, i.hnc, i.b)))
                .collect()
        };
        for (floors, min_qual) in [(Some(&ladder[..]), 25u8), (None, 11), (None, 25)] {
            let qual = QualOpts {
                min_count: 2,
                min_qual,
            };
            let streamed = count_kmers::<u64, _>(
                &mut [reads.clone().into_iter()],
                11,
                &qual,
                floors,
                false,
                true,
            )
            .0;
            let replayed =
                count_kmers::<u64, _>(&mut [store.records()], 11, &qual, floors, false, true).0;
            assert_eq!(
                counts(streamed),
                counts(replayed),
                "floors={floors:?} min_qual={min_qual}"
            );
        }
    }

    /// Every carried k-mer passes the cutoff. A count the reads already clear is never touched, and
    /// carrying a contig twice changes nothing.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn carried_contig_kmers_survive_the_cutoff() {
        let (k, minc) = (11usize, 3u16);
        let reads = counter_test_reads();
        let ladder = [0u8, 11, 25];
        let qual = QualOpts {
            min_count: minc,
            min_qual: 25,
        };
        let novel = b"TTTTGGGGCCCCAAAATTTGGGCCCAAATGCATGCA".to_vec();
        let contigs = vec![novel.clone(), reads[0].0.clone(), novel];
        let records: Vec<OwnedRecord> = contigs.iter().map(|s| (s.clone(), None)).collect();
        let mut occurrences: HashMap<u64, u32> = HashMap::default();
        for (hc, ..) in hash_batch::<u64>(&records, k, 0, None, 0) {
            *occurrences.entry(hc).or_insert(0) += 1;
        }
        let as_counts = |shards: &[CountMap<u64>]| -> HashMap<u64, u32> {
            shards
                .iter()
                .flatten()
                .map(|(h, i)| (*h, i.count))
                .collect()
        };
        let (mut shards, _, _) = count_kmers::<u64, _>(
            &mut [reads.clone().into_iter()],
            k,
            &qual,
            Some(&ladder[..]),
            false,
            true,
        );
        let before = as_counts(&shards);
        let absent = carry_contig_kmers(&mut shards, &contigs, k, minc);
        let after = as_counts(&shards);
        let minc = u32::from(minc);
        assert_eq!(
            absent,
            occurrences
                .keys()
                .filter(|&h| !before.contains_key(h))
                .count()
        );
        for (h, &count) in &after {
            let read = before.get(h).copied().unwrap_or(0);
            let want = occurrences.get(h).map_or(read, |_| read.max(minc));
            assert_eq!(count, want);
        }
    }

    /// Deterministic ACGT, so a 2 kb genome has no repeated 20-mer to branch on.
    #[cfg(not(target_family = "wasm"))]
    fn pseudo_genome(len: usize) -> Vec<u8> {
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                b"ACGT"[((state >> 33) & 3) as usize]
            })
            .collect()
    }

    /// The point of multi-k: two tilings overlapping by 30 bases read every 21-mer but miss ten
    /// 41-mers, so k=41 alone breaks there, and carrying the k=21 contig joins it again. `u128`
    /// holds both k, as the pipeline would pick it for k=41.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn carrying_the_small_k_contig_bridges_a_seam_the_large_k_reads_miss() {
        use crate::algorithms::corrector::{CorrectionOpts, CoverageRef};
        use crate::graph_works::{Assemble, BasicAsm};

        let genome = pseudo_genome(2000);
        let mut reads: Vec<OwnedRecord> = Vec::new();
        for start in (0..=915).step_by(5).chain((985..=1900).step_by(5)) {
            let read = genome[start..start + 100].to_vec();
            reads.push((read.clone(), None));
            reads.push((read, None));
        }
        let qual = QualOpts {
            min_count: 2,
            min_qual: 0,
        };
        let correction = CorrectionOpts {
            do_bubble_collapse: false,
            do_dead_end_removal: false,
            pop_ratio: 0.1,
            dead_end_nts: 0,
            dead_end_coverage_nts: 0,
            dead_end_coverage_ratio: 2.0,
            coverage: CoverageRef::unknown(),
            do_ec_removal: false,
            ec_ratio: 4.0,
            ec_require_both_flanks: false,
        };
        let assemble = |k: usize, carried: Option<&[Vec<u8>]>| -> Vec<Vec<u8>> {
            let (shards, sketch, present) = count_kmers::<u128, _>(
                &mut [reads.clone().into_iter()],
                k,
                &qual,
                None,
                false,
                true,
            );
            let (mut kmers, ..) = choose_cutoff_and_filter::<u128>(
                shards, k, &qual, None, &present, sketch, false, false, carried, &mut None, false,
                None,
            )
            .expect("the fixture leaves k-mers after filtering");
            let contigs = BasicAsm::assemble::<u128>(k, &mut kmers, &mut None, None, correction);
            crate::save_functions::spell_contigs::<u128>(
                &contigs,
                |hash| kmers.get_packed(hash).copied(),
                k,
            )
        };

        let small = assemble(21, None);
        assert_eq!(small.len(), 1, "every 21-mer is read");
        assert_eq!(assemble(41, None).len(), 2, "ten 41-mers are never read");
        let large = assemble(41, Some(&small));
        assert_eq!(large.len(), 1, "the carried contig bridges the seam");
        assert_eq!(large[0].len(), genome.len());
    }

    /// The min-count, and the floor it was read at. Every candidate floor is evaluated and only then
    /// compared, all against one reference fixed beforehand, so the answer cannot depend on the order the
    /// floors happen to be visited in.
    #[cfg(not(target_family = "wasm"))]
    fn choose_min_count_and_floor(
        histovec: &[u32],
        sketch: &SpectrumSketch,
        floors: &[u8],
    ) -> (u16, u8, PeakSource, Option<SpectrumPlotDiagnostics>) {
        choose_min_count_and_floor_with_fit(
            histovec,
            sketch,
            floors,
            fit_and_log,
            &[true; MAX_GROUPS],
        )
    }

    #[cfg(not(target_family = "wasm"))]
    fn choose_min_count_and_floor_with_fit<F>(
        histovec: &[u32],
        sketch: &SpectrumSketch,
        floors: &[u8],
        fit: F,
        floor_has_kmers: &[bool],
    ) -> (u16, u8, PeakSource, Option<SpectrumPlotDiagnostics>)
    where
        F: FnMut(&[u32], &SpectrumEstimate, f64) -> Option<SpectrumFit>,
    {
        // Its callers hand it an exact strict-floor spectrum.
        let (min_count, floor, peak, diagnostics, _) = choose_min_count_and_floor_with_debug(
            histovec,
            1.0,
            sketch,
            floors,
            fit,
            floor_has_kmers,
            false,
        );
        (min_count, floor, peak, diagnostics)
    }

    /// Hash one batch into one flat vector, in record order. The counting path no longer materialises this
    /// -- it writes straight into per-shard buckets -- so this is the tests' independent reference.
    #[cfg(not(target_family = "wasm"))]
    fn hash_batch<IntT>(
        batch: &[OwnedRecord],
        k: usize,
        min_qual: u8,
        floors: Option<&[u8]>,
        keep: u8,
    ) -> Vec<(u64, u64, u8, u8, Option<IntT>)>
    where
        IntT: for<'a> UInt<'a>,
    {
        batch
            .par_iter()
            .flat_map_iter(|(seq, qual_bytes)| {
                let mut local = Vec::new();
                let mut groups: Vec<u8> = Vec::new();
                let w = KmerWalk {
                    k,
                    min_qual,
                    floors,
                    keep,
                };
                for_each_kmer::<IntT, _>(
                    seq,
                    qual_bytes.as_deref(),
                    w,
                    &mut groups,
                    |hc, hnc, b, g, km| local.push((hc, hnc, b, g, km)),
                );
                local
            })
            .collect()
    }
}
