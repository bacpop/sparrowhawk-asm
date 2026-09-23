//! Some docs should be here

#[cfg(not(target_family = "wasm"))]
use std::{path::PathBuf, time::Instant};

use libm::lgamma;
use nohash_hasher::NoHashHasher;
#[cfg(target_family = "wasm")]
use std::cmp::Ordering;
use std::{collections::HashMap, hash::BuildHasherDefault};

use rayon::prelude::*;
// use std::process::exit;

#[cfg(not(target_family = "wasm"))]
use plotters::coord::Shift;
#[cfg(not(target_family = "wasm"))]
use plotters::prelude::*;
#[cfg(not(target_family = "wasm"))]
use plotters::style::text_anchor::{HPos, Pos, VPos};

use super::HashInfoSimple;
use super::QualOpts;

// #[cfg(not(target_family = "wasm"))]
// use super::bit_encoding::{encode_base, rc_base};

use crate::bit_encoding::UInt;
#[cfg(not(target_family = "wasm"))]
use crate::bloom_filter::BloomBits;
#[cfg(target_family = "wasm")]
use crate::bloom_filter::KmerFilter;
#[cfg(not(target_family = "wasm"))]
use crate::indexed_kmers::IndexedKmers;
use crate::kmer::Kmer;
use crate::logw;
#[cfg(not(target_family = "wasm"))]
use crate::qual_profile::{window_groups, MAX_GROUPS, NONE};
#[cfg(not(target_family = "wasm"))]
use crate::spectrum_fitter::{fit_native_spectrum, FitAttempt, GenomeModel, NativeSpectrumFit};
#[cfg(target_family = "wasm")]
use crate::spectrum_fitter::{fit_spectrum, SpectrumFit};

/// Tuple for name and list of input files
pub type InputFastx = (String, Vec<String>);

/// Everything the preprocessing of one k value produces. Named fields rather than a tuple, since the
/// backends and the public entry point order these differently.
#[cfg(not(target_family = "wasm"))]
pub struct PreprocessedK<IntT> {
    /// The k this was built with. Hashes from different k live in disjoint spaces, so carrying it
    /// alongside the maps is what stops us mixing them up.
    pub k: usize,
    /// K-mer fields stored once in aligned vectors, plus their hash-to-index lookup.
    pub kmers: IndexedKmers<IntT>,
    /// `MAXSIZEHISTO`-bin k-mer spectrum; index `c-1` holds the number of distinct k-mers seen `c` times
    pub histovec: Vec<u32>,
    /// the min-count actually applied (fitted, or taken from the CLI)
    pub used_min_count: u16,
    /// the base-quality floor the spectrum asked for. Below `QualOpts::min_qual` when the lobes only
    /// separated at a looser floor, which is the caller's signal to recount.
    pub chosen_min_qual: u8,
    /// Single-copy coverage of the spectrum *this* map was filtered against, as a count. Correction
    /// reads it to tell an error branch from a real one.
    pub genomic_peak: PeakSource,
}

// #[cfg(target_family = "wasm")]
// use wasm_bindgen::prelude::*;
#[cfg(target_family = "wasm")]
use crate::fastx_wasm::open_fastq;
#[cfg(target_family = "wasm")]
use crate::post_state;
#[cfg(target_family = "wasm")]
use seq_io::fastq::Record;
#[cfg(target_family = "wasm")]
use wasm_bindgen_file_reader::WebSysFile;

// For the fitting, we'll use actually MAXSIZEHISTO - 1
const MAXSIZEHISTO: usize = 8000;

/// Historical window used by [`coverage_peak`] and returned through the WASM preprocessing JSON. It
/// does not limit the current valley or mixture fit.
pub(crate) const LEGACY_HISTO_RANGE: usize = 500;

/// Smallest useful horizontal range for a native diagnostic with no empirical genomic peak.
#[cfg(not(target_family = "wasm"))]
const MIN_NATIVE_PLOT_RANGE: usize = 10;

/// The pinned window has to fit inside the histogram, or the legacy readers would index out of bounds.
const _: () = assert!(LEGACY_HISTO_RANGE <= MAXSIZEHISTO);

#[inline]
fn add_to_histogram(histovec: &mut [u32], count: u32) {
    let idx = if (count as usize) >= MAXSIZEHISTO {
        MAXSIZEHISTO - 1
    } else {
        count.saturating_sub(1) as usize
    };
    histovec[idx] = histovec[idx].saturating_add(1);
}

/// Sampled hashes held before the threshold halves
#[cfg(not(target_family = "wasm"))]
const SKETCH_BUDGET: usize = 250_000;

/// The canonical k-mer hash never returns a value below this, so the sample is taken from the interval
/// above it: halving from zero would put the threshold in a range the hash cannot reach, and the whole
/// table would be retained away. Measured over 8M k-mers at k = 21, 31, 41 and 71.
#[cfg(not(target_family = "wasm"))]
const HASH_FLOOR: u64 = 1 << 52;

/// Per-floor k-mer spectra from a fixed-size sample of hash space. Subsampling hash space does not move
/// lambda, as a k-mer's hash does not depend on how often it occurs.
#[cfg(not(target_family = "wasm"))]
pub struct SpectrumSketch {
    threshold: u64,
    counts: HashMap<u64, [u32; MAX_GROUPS], BuildHasherDefault<NoHashHasher<u64>>>,
    /// Set once the threshold can fall no further without emptying the table.
    saturated: bool,
    /// Times the threshold has halved, for the diagnostic line only.
    shrinks: u32,
}

#[cfg(not(target_family = "wasm"))]
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
}

/// Single-copy coverage estimate straight from the spectrum: the tallest bin above the error peak.
///
/// Scans counts 3..=499 and returns a **count, not an index**; the range is pinned to
/// [`LEGACY_HISTO_RANGE`], so above ~500x the true genomic peak is invisible here. It is also a *global* argmax,
/// so on a deep library the error lobe outvotes the genome lobe — see [`find_genomic_peak`].
///
/// This now has no production callers; it remains as the pinned legacy estimator exercised by the
/// compatibility tests below. The browser independently keeps returning the same first 500 bins.
#[cfg_attr(all(not(target_family = "wasm"), not(test)), allow(dead_code))]
fn coverage_peak(histovec: &[u32]) -> usize {
    let mut best_count = 2usize; // nothing above the error peak; the caller's floor of 2 then applies
    let mut best_n = 0u32;
    for (i, &n) in histovec[2..(LEGACY_HISTO_RANGE - 1)].iter().enumerate() {
        if n > best_n {
            // Strict, so ties keep the lowest count and the result is deterministic.
            best_n = n;
            best_count = i + 3; // slice index 0 is count 3
        }
    }
    best_count
}

// =====================================================================================================
// An alternative `min_count` estimator, running beside the fit and for now only logged. It assumes no
// distribution at all: it walks up from the error lobe to the first valley and takes the genomic peak above it.
// =====================================================================================================

/// Bins either side of a count in the smoothed spectrum, and the run of rising bins that confirms we
/// have left the error lobe rather than hit noise.
const SMOOTH: usize = 2;
const RISE_RUN: usize = 3;
/// Genome k-mers the cutoff may delete, and how far the genome lobe must stand above the valley.
const MAX_GENOME_LOSS: f64 = 0.01;
/// Spectra with a textbook valley and a broad lobe were being rejected at a ratio near 2.4, so a
/// threshold of 3 cut into good data.
const MIN_GP_TO_V_RATIO: f64 = 1.75;
/// Share of k-mer *instances* the lobe above the valley must hold. A handful of noise bins clears 1 %
/// on repeats and adapters alone; a genuine lobe holds 0.75-0.95 of the sequence.
const MIN_CAND_KMER_FRAC: f64 = 0.20;
/// Used when the lobes cannot be separated. Not 1: at a genomic peak of 3-4 every singleton error survives and
/// we exhaust memory, which is worse than the ~20 % genome loss cutting at 2 costs there.
const UNRESOLVED_MINCOUNT: u16 = 2;

/// What the valley estimator concluded, and every intermediate it passed through. The extra fields are
/// carried so one log line can reproduce the decision.
#[derive(Clone, Copy, Default)]
struct SpectrumEstimate {
    /// Where [`find_valley_seed`]'s walk stopped, before [`find_valley`] refined it.
    valley_seed: usize,
    valley: usize,
    genomic_peak: usize,
    /// Heights at those two counts, so the ratio below can be checked rather than trusted.
    valley_n: u32,
    genomic_peak_n: u32,
    /// `genomic_peak_n / valley_n`, the quantity [`MIN_GP_TO_V_RATIO`] guards.
    gp_to_v_ratio: f64,
    /// `valley / genomic_peak` as counts. Logged, never gated on: near 1.0 usually means the valley search found
    /// nothing and stopped under the genomic peak, but a tight clean separation reads the same way.
    valley_to_peak_xratio: f64,
    /// Share of k-mer instances at or above the valley, the quantity [`MIN_CAND_KMER_FRAC`] guards.
    cand_kmer_frac: f64,
    /// Distinct k-mers at or above the valley: the lobe's breadth, where `genomic_peak` is its depth.
    /// Logged only for now; a floor's real cost is in breadth, and nothing yet reads it.
    distinct_above: u64,
    /// The two cutoff ceilings, kept apart so that which one binds can be read off a run.
    guard_measured: u16,
    guard_poisson: u16,
    min_count: u16,
    /// Variance-to-mean ratio of the genome lobe, or NaN when there is no usable lobe.
    dispersion: f64,
    verdict: Verdict,
}

/// Values needed to explain the spectrum decision in the native PNG. Kept independent of plotting
/// types so the fitting path can carry it without pulling native-only code into WASM.
#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy, Default)]
struct SpectrumPlotDiagnostics {
    valley: Option<usize>,
    empirical_peak: Option<usize>,
    fit: Option<NativeSpectrumFit>,
    fit_attempted: bool,
    shadow_floors: Option<[ShadowFloorDiagnostic; 4]>,
}

#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy, Debug, Default)]
struct ShadowFloorDiagnostic {
    reference_pct: u8,
    floor: u16,
    observed_distinct_removed: u64,
    fitted_error_removed: f64,
    fitted_genome_removed: f64,
}

#[cfg(not(target_family = "wasm"))]
const SHADOW_ERROR_REFERENCES: [f64; 4] = [0.25, 0.50, 0.75, 1.00];

#[cfg(not(target_family = "wasm"))]
impl SpectrumPlotDiagnostics {
    fn new(
        histovec: &[u32],
        estimate: &SpectrumEstimate,
        fit: Option<NativeSpectrumFit>,
        fit_attempted: bool,
    ) -> Self {
        Self {
            valley: (estimate.valley > 0).then_some(estimate.valley),
            empirical_peak: (estimate.genomic_peak > 0).then_some(estimate.genomic_peak),
            fit,
            fit_attempted,
            shadow_floors: fit
                .map(|fit| shadow_floor_diagnostics(histovec, estimate.min_count, &fit)),
        }
    }

    fn record_fit(
        &mut self,
        histovec: &[u32],
        estimate: &SpectrumEstimate,
        fit: Option<NativeSpectrumFit>,
    ) {
        self.fit = fit;
        self.fit_attempted = true;
        self.shadow_floors =
            fit.map(|fit| shadow_floor_diagnostics(histovec, estimate.min_count, &fit));
    }
}

#[cfg(target_family = "wasm")]
#[derive(Clone, Copy, Default)]
struct SpectrumPlotDiagnostics;

#[cfg(target_family = "wasm")]
impl SpectrumPlotDiagnostics {
    fn new(
        _histovec: &[u32],
        _estimate: &SpectrumEstimate,
        _fit: Option<SpectrumFit>,
        _fit_attempted: bool,
    ) -> Self {
        Self
    }
}

/// Why the estimator accepted or refused a spectrum. The default is a *refusal* on purpose: every path
/// through [`estimate_by_valley`] sets it, so the default is unreachable, and if one ever stops setting
/// it the run should fail closed rather than silently report a resolved spectrum.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
enum Verdict {
    #[default]
    NeverTurnsUp,
    NoPeakAboveValley,
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
            Verdict::NoPeakAboveValley => "no genomic_peak above the valley",
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

/// The fitted peak when the lobes separated, else the occurrence-weighted median.
///
/// `genomic_peak` is an argmax *above a valley*, so on a merged spectrum it names the error lobe.
/// Refusing it there is what lets every consumer treat `Fitted` as trustworthy.
fn peak_of(estimate: &SpectrumEstimate, histovec: &[u32]) -> PeakSource {
    if estimate.verdict.is_ok() {
        PeakSource::Fitted(estimate.genomic_peak as u32)
    } else {
        occurrence_weighted_median(histovec).map_or(PeakSource::Unknown, PeakSource::Fallback)
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

/// Smoothed spectrum height at `count` (a count, not an index).
fn smoothed(histovec: &[u32], count: usize) -> u64 {
    let lo = count.saturating_sub(SMOOTH).max(1);
    let hi = (count + SMOOTH).min(histovec.len() - 1);
    (lo..=hi).map(|c| histovec[c - 1] as u64).sum::<u64>() / (hi - lo + 1) as u64
}

/// First local minimum followed by [`RISE_RUN`] rising bins, as a **count**. `None` when the spectrum
/// never turns back up. Only a *valley seed* for [`find_valley`]: on a deep library whose error lobe decays
/// without a local minimum this walks into the genome lobe, which is harmless once bounded by the genomic peak.
fn find_valley_seed(histovec: &[u32]) -> Option<usize> {
    let hi = histovec.len() - 1; // the saturating bin is not part of the shape
    let mut best_count = 2usize;
    let mut best_n = smoothed(histovec, 2);
    let mut rising = 0usize;
    for count in 3..hi {
        let n = smoothed(histovec, count);
        if n < best_n {
            best_n = n;
            best_count = count;
            rising = 0;
        } else {
            rising += 1;
            if rising >= RISE_RUN {
                return Some(best_count);
            }
        }
    }
    None
}

/// Lowest smoothed bin in `2..=genomic_peak`, as a **count**: the valley between the error lobe and the genome
/// lobe. Bounded above by the genomic peak, so unlike [`find_valley_seed`] it cannot walk off into the lobe itself.
fn find_valley(histovec: &[u32], genomic_peak: usize) -> usize {
    let mut best_count = 2usize;
    let mut best_n = smoothed(histovec, 2);
    for count in 3..=genomic_peak.min(histovec.len() - 1) {
        let n = smoothed(histovec, count);
        if n < best_n {
            best_n = n;
            best_count = count;
        }
    }
    best_count
}

/// Largest cutoff whose *measured* cost is at most [`MAX_GENOME_LOSS`] of the k-mers above the valley.
/// This is the binding one at depth, where the genome lobe measures 8-12x overdispersed and a Poisson
/// tail would permit a cutoff about 35 % too high.
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

/// Largest cutoff deleting at most [`MAX_GENOME_LOSS`] of a Poisson(`genomic_peak`) genome. This is the binding
/// one at low coverage, where the lobe really is near-Poisson and the measured bound is loosened by the
/// error k-mers that still sit above the valley.
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

/// Variance-to-mean ratio of the single-copy lobe; 1.0 is Poisson. Logged because it is what says
/// whether a negative binomial in the mixture fit would be worth the work.
///
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

/// Tallest bin at or above the valley, as a **count**. Unlike [`coverage_peak`] this cannot lock onto
/// the error lobe, which is the whole difference between the two estimators.
fn find_genomic_peak(histovec: &[u32], valley: usize) -> usize {
    let hi = histovec.len() - 1;
    let mut best_count = valley;
    let mut best_n = 0u32;
    for count in valley..hi {
        if histovec[count - 1] > best_n {
            // Strict, so ties keep the lowest count.
            best_n = histovec[count - 1];
            best_count = count;
        }
    }
    best_count
}

/// The valley estimate for this spectrum. Falls back to [`UNRESOLVED_MINCOUNT`] with a stated verdict
/// wherever the lobes are not separated — at 3-5x, trusting it blindly deleted 98-99 % of the genome.
fn estimate_by_valley(histovec: &[u32]) -> SpectrumEstimate {
    let mut est = SpectrumEstimate {
        min_count: UNRESOLVED_MINCOUNT,
        dispersion: f64::NAN,
        ..Default::default()
    };
    // The valley seed only has to land past the error head, not on the valley: the search below runs from 2,
    // so an overshoot into the genome lobe still yields the right answer.
    let Some(valley_seed) = find_valley_seed(histovec) else {
        est.verdict = Verdict::NeverTurnsUp;
        return est;
    };
    est.valley_seed = valley_seed;
    est.genomic_peak = find_genomic_peak(histovec, valley_seed);
    est.valley = find_valley(histovec, est.genomic_peak);
    est.genomic_peak_n = histovec[est.genomic_peak - 1];
    est.valley_n = histovec[est.valley - 1];
    est.gp_to_v_ratio = est.genomic_peak_n as f64 / est.valley_n.max(1) as f64;
    est.valley_to_peak_xratio = est.valley as f64 / est.genomic_peak as f64;

    // k-mer INSTANCES, not distinct k-mers: at 500x the genome is under 1 % of distinct k-mers but most
    // of the sequence, so a distinct-count test would reject a perfectly healthy deep library.
    let instances_from = |from: usize| -> u128 {
        (from..=histovec.len())
            .map(|c| c as u128 * histovec[c - 1] as u128)
            .sum()
    };
    let total = instances_from(1);
    est.cand_kmer_frac = if total == 0 {
        0.0
    } else {
        instances_from(est.valley) as f64 / total as f64
    };

    // Computed unconditionally so that a bail-out can still report what the guards would have allowed.
    // Both are O(genomic peak), once per k.
    est.guard_measured = measured_guard(histovec, est.valley, est.genomic_peak);
    est.guard_poisson = poisson_guard(est.genomic_peak);
    est.dispersion = dispersion_above(histovec, est.valley, est.genomic_peak);
    est.distinct_above = distinct_above(histovec, est.valley);

    est.verdict = if est.genomic_peak <= est.valley {
        Verdict::NoPeakAboveValley
    } else if est.cand_kmer_frac < MIN_CAND_KMER_FRAC {
        Verdict::TooFewCandidateKmers
    } else if est.gp_to_v_ratio < MIN_GP_TO_V_RATIO {
        Verdict::PeakNotClearOfValley
    } else {
        Verdict::Ok
    };
    if est.verdict.is_ok() {
        // The valley removes the errors; the guards bound what that costs in genome. The Poisson one
        // binds at low coverage, where the two lobes crowd together; at depth it has slack spare.
        est.min_count = (est.valley as u16).clamp(2, est.guard_measured.min(est.guard_poisson));
    }
    est
}

/// The whole derivation, as `key=value` pairs so a directory of logs parses into a table. The prose a
/// user reads is the warning in [`choose_min_count`], not this.
fn log_spectrum(histovec: &[u32], estimate: &SpectrumEstimate) {
    logw(
        &format!(
            "K-mer spectrum: valley_seed={} valley={} genomic_peak={} valley_n={} genomic_peak_n={} \
             gp_to_v_ratio={:.2} valley_to_peak_xratio={:.3} cand_kmer_frac={:.4} \
             guard_measured={} guard_poisson={} dispersion={:.1} min_count={} distinct_above={} \
             saturates={} verdict={:?}",
            estimate.valley_seed,
            estimate.valley,
            estimate.genomic_peak,
            estimate.valley_n,
            estimate.genomic_peak_n,
            estimate.gp_to_v_ratio,
            estimate.valley_to_peak_xratio,
            estimate.cand_kmer_frac,
            estimate.guard_measured,
            estimate.guard_poisson,
            estimate.dispersion,
            estimate.min_count,
            estimate.distinct_above,
            histovec[histovec.len() - 1] > 0,
            estimate.verdict
        ),
        Some("info"),
    );
}

/// Fit the spectrum and report it. Seeded from the valley estimator's own peak and dispersion: the
/// estimator locates the lobe, the fit refines it and separates the error lobe from it explicitly.
/// `None` when no start converges, which leaves the estimator's answer standing.
#[cfg(target_family = "wasm")]
fn fit_and_log(histovec: &[u32], estimate: &SpectrumEstimate) -> Option<SpectrumFit> {
    match fit_spectrum(histovec, estimate.genomic_peak, estimate.dispersion) {
        Ok(fit) => {
            logw(
                &format!(
                    "Spectrum fit: mean={:.1} dispersion={:.2} error_mean={:.2} w=({:.3}/{:.3}/{:.3}) \
                     genome_kmers={:.3e} crossover={} hole_cutoff={}",
                    fit.mean,
                    fit.dispersion,
                    fit.error_mean,
                    fit.w_error,
                    fit.w_single,
                    fit.w_repeat,
                    fit.genome_kmers,
                    fit.crossover(),
                    fit.hole_cutoff(MAX_GENOME_HOLES),
                ),
                Some("info"),
            );
            Some(fit)
        }
        Err(e) => {
            logw(
                &format!("Spectrum fit did not converge ({e}); keeping the valley estimate."),
                Some("info"),
            );
            None
        }
    }
}

#[cfg(not(target_family = "wasm"))]
fn fit_and_log(histovec: &[u32], estimate: &SpectrumEstimate) -> Option<NativeSpectrumFit> {
    let started = Instant::now();
    let result = match fit_native_spectrum(
        histovec,
        estimate.valley,
        estimate.genomic_peak,
        estimate.dispersion,
    ) {
        Ok(result) => result,
        Err(error) => {
            logw(
                &format!(
                    "Native spectrum fits could not start ({error}); keeping the valley estimate."
                ),
                Some("info"),
            );
            return None;
        }
    };
    for attempt in &result.attempts {
        log_native_fit_attempt(attempt);
    }
    let elapsed = started.elapsed();
    if let Some(fit) = result.selected {
        logw(
            &format!(
                "Spectrum fit selected: error_model={} genome_model={} log_likelihood={:.6e} bic={:.6e} deviance={:.6e} \
                 mean={:.1} mode={} repeat_mode={} dispersion={:.2} singleton_probability={:.6} tail_exponent={:.6} weibull_shape={} \
                 w=({:.3}/{:.3}/{:.3}) genome_kmers={:.3e} error_kmers={:.3e} crossover={} \
                 hole_cutoff={} best_iterations={} elapsed_ms={}",
                fit.error_model,
                fit.genome_model,
                fit.log_likelihood,
                fit.bic,
                fit.deviance,
                fit.mean,
                fit.primary_mode(),
                fit.repeat_mode(),
                fit.dispersion,
                fit.error_params.singleton_probability,
                fit.error_params.tail_exponent,
                fit.error_params.weibull_shape.map_or_else(|| "n/a".to_string(), |shape| format!("{shape:.6}")),
                fit.w_error,
                fit.w_single,
                fit.w_repeat,
                fit.genome_kmers,
                fit.error_kmers,
                fit.crossover(),
                fit.hole_cutoff(MAX_GENOME_HOLES),
                fit.best_iterations,
                elapsed.as_millis(),
            ),
            Some("info"),
        );
        Some(fit)
    } else {
        logw(
            &format!(
                "All native spectrum fits were rejected; keeping the valley estimate. elapsed_ms={}",
                elapsed.as_millis()
            ),
            Some("info"),
        );
        None
    }
}

#[cfg(not(target_family = "wasm"))]
fn log_native_fit_attempt(attempt: &FitAttempt) {
    let selection_eligible = attempt.genome_model == GenomeModel::NegativeBinomial;
    if let Some(fit) = attempt.candidate {
        let status = attempt.rejection.map_or_else(
            || "accepted".to_string(),
            |reason| format!("rejected:{reason}"),
        );
        logw(
            &format!(
                "Spectrum fit candidate: error_model={} genome_model={} status={} log_likelihood={:.6e} bic={:.6e} \
                 deviance={:.6e} mode={} repeat_mode={} error_mode={} mean={:.1} dispersion={:.2} \
                 singleton_probability={:.6} tail_exponent={:.6} weibull_shape={} total_iterations={} capped_starts={} best_iterations={} selection_eligible={}",
                attempt.error_model,
                attempt.genome_model,
                status,
                fit.log_likelihood,
                fit.bic,
                fit.deviance,
                fit.primary_mode(),
                fit.repeat_mode(),
                fit.error_mode(),
                fit.mean,
                fit.dispersion,
                fit.error_params.singleton_probability,
                fit.error_params.tail_exponent,
                fit.error_params.weibull_shape.map_or_else(|| "n/a".to_string(), |shape| format!("{shape:.6}")),
                attempt.total_iterations,
                attempt.capped_starts,
                fit.best_iterations,
                selection_eligible,
            ),
            Some("info"),
        );
    } else {
        logw(
            &format!(
                "Spectrum fit candidate: error_model={} genome_model={} status=rejected:{} total_iterations={} capped_starts={} selection_eligible={}",
                attempt.error_model,
                attempt.genome_model,
                attempt
                    .rejection
                    .expect("an absent candidate always has a rejection reason"),
                attempt.total_iterations,
                attempt.capped_starts,
                selection_eligible,
            ),
            Some("info"),
        );
    }
}

// =====================================================================================================

/// Expected single-copy k-mers the cutoff may strand below itself. Each is a hole the graph cannot
/// bridge, so it severs a contig: what predicts contiguity is their number, not their share. One
/// percent of a 4.6 Mb genome is 46 000 severed paths, which is why a loss-fraction bound never binds.
const MAX_GENOME_HOLES: f64 = 1.0;
#[cfg_attr(all(not(target_family = "wasm"), not(test)), allow(dead_code))]
const AUTO_FIT_INITIAL_MIN_COUNT: u16 = crate::cli::MIN_BLOOM_COUNT;

/// Natively the sharded filter has no threshold of its own; the browser's `KmerFilter` still does.
#[cfg_attr(all(not(target_family = "wasm"), not(test)), allow(dead_code))]
#[inline]
fn initial_bloom_min_count(qual: &QualOpts, do_fit: bool) -> u16 {
    if do_fit {
        AUTO_FIT_INITIAL_MIN_COUNT
    } else {
        qual.min_count
    }
}

/// Choose the minimum k-mer count from the spectrum, as an **inclusive** minimum: both filter sites
/// keep k-mers with `count >= min_count`. Natively this has no callers outside the tests now; the
/// browser still reaches it from its chunked and Bloom entry points.
#[cfg_attr(all(not(target_family = "wasm"), not(test)), allow(dead_code))]
fn choose_min_count(histovec: &[u32]) -> u16 {
    choose_min_count_and_peak(histovec).0
}

/// As [`choose_min_count`], and also the single-copy coverage the same estimate found.
///
/// The valley separates the lobes; the fit then says how much genome cutting there would cost. Those
/// answer different questions, and on deep libraries the valley is far the more aggressive.
fn choose_min_count_and_peak(histovec: &[u32]) -> (u16, PeakSource, SpectrumPlotDiagnostics) {
    let mut estimate = estimate_by_valley(histovec);
    log_spectrum(histovec, &estimate);
    let fit = fit_and_log(histovec, &estimate);
    apply_hole_guard(&mut estimate, fit.as_ref(), histovec);
    warn_unresolved(&estimate);
    (
        estimate.min_count,
        peak_of(&estimate, histovec),
        SpectrumPlotDiagnostics::new(histovec, &estimate, fit, true),
    )
}

/// Lower the cutoff to whatever strands at most [`MAX_GENOME_HOLES`] single-copy k-mers.
///
/// A **ceiling** on the valley, never a raise: the valley already removes the errors, and the only
/// failure measured was cutting too deep. Floored at 2, and a no-op when the fit did not converge, so
/// the worst case is exactly the behaviour without a fit.
#[cfg(target_family = "wasm")]
fn apply_hole_guard(estimate: &mut SpectrumEstimate, fit: Option<&SpectrumFit>, _histovec: &[u32]) {
    if !estimate.verdict.is_ok() {
        return;
    }
    let Some(fit) = fit else { return };

    let guarded = fit.hole_cutoff(MAX_GENOME_HOLES);
    if guarded < estimate.min_count {
        logw(
            &format!(
                "Cutting at {} would strand more than {} single-copy k-mers; using {} instead.",
                estimate.min_count, MAX_GENOME_HOLES, guarded
            ),
            Some("info"),
        );
        estimate.min_count = guarded.max(2);
    }
}

#[cfg(not(target_family = "wasm"))]
fn apply_hole_guard(
    estimate: &mut SpectrumEstimate,
    fit: Option<&NativeSpectrumFit>,
    histovec: &[u32],
) {
    let Some(fit) = fit else { return };
    if estimate.verdict.is_ok() {
        let guarded = fit.hole_cutoff(MAX_GENOME_HOLES);
        if guarded < estimate.min_count {
            logw(
                &format!(
                    "Cutting at {} would strand more than {} single-copy k-mers; using {} instead.",
                    estimate.min_count, MAX_GENOME_HOLES, guarded
                ),
                Some("info"),
            );
            estimate.min_count = guarded.max(2);
        }
    }

    for diagnostic in shadow_floor_diagnostics(histovec, estimate.min_count, fit) {
        let ratio = if diagnostic.fitted_genome_removed > 0.0 {
            diagnostic.fitted_error_removed / diagnostic.fitted_genome_removed
        } else if diagnostic.fitted_error_removed > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        logw(
            &format!(
                "Spectrum error-floor shadow: reference_pct={} current_min_count={} shadow_floor={} \
                 observed_distinct_removed={} fitted_error_removed={:.3e} \
                 fitted_single_copy_removed={:.3e} fitted_error_per_genomic_removed={:.3e}",
                diagnostic.reference_pct,
                estimate.min_count,
                diagnostic.floor,
                diagnostic.observed_distinct_removed,
                diagnostic.fitted_error_removed,
                diagnostic.fitted_genome_removed,
                ratio,
            ),
            Some("info"),
        );
    }
}

#[cfg(not(target_family = "wasm"))]
fn shadow_floor_diagnostics(
    histovec: &[u32],
    current_min_count: u16,
    fit: &NativeSpectrumFit,
) -> [ShadowFloorDiagnostic; 4] {
    SHADOW_ERROR_REFERENCES.map(|reference| {
        let floor = fit.shadow_error_floor(reference);
        let effective_floor = floor.max(current_min_count);
        let start = usize::from(current_min_count).max(1);
        let end = usize::from(effective_floor).min(histovec.len().saturating_add(1));
        let observed_distinct_removed = if end > start {
            histovec[start - 1..end - 1]
                .iter()
                .map(|&count| u64::from(count))
                .sum()
        } else {
            0
        };
        ShadowFloorDiagnostic {
            reference_pct: (100.0 * reference).round() as u8,
            floor,
            observed_distinct_removed,
            fitted_error_removed: fit.expected_error_between(current_min_count, effective_floor),
            fitted_genome_removed: fit.expected_genome_between(current_min_count, effective_floor),
        }
    })
}

/// The warning for a spectrum that yielded no usable cutoff. Shared, so the one-pass and the sketch
/// paths cannot drift apart in what they tell the user.
fn warn_unresolved(estimate: &SpectrumEstimate) {
    // Keyed on the outcome, not only the verdict: a spectrum can be called `ok` and still have the loss
    // guard collapse the cutoff to `UNRESOLVED_MINCOUNT`, which is the same thin-spectrum situation.
    if !estimate.verdict.is_ok() || estimate.min_count == UNRESOLVED_MINCOUNT {
        logw(
            &format!(
                "The k-mer spectrum's error and genome lobes are not separated ({}), so no reliable \
                 minimum count exists and {} will be used — k-mers seen once are discarded and nothing \
                 else is. Expect a fragmented assembly if this is a deep library. This usually means \
                 the spectrum is thin: low coverage, or a large k. Check the k-mer spectrum histogram.",
                estimate.verdict.reason(), estimate.min_count
            ),
            Some("warn"),
        );
    }
}

/// Resolved *and* with room to spare. A lobe only just clearing [`MIN_GP_TO_V_RATIO`] is the marginal case
/// worth a second opinion from the sketch; the band is narrow because a ratio not far above the guard is
/// still a healthy spectrum, and widening it only buys second passes nobody needs.
#[cfg(not(target_family = "wasm"))]
const COMFORTABLE_GP_TO_V_RATIO: f64 = MIN_GP_TO_V_RATIO + 0.5;
/// Coverage below which a resolving spectrum is not taken at face value: the floor, rather than the
/// library, may be what made it shallow.
#[cfg(not(target_family = "wasm"))]
const MIN_USEFUL_COVERAGE: usize = 25;
/// A looser floor must lift the genomic peak by at least this much to justify a second pass. Measured:
/// starved libraries gain only 1.13-1.20 there while loosening is worth 2.2x, so 1.20 refused too much.
#[cfg(not(target_family = "wasm"))]
const MIN_COVERAGE_GAIN: f64 = 1.10;

#[cfg(not(target_family = "wasm"))]
fn resolves(estimate: &SpectrumEstimate) -> bool {
    estimate.verdict.is_ok() && estimate.gp_to_v_ratio >= COMFORTABLE_GP_TO_V_RATIO
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
    let strict_floor = floors[floors.len() - 1];
    let mut strict = estimate_by_valley(histovec);
    log_spectrum(histovec, &strict);
    // A library that already separates at depth needs nothing looser, and this is the only path that
    // avoids building the sketch spectra at all. It is also the common case, so the hole guard has to
    // be applied here too and not only in `choose_min_count_and_peak`.
    if resolves(&strict) && strict.genomic_peak >= MIN_USEFUL_COVERAGE {
        let fit = fit_and_log(histovec, &strict);
        apply_hole_guard(&mut strict, fit.as_ref(), histovec);
        return (
            strict.min_count,
            strict_floor,
            peak_of(&strict, histovec),
            Some(SpectrumPlotDiagnostics::new(histovec, &strict, fit, true)),
        );
    }

    // Every floor that separates, as `(floor, min_count, genomic peak)`. Collected before anything is
    // judged: accepting one used to raise the bar for the next, stranding runs on a middle floor.
    let mut cands: Vec<(u8, u16, usize)> = Vec::with_capacity(floors.len());

    let spectra = sketch.spectra(floors.len());
    for g in (0..floors.len() - 1).rev() {
        let estimate = estimate_by_valley(&spectra[g]);
        logw(
            &format!(
                "Sketch at a base-quality floor of {}: valley={} genomic_peak={} gp_to_v_ratio={:.2} \
                 min_count={} distinct_above={} verdict={:?}",
                floors[g],
                estimate.valley,
                estimate.genomic_peak,
                estimate.gp_to_v_ratio,
                estimate.min_count,
                estimate.distinct_above,
                estimate.verdict
            ),
            Some("info"),
        );
        if resolves(&estimate) {
            cands.push((floors[g], estimate.min_count, estimate.genomic_peak));
        }
    }
    if resolves(&strict) {
        cands.push((strict_floor, strict.min_count, strict.genomic_peak));
    }

    // The reference is the strictest floor that separated, which is the strict one whenever it did.
    let Some(&(anchor, _, anchor_peak)) = cands.iter().max_by_key(|&&(floor, _, _)| floor) else {
        return unresolved(floors, histovec);
    };
    cands.retain(|&(floor, _, peak)| {
        floor == anchor || (peak as f64) >= MIN_COVERAGE_GAIN * anchor_peak as f64
    });

    // Every candidate resolved, so its peak is fitted. A peak from a *sketch* candidate is measured on
    // a subsample at a floor the table was not counted at — but that case always loosens the floor, and
    // the caller then recounts, so such a value never reaches a graph.
    // Enough coverage somewhere: the strictest floor reaching it admits the fewest error k-mers.
    if let Some(&(floor, min_count, peak)) = cands
        .iter()
        .filter(|&&(_, _, peak)| peak >= MIN_USEFUL_COVERAGE)
        .max_by_key(|&&(floor, _, _)| floor)
    {
        return (
            min_count,
            floor,
            PeakSource::Fitted(peak as u32),
            (floor == strict_floor)
                .then(|| SpectrumPlotDiagnostics::new(histovec, &strict, None, false)),
        );
    }
    // Starved everywhere, so take all the depth on offer. Equal peaks are not equal assemblies: a floor
    // also breaks the run of k consecutive passing bases a k-mer needs, which no spectrum shows.
    let &(floor, min_count, peak) = cands
        .iter()
        .min_by_key(|&&(floor, _, _)| floor)
        .expect("the anchor is always a candidate");
    (
        min_count,
        floor,
        PeakSource::Fitted(peak as u32),
        (floor == strict_floor)
            .then(|| SpectrumPlotDiagnostics::new(histovec, &strict, None, false)),
    )
}

/// Nothing separated at any floor: drop the filter, which is the most depth available, and warn, because
/// the assembly will be fragmented whatever is chosen.
#[cfg(not(target_family = "wasm"))]
fn unresolved(
    floors: &[u8],
    histovec: &[u32],
) -> (u16, u8, PeakSource, Option<SpectrumPlotDiagnostics>) {
    let loosest = floors[0];
    logw(
        &format!(
            "The k-mer spectrum does not separate at any candidate base-quality floor ({floors:?}), so \
             the floor will be dropped to {loosest} and {UNRESOLVED_MINCOUNT} used as the minimum \
             count. Expect a fragmented assembly. Check the k-mer spectrum histogram.",
        ),
        Some("warn"),
    );
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
#[cfg(not(target_family = "wasm"))]
fn check_sketch_against_table(histovec: &[u32], sketch: &SpectrumSketch, keep: usize) {
    let sampled = &sketch.spectra(keep + 1)[keep];
    let (table, sample): (u64, u64) = (
        histovec.iter().map(|&n| n as u64).sum(),
        sampled.iter().map(|&n| n as u64).sum(),
    );
    let expected = table as f64 * sketch.fraction();
    // The table size is reported alongside the strict-group total: a sketch that has emptied itself
    // used to be indistinguishable here from a library with no strict-floor k-mers.
    logw(
        &format!(
            "Sketch: {sample} distinct k-mers sampled from {table} at a hash fraction of {:.3e} \
             (expected {expected:.0}; table holds {} after {} halvings{})",
            sketch.fraction(),
            sketch.counts.len(),
            sketch.shrinks,
            if sketch.saturated { ", saturated" } else { "" }
        ),
        Some("info"),
    );
}

/// The sketch's strict-floor spectrum, scaled back to the whole library. The hole guard budgets in
/// absolute k-mers, so bin heights matter to it; the counts on the x-axis are scale-invariant.
#[cfg(not(target_family = "wasm"))]
fn rescaled_strict_spectrum(sketch: &SpectrumSketch, keep: u8) -> Vec<u32> {
    let scale = 1.0 / sketch.fraction();
    let mut out = sketch.spectra(keep as usize + 1).pop().unwrap_or_default();
    for bin in out.iter_mut() {
        *bin = ((*bin as f64) * scale).min(u32::MAX as f64) as u32;
    }
    out
}

#[cfg_attr(not(target_family = "wasm"), allow(dead_code))]
fn build_histogram_from_countmap(
    countmap: &HashMap<u64, (u32, u64, u8), BuildHasherDefault<NoHashHasher<u64>>>,
    histovec: &mut [u32],
) {
    for (_, tup) in countmap.iter() {
        add_to_histogram(histovec, tup.0);
    }
}

#[cfg_attr(all(not(target_family = "wasm"), not(test)), allow(dead_code))]
fn drain_countmap_into_themap<IntT>(
    countmap: &mut HashMap<u64, (u32, u64, u8), BuildHasherDefault<NoHashHasher<u64>>>,
    themap: &mut HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
    outdict: &mut HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    minmaxdict: &mut HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
    minc: u16,
    mut histovec_opt: Option<&mut [u32]>,
) where
    IntT: for<'a> UInt<'a>,
{
    countmap.retain(|h, tup| {
        if tup.0 >= minc as u32 {
            themap.entry(*h).or_insert_with(|| HashInfoSimple {
                hnc: tup.1,
                b: tup.2,
                pre: Vec::new(),
                post: Vec::new(),
                counts: tup.0,
            });
        } else {
            outdict.remove(h);
            minmaxdict.remove(&tup.1);
        }
        if let Some(ref mut hv) = histovec_opt {
            add_to_histogram(hv, tup.0);
        }
        false
    });
}

/// One FASTQ record, owned: needletail's `Cow` borrows a reader buffer invalidated on the next
/// `next()`, so a batch processed in parallel must own its bytes.
#[cfg(not(target_family = "wasm"))]
type OwnedRecord = (Vec<u8>, Option<Vec<u8>>);

/// How many records go to the workers at a time: ~1M k-mers at 150 bp and k=31, small enough to stay
/// cache-friendly and large enough to amortise the rayon fork/join.
#[cfg(not(target_family = "wasm"))]
const BATCH_RECORDS: usize = 8192;

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

/// Writes the full spectrum beside its plots as `count<TAB>distinct`, so a run's cutoff decision can
/// be replayed offline. The PNG and SVG show the fitting window and cannot be read back.
#[cfg(not(target_family = "wasm"))]
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

#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MarkerLineStyle {
    Solid,
    Dashed,
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
        let crossover = fit.crossover();
        markers.push(PlotMarker {
            x: f64::from(crossover),
            colour: RGBColor(120, 70, 200),
            label: format!("Error/genome crossover = {crossover}"),
            line_style: MarkerLineStyle::Solid,
        });
    }
    if let Some(shadows) = diagnostics.shadow_floors {
        let mut grouped: Vec<(u16, Vec<u8>)> = Vec::new();
        for shadow in shadows {
            if let Some((_, percentages)) =
                grouped.iter_mut().find(|(floor, _)| *floor == shadow.floor)
            {
                percentages.push(shadow.reference_pct);
            } else {
                grouped.push((shadow.floor, vec![shadow.reference_pct]));
            }
        }
        for (floor, percentages) in grouped {
            let percentages = percentages
                .iter()
                .map(|percentage| format!("{percentage}%"))
                .collect::<Vec<_>>()
                .join("/");
            markers.push(PlotMarker {
                x: f64::from(floor),
                colour: RGBColor(105, 105, 105),
                label: format!("Shadow error floor {percentages} = {floor} (not applied)"),
                line_style: MarkerLineStyle::Dashed,
            });
        }
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
fn native_plot_end(
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
        .unwrap_or(MIN_NATIVE_PLOT_RANGE);
    let marker_end = plot_markers(diagnostics, used_min_count)
        .iter()
        .map(|marker| marker.x.ceil() as usize)
        .max()
        .unwrap_or(0);
    fit_end
        .max(marker_end)
        .max(MIN_NATIVE_PLOT_RANGE)
        .min(histogram_end)
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
) where
    DB::ErrorType: 'static,
{
    const Y_MAX: f64 = 200_000.0;

    let markers = plot_markers(diagnostics, used_min_count);
    let plot_end = native_plot_end(decision_spectrum.len(), diagnostics, used_min_count);
    let mut notes: Vec<String> = markers
        .iter()
        .filter(|marker| marker.x > plot_end as f64)
        .map(|marker| format!("{} (off-scale)", marker.label))
        .collect();
    if diagnostics.fit_attempted && diagnostics.fit.is_none() {
        notes.push("mixture fit unavailable".to_string());
    }
    let caption = if notes.is_empty() {
        "k-mer spectrum".to_string()
    } else {
        format!("k-mer spectrum — {}", notes.join(", "))
    };

    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(35)
        .y_label_area_size(65)
        .margin(5)
        .caption(caption, ("ibm-plex-sans", 24.0))
        .build_cartesian_2d(0.0f64..plot_end as f64, 0.0f64..Y_MAX)
        .unwrap();
    chart
        .configure_mesh()
        .disable_x_mesh()
        .x_labels(15)
        .y_labels(11)
        .max_light_lines(2)
        .light_line_style(RGBColor(220, 220, 220).mix(0.35))
        .bold_line_style(RGBColor(180, 180, 180).mix(0.4))
        .y_label_formatter(&|y| format!("{y:.0}"))
        .draw()
        .unwrap();

    let shown = &decision_spectrum[..plot_end.min(decision_spectrum.len())];
    chart
        .draw_series(shown.iter().enumerate().map(|(i, &height)| {
            let count = (i + 1) as f64;
            Rectangle::new(
                [(count - 0.5, 0.0), (count + 0.5, f64::from(height))],
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
                    .map(|(i, &height)| ((i + 1) as f64, f64::from(height))),
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
                    .map(|(count, values)| (*count, values.iter().sum())),
                BLACK.stroke_width(2),
            ))
            .unwrap()
            .label("Fitted mixture")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 16, y)], BLACK.stroke_width(2)));
        chart
            .draw_series(DashedLineSeries::new(
                components.iter().map(|(count, values)| (*count, values[0])),
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
                components.iter().map(|(count, values)| (*count, values[1])),
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
        chart
            .draw_series(DashedLineSeries::new(
                components.iter().map(|(count, values)| (*count, values[2])),
                2,
                5,
                RGBColor(100, 100, 100).stroke_width(1),
            ))
            .unwrap()
            .label(format!("Two-copy ({})", fit.genome_model))
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
                [(marker.x, 0.0), (marker.x, Y_MAX)],
                marker.colour.stroke_width(1),
            )),
            MarkerLineStyle::Dashed => chart.draw_series(DashedLineSeries::new(
                [(marker.x, 0.0), (marker.x, Y_MAX)],
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
    let axis_style = TextStyle::from(("ibm-plex-sans", 15).into_font());
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

#[cfg(not(target_family = "wasm"))]
fn plot_kmer_histogram(
    decision_spectrum: &[u32],
    raw_bloom_spectrum: Option<&[u32]>,
    diagnostics: &SpectrumPlotDiagnostics,
    used_min_count: u16,
    out_path: &std::path::Path,
) {
    draw_kmer_histogram(
        BitMapBackend::new(out_path, (1280, 960)).into_drawing_area(),
        decision_spectrum,
        raw_bloom_spectrum,
        diagnostics,
        used_min_count,
    );
    let svg_path = out_path.with_extension("svg");
    draw_kmer_histogram(
        SVGBackend::new(&svg_path, (1280, 960)).into_drawing_area(),
        decision_spectrum,
        raw_bloom_spectrum,
        diagnostics,
        used_min_count,
    );
}

// =====================================================================================================

// NOTE: these two functions were implemented to save the whole sequence in memory for GPGPU processing.
// This might be useful again in the future, but now it was just meaning that we were wasting memory!
// I'm disabling them for the moment
//
// #[cfg(not(target_family = "wasm"))]
// fn put_these_nts_into_an_efficient_vector(charseq : &[u8], compseq : &mut Vec<u64>, occ : u8) {
//     let mut tmpu64 : u64 = 0;
//     let mut tmpind : u8  = 0;
//
//     if occ != 0 {
//         tmpind = occ;
//         tmpu64 = compseq.pop().unwrap();
//     }
// //     log::debug!("{}", tmpind);
//
//     for nt in charseq {
// //         log::debug!("\n{:#010b}\n{}\n{:#066b}\n{:#066b}", *nt, tmpind, tmpu64, (encode_base(*nt) as u64));
//         tmpu64 <<= 2;
//         tmpu64 |= encode_base(*nt) as u64;
// //         log::debug!("{:#066b}", tmpu64);
//         if tmpind == 31 {
//             compseq.push(tmpu64);
//             tmpu64 = 0;
//             tmpind = 0;
//         } else {
//             tmpind += 1;
//         }
// //         log::debug!("{}", tmpind);
//     }
//
//     if tmpind != 0 {
//         compseq.push(tmpu64);
//     }
// }

// #[cfg(not(target_family = "wasm"))]
// fn put_these_nts_into_an_efficient_vector_rc(charseq : &[u8], compseq : &mut Vec<u64>, occ : u8) {
//     let mut tmpu64 : u64 = 0;
//     let mut tmpind : u8  = 0;
//
//     if occ != 0 {
//         tmpind = occ;
//         tmpu64 = compseq.pop().unwrap();
//     }
// //     log::debug!("{}", tmpind);
//     for nt in charseq.iter().rev() {
// //         log::debug!("\n{:#010b}\n{}\n{:#066b}\n{:#066b}", *nt, tmpind, tmpu64, (rc_base(encode_base(*nt)) as u64));
//         tmpu64 <<= 2;
//         tmpu64 |= rc_base(encode_base(*nt)) as u64;
// //         log::debug!("{:#066b}", tmpu64);
//         if tmpind == 31 {
//             compseq.push(tmpu64);
//             tmpu64 = 0;
//             tmpind = 0;
//         } else {
//             tmpind += 1;
//         }
//     }
//
//     if tmpind != 0 {
//         compseq.push(tmpu64);
//     }
// }

#[cfg(target_family = "wasm")]
fn chunked_processing_wasm<IntT>(
    file1: &mut WebSysFile,
    file2: Option<&mut WebSysFile>,
    k: usize,
    qual: &QualOpts,
    outvec: &mut Vec<(u64, u64, u8)>,
    csize: usize,
    do_fit: bool,
) -> (
    HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
    HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
    Vec<u32>,
    u16,
)
where
    IntT: for<'a> UInt<'a>,
{
    post_state("preprocess:chunked:start");
    logw(
        "Getting kmers from first file. Creating reader...",
        Some("info"),
    );

    let mut outdict = HashMap::with_hasher(BuildHasherDefault::default());
    let mut minmaxdict = HashMap::with_hasher(BuildHasherDefault::default());
    let mut themap = HashMap::with_hasher(BuildHasherDefault::default());
    let mut countmap: HashMap<u64, (u32, u64, u8), BuildHasherDefault<NoHashHasher<u64>>> =
        HashMap::with_hasher(BuildHasherDefault::default());

    let mut reader = open_fastq(file1);
    let mut histovec: Vec<u32> = vec![0; MAXSIZEHISTO];

    logw("Entering while loop...", Some("info"));

    // todo: use the same counter and is_multiple_of
    let mut i_record = 0;
    let mut count: usize = 0;
    post_state("preprocess:chunked:loop:start");

    while let Some(record) = reader.next() {
        let seqrec = record.expect("Invalid FASTQ record");
        let rl = seqrec.seq().len();
        let kmer_opt = Kmer::<IntT>::new(
            std::borrow::Cow::Borrowed(seqrec.seq()),
            rl,
            Some(seqrec.qual()),
            k,
            qual.min_qual,
            true,
        );
        if let Some(mut kmer_it) = kmer_opt {
            let (hc, hnc, b, km) = kmer_it.get_curr_kmerhash_and_bases_and_kmer();
            outvec.push((hc, hnc, b));
            outdict.entry(hc).or_insert(km);
            minmaxdict.entry(hnc).or_insert(hc);
            while let Some(tmptuple) = kmer_it.get_next_kmer_and_give_us_things() {
                let (hc, hnc, b, km) = tmptuple;
                outvec.push((hc, hnc, b));
                outdict.entry(hc).or_insert(km);
                minmaxdict.entry(hnc).or_insert(hc);
            }
        }

        i_record += 1;
        count += 1;
        if i_record >= csize {
            // Processssssss! And reset.
            if !outvec.is_empty() {
                logw("Processing chunk. Sorting k-mers...", Some("debug"));
                outvec.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
                logw("k-mers sorted. Counting k-mers...", Some("debug"));
                // Then, do a counting of everything and save the results in a dictionary and return it

                update_countmap(outvec, &mut countmap);
            }

            // Reset
            outvec.clear();
            post_state(&format!("preprocess:chunked:loop:{:?}", count));
            i_record = 0;
        }
    }
    logw(
        "Finished getting kmers from first file. Starting with the second...",
        Some("info"),
    );

    post_state(&format!("preprocess:chunked:loop:{:?}:50", count));
    let percentageblock = (count as f64 / 10_f64) as usize;

    if let Some(file2) = file2 {
        let mut reader = open_fastq(file2);

        // Filling the seq of the second file!
        while let Some(record) = reader.next() {
            let seqrec = record.expect("Invalid FASTQ record");
            // put_these_nts_into_an_efficient_vector_rc(&seqrec.seq(), &mut theseq, (itrecord % 32) as u8);
            let rl = seqrec.seq().len();
            let kmer_opt = Kmer::<IntT>::new(
                std::borrow::Cow::Borrowed(seqrec.seq()),
                rl,
                Some(seqrec.qual()),
                k,
                qual.min_qual,
                true,
            );
            if let Some(mut kmer_it) = kmer_opt {
                let (hc, hnc, b, km) = kmer_it.get_curr_kmerhash_and_bases_and_kmer();
                outvec.push((hc, hnc, b));
                outdict.entry(hc).or_insert(km);
                minmaxdict.entry(hnc).or_insert(hc);
                while let Some(tmptuple) = kmer_it.get_next_kmer_and_give_us_things() {
                    let (hc, hnc, b, km) = tmptuple;
                    outvec.push((hc, hnc, b));
                    outdict.entry(hc).or_insert(km);
                    minmaxdict.entry(hnc).or_insert(hc);
                }
            }

            i_record += 1;
            count += 1;
            if i_record >= csize {
                // Processssssss! And reset.
                if !outvec.is_empty() {
                    logw("Processing chunk. Sorting k-mers...", Some("info"));
                    outvec.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
                    logw("k-mers sorted. Counting k-mers...", Some("info"));
                    // Then, do a counting of everything and save the results in a dictionary and return it

                    update_countmap(outvec, &mut countmap);
                }

                // Reset
                outvec.clear();
                i_record = 0;
            }

            // This might be done slightly more efficiently??
            if percentageblock > 0 && count.is_multiple_of(percentageblock) {
                post_state(&format!(
                    "preprocess:chunked:loop:{:?}:{:?}",
                    count,
                    count / percentageblock * 5
                ));
            }
        }
    }

    logw(
        "Finished getting kmers from the input file(s)",
        Some("info"),
    );

    // The residual chunk. This MUST stay outside the `if let Some(file2)` block above, or single-file
    // input silently loses the k-mers of its trailing partial chunk.
    if !outvec.is_empty() {
        logw("Processing last chunk. Sorting k-mers...", Some("info"));
        outvec.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
        logw("k-mers sorted. Counting k-mers...", Some("info"));
        update_countmap(outvec, &mut countmap);
        outvec.clear();
    }

    post_state("preprocess:chunked:loop:end");
    logw("Filtering...", Some("info"));

    // Now, get themap, histovec, and filter outdict and minmaxdict
    countmap.shrink_to_fit();
    let mut minc: u16 = qual.min_count;

    // This can be optimised. also better written: I had to repeat the code for the retains, to try to improve slightly the running time in
    // case no autofitting is requested. In any case, it could be improved in the future.
    if do_fit {
        post_state("preprocess:chunked:fitting");
        build_histogram_from_countmap(&countmap, &mut histovec);

        // Remove the last bin, as it might affect the fit, but we want it in the vector to plot it in case the coverage is really
        // large (and so that we can detect it).
        logw(
            "Counting finished. Choosing the minimum count...",
            Some("info"),
        );
        minc = choose_min_count(&histovec);
        logw(
            format!("Minimum count chosen: {}. Starting filtering...", minc).as_str(),
            Some("info"),
        );

        post_state("preprocess:chunked:filtering");
        drain_countmap_into_themap(
            &mut countmap,
            &mut themap,
            &mut outdict,
            &mut minmaxdict,
            minc,
            None,
        );
    } else {
        post_state("preprocess:chunked:filtering");
        drain_countmap_into_themap(
            &mut countmap,
            &mut themap,
            &mut outdict,
            &mut minmaxdict,
            minc,
            Some(&mut histovec),
        );
    }

    outdict.shrink_to_fit();
    minmaxdict.shrink_to_fit();
    histovec.shrink_to_fit();
    drop(countmap);

    (outdict, minmaxdict, themap, histovec, minc)
}

#[cfg(target_family = "wasm")]
fn bloom_filter_preprocessing_wasm<IntT>(
    file1: &mut WebSysFile,
    file2: Option<&mut WebSysFile>,
    k: usize,
    qual: &QualOpts,
    do_fit: bool,
) -> (
    HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
    HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
    Vec<u32>,
    u16,
)
where
    IntT: for<'a> UInt<'a>,
{
    post_state("preprocess:bloom:start");
    logw("Getting kmers from first file with Bloom filter. Creating reader and initialising filter...", Some("info"));

    let mut outdict = HashMap::with_hasher(BuildHasherDefault::default());
    let mut minmaxdict = HashMap::with_hasher(BuildHasherDefault::default());
    let mut themap = HashMap::with_hasher(BuildHasherDefault::default());

    let mut histovec: Vec<u32> = vec![0; MAXSIZEHISTO];

    let mut kmer_filter = KmerFilter::new(initial_bloom_min_count(qual, do_fit));
    kmer_filter.init();

    logw("Entering while loop for the first file...", Some("info"));
    post_state("preprocess:bloom:loop:start");

    let mut count: usize = 0;

    let mut reader = open_fastq(file1);
    while let Some(record) = reader.next() {
        let seqrec = record.expect("Invalid FASTQ record");
        let rl = seqrec.seq().len();
        let kmer_opt = Kmer::<IntT>::new(
            std::borrow::Cow::Borrowed(seqrec.seq()),
            rl,
            Some(seqrec.qual()),
            k,
            qual.min_qual,
            true,
        );
        if let Some(mut kmer_it) = kmer_opt {
            let (hc, hnc, b) = kmer_it.get_curr_hash_and_bases();
            if Ordering::is_eq(kmer_filter.filter(hc, hnc, b)) {
                outdict.entry(hc).or_insert_with(|| kmer_it.get_kmer());
                minmaxdict.entry(hnc).or_insert(hc);
            }
            while let Some((hc, hnc, b)) = kmer_it.get_next_hash_and_bases() {
                if Ordering::is_eq(kmer_filter.filter(hc, hnc, b)) {
                    outdict.entry(hc).or_insert_with(|| kmer_it.get_kmer());
                    minmaxdict.entry(hnc).or_insert(hc);
                }
            }
        }
        count += 1;
        if count.is_multiple_of(150000) {
            post_state(&format!("preprocess:bloom:loop:{:?}", count));
        }
    }
    logw(
        "Finished getting kmers from first file. Starting with the second...",
        Some("info"),
    );
    post_state(&format!("preprocess:bloom:loop:{:?}:50", count));
    let percentageblock = (count as f64 / 10_f64) as usize;

    if let Some(file2) = file2 {
        let mut reader = open_fastq(file2);

        // Filling the seq of the second file!
        while let Some(record) = reader.next() {
            let seqrec = record.expect("Invalid FASTQ record");
            // put_these_nts_into_an_efficient_vector_rc(&seqrec.seq(), &mut theseq, (itrecord % 32) as u8);
            let rl = seqrec.seq().len();
            let kmer_opt = Kmer::<IntT>::new(
                std::borrow::Cow::Borrowed(seqrec.seq()),
                rl,
                Some(seqrec.qual()),
                k,
                qual.min_qual,
                true,
            );
            if let Some(mut kmer_it) = kmer_opt {
                let (hc, hnc, b) = kmer_it.get_curr_hash_and_bases();
                if Ordering::is_eq(kmer_filter.filter(hc, hnc, b)) {
                    outdict.entry(hc).or_insert_with(|| kmer_it.get_kmer());
                    minmaxdict.entry(hnc).or_insert(hc);
                }
                while let Some((hc, hnc, b)) = kmer_it.get_next_hash_and_bases() {
                    if Ordering::is_eq(kmer_filter.filter(hc, hnc, b)) {
                        outdict.entry(hc).or_insert_with(|| kmer_it.get_kmer());
                        minmaxdict.entry(hnc).or_insert(hc);
                    }
                }
            }
            count += 1;

            // This might be done slightly more efficiently??
            if percentageblock > 0 && count.is_multiple_of(percentageblock) {
                post_state(&format!(
                    "preprocess:bloom:loop:{:?}:{:?}",
                    count,
                    count / percentageblock * 5
                ));
            }
        }
    }

    post_state("preprocess:bloom:loop:end");
    logw("Finished getting kmers from the second file", Some("info"));
    logw("Second part of filtering...", Some("info"));

    // // Now, get themap, histovec, and filter outdict and minmaxdict
    let mut countmap = kmer_filter.get_counts_map();
    countmap.shrink_to_fit();

    // This can be optimised. also better written: I had to repeat the code for the retains, to try to improve slightly the running time in
    // case no autofitting is requested. In any case, it could be improved in the future.
    let mut minc = qual.min_count;
    if do_fit {
        post_state("preprocess:bloom:fitting");
        build_histogram_from_countmap(&countmap, &mut histovec);

        // Remove the last bin, as it might affect the fit, but we want it in the vector to plot it in case the coverage is really
        // large (and so that we can detect it).
        logw(
            "Counting finished. Choosing the minimum count...",
            Some("info"),
        );
        minc = choose_min_count(&histovec);
        logw(
            format!("Minimum count chosen: {}. Starting filtering...", minc).as_str(),
            Some("info"),
        );

        post_state("preprocess:bloom:filtering");
        drain_countmap_into_themap(
            &mut countmap,
            &mut themap,
            &mut outdict,
            &mut minmaxdict,
            minc,
            None,
        );
    } else {
        post_state("preprocess:bloom:filtering");
        // I think this part can be improved: now that the min_count error has been corrected, some conditionals could be removed here?
        drain_countmap_into_themap(
            &mut countmap,
            &mut themap,
            &mut outdict,
            &mut minmaxdict,
            minc,
            Some(&mut histovec),
        );
    }

    outdict.shrink_to_fit();
    minmaxdict.shrink_to_fit();

    (outdict, minmaxdict, themap, histovec, minc)
}

/// Natively this now has no callers outside the tests: the map counter never materialises a list of
/// occurrences to run-length count. The browser still reaches it through `chunked_processing_wasm`.
#[cfg_attr(all(not(target_family = "wasm"), not(test)), allow(dead_code))]
fn update_countmap(
    invec: &[(u64, u64, u8)],
    countmap: &mut HashMap<u64, (u32, u64, u8), BuildHasherDefault<NoHashHasher<u64>>>,
) {
    let mut i = 0;
    let mut c: u32 = 0;
    let mut tmphash = invec[i].0;
    // let mut tmpcounter = 0;

    while i < invec.len() {
        if tmphash != invec[i].0 {
            let tmpref = countmap
                .entry(tmphash)
                .or_insert((0, invec[i - 1].1, invec[i - 1].2));
            tmpref.0 = tmpref.0.saturating_add(c);

            tmphash = invec[i].0;
            c = 1;
        } else {
            c = c.saturating_add(1);
        }
        i += 1;
    }

    let tmpref = countmap
        .entry(tmphash)
        .or_insert((0, invec[i - 1].1, invec[i - 1].2));
    tmpref.0 = tmpref.0.saturating_add(c);
}

/// The constants of one counting pass: fixed for every record, so they travel together.
#[cfg(not(target_family = "wasm"))]
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
#[cfg(not(target_family = "wasm"))]
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
    let kmer_opt = Kmer::<IntT>::new(
        std::borrow::Cow::Borrowed(seq),
        seq.len(),
        qual_bytes,
        w.k,
        // Filtering nothing at the iterator is what "never stop because of qualities" means.
        if groups.is_empty() { w.min_qual } else { 0 },
        true,
    );
    if let Some(mut kmer_it) = kmer_opt {
        let (mut hc, mut hnc, mut b) = kmer_it.get_curr_hash_and_bases();
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

/// Hash one batch into one flat vector, in record order. The counting path no longer materialises this
/// -- it writes straight into per-shard buckets -- so this is the tests' independent reference.
#[cfg(all(not(target_family = "wasm"), test))]
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

/// One distinct k-mer, accumulated while counting in bulk.
///
/// `hnc`, `b` and `km` describe the *k-mer*, not the *occurrence*, so they are recorded once, on first
/// sight, rather than once per occurrence as the old sorting vector did.
#[cfg(not(target_family = "wasm"))]
struct KmerInfo<IntT> {
    count: u32,
    hnc: u64,
    b: u8,
    km: IntT,
}

#[cfg(not(target_family = "wasm"))]
type CountMap<IntT> = HashMap<u64, KmerInfo<IntT>, BuildHasherDefault<NoHashHasher<u64>>>;

/// Where the counting phase spends its time, accumulated over every batch. It is bandwidth-bound
/// rather than synchronisation-bound, so wall time alone misattributes it, and a few thousand clock
/// reads per run cost nothing against that.
#[cfg(not(target_family = "wasm"))]
#[derive(Default)]
struct CountTimings {
    hash_ns: u128,
    sketch_ns: u128,
    split_ns: u128,
    absorb_ns: u128,
    batches: usize,
}

#[cfg(not(target_family = "wasm"))]
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

/// Never fewer shards than this: below it each map stops being a small fraction of the working set. Heuristically set.
#[cfg(not(target_family = "wasm"))]
const MIN_COUNTMAP_SHARDS: usize = 64;
/// Nor more than this, or the per-batch split pays for buckets that hold almost nothing.
#[cfg(not(target_family = "wasm"))]
const MAX_COUNTMAP_SHARDS: usize = 256;
/// Shards per thread. One, not more: oversubscribing to give rayon work to steal was measured and is a
/// bad trade, since every extra shard is another half-empty table. At k=81 on 12 threads, 64 shards
/// cost 17% peak RSS to save 4% wall against 16.
#[cfg(not(target_family = "wasm"))]
const SHARDS_PER_THREAD: usize = 1;

/// How many shards to split the count-map into.
///
/// Shards partition the key space, so counting runs on all cores with no locking, and each map is a
/// fraction of the working set, which probes better than one big one. The floor of 16 is what every
/// run up to 16 threads gets, so this only scales where fewer shards than threads would idle workers.
///
/// Always a power of two, because [`shard_of`] shifts and masks: a non-power-of-two silently collapses
/// onto the one below it -- 6 shards would use 2, 12 would use 4 -- with no error, and a balance report
/// that still reads as healthy because the shards in use stay balanced among themselves.
#[cfg(not(target_family = "wasm"))]
fn countmap_shards() -> usize {
    (rayon::current_num_threads() * SHARDS_PER_THREAD)
        .next_power_of_two()
        .clamp(MIN_COUNTMAP_SHARDS, MAX_COUNTMAP_SHARDS)
}

/// Pick a shard for a canonical hash, mixing first. `shards` must be a power of two.
///
/// Neither end of `hc` works raw. Not the high bits: ntHash is uniform but `hc = min(fwd, rc)` is not,
/// and measured at k=31 the top 4 bits run 12.2% down to 0.37%, a 33x spread that left shard 15 empty.
/// Not the low bits: `NoHashHasher` passes the hash through and hashbrown indexes buckets with them.
#[cfg(not(target_family = "wasm"))]
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
#[cfg(not(target_family = "wasm"))]
type Bucket<IntT> = Vec<(u64, u64, u8, IntT)>;

/// One rayon task's private scratch, allocated once and reused across every batch.
#[cfg(not(target_family = "wasm"))]
struct TaskState<IntT> {
    /// Kept k-mers, already split by shard. Absorb drains these, so the capacity carries over.
    buckets: Vec<Bucket<IntT>>,
    /// `window_groups` scratch, reused for every record the task sees.
    groups: Vec<u8>,
    /// Sketch candidates, in record order.
    cand: Vec<(u64, u8)>,
}

#[cfg(not(target_family = "wasm"))]
impl<IntT> TaskState<IntT> {
    fn new(n_shards: usize) -> Self {
        Self {
            buckets: (0..n_shards).map(|_| Vec::new()).collect(),
            groups: Vec::new(),
            cand: Vec::new(),
        }
    }
}

/// Hash one batch straight into the tasks' per-shard buckets, then absorb those into the count-map.
/// Nothing is materialised in between: a k-mer is written once into a bucket and once into the map,
/// where the flat-vector version wrote it four times. Shards partition the keys, so absorb needs no lock.
#[cfg(not(target_family = "wasm"))]
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
    // exactly what the serial `KmerFilter::filter` did.
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
            } = st;
            for (seq, qual_bytes) in recs {
                for_each_kmer::<IntT, _>(
                    seq,
                    qual_bytes.as_deref(),
                    w,
                    groups,
                    |hc, hnc, b, g, km| {
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

/// Count every k-mer straight into a [`CountMap`], with the per-floor sketch.
///
/// Replaces "push every occurrence, sort, run-length count". Bringing equal k-mers together is all that
/// vector ever did, and a hash map does it in one probe per occurrence — against a push, a sort of every
/// sighting, and two dictionary probes. It also drops the occurrence buffer entirely.
#[cfg(not(target_family = "wasm"))]
fn bulk_preprocessing_standalone_cpu<IntT, I>(
    input_iters: &mut [I],
    k: usize,
    qual: &QualOpts,
    floors: Option<&[u8]>,
    do_bloom: bool,
) -> (Vec<CountMap<IntT>>, Option<SpectrumSketch>)
where
    IntT: for<'a> UInt<'a>,
    I: Iterator<Item = (Vec<u8>, Option<Vec<u8>>)> + Send,
{
    // The ladder is ascending, so the last group is the strict floor and `keep` is its index.
    let keep = floors.map_or(0u8, |f| (f.len() - 1) as u8);
    // A Bloom table has no count-1 bin, so its spectrum has to come from the sketch whether or not a
    // ladder was asked for.
    let mut sketch = (floors.is_some() || do_bloom).then(SpectrumSketch::new);
    let n_shards = countmap_shards();
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
    t.log(total);

    (shards, sketch)
}

/// Consume the count-map into aligned storage, keeping k-mers seen `minc` times.
#[cfg(not(target_family = "wasm"))]
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

/// Choose, filter and plot from a finished sharded count-map.
#[cfg(not(target_family = "wasm"))]
#[allow(clippy::type_complexity)]
fn finish_map_counter<IntT>(
    shards: Vec<CountMap<IntT>>,
    qual: &QualOpts,
    floors: Option<&[u8]>,
    sketch: Option<SpectrumSketch>,
    do_fit: bool,
    do_bloom: bool,
    out_path: &mut Option<PathBuf>,
) -> (IndexedKmers<IntT>, Vec<u32>, u16, u8, PeakSource)
where
    IntT: for<'a> UInt<'a>,
{
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
    let spectrum: &[u32] = if do_bloom {
        let keep = floors.map_or(0u8, |f| (f.len() - 1) as u8);
        sketch_spectrum = rescaled_strict_spectrum(
            sketch.as_ref().expect("the Bloom path always sketches"),
            keep,
        );
        &sketch_spectrum
    } else {
        &histovec
    };

    let minc;
    // The floor the spectrum asks for. Only the fitting path can ask for a looser one.
    let mut chosen_min_qual = qual.min_qual;
    // Single-copy coverage of the spectrum this very map is filtered against.
    let genomic_peak;
    let mut plot_diagnostics;
    if do_fit {
        log::info!("Counting finished. Choosing the minimum count...");
        match (&sketch, floors) {
            (Some(sketch), Some(floors)) => {
                // The table is a bin short under Bloom counting, so the comparison would be noise.
                if !do_bloom {
                    check_sketch_against_table(&histovec, sketch, floors.len() - 1);
                }
                (minc, chosen_min_qual, genomic_peak, plot_diagnostics) =
                    choose_min_count_and_floor(spectrum, sketch, floors);
            }
            _ => {
                let diagnostics;
                (minc, genomic_peak, diagnostics) = choose_min_count_and_peak(spectrum);
                plot_diagnostics = Some(diagnostics);
            }
        }
        if chosen_min_qual < qual.min_qual {
            // The caller recounts at the looser floor, so building the maps here is wasted work and
            // wasted memory. Dropping `shards` on the way out is the whole saving.
            return (
                IndexedKmers::default(),
                histovec,
                minc,
                chosen_min_qual,
                genomic_peak,
            );
        }
        log::info!("Minimum count chosen: {minc}. Starting filtering...");
    } else {
        minc = qual.min_count;
        // `--min-count` fixes the cutoff, not the coverage. The spectrum above spans every distinct
        // k-mer including singletons, so it is the same one the fitting path reads.
        let estimate = estimate_by_valley(spectrum);
        genomic_peak = peak_of(&estimate, spectrum);
        plot_diagnostics = Some(SpectrumPlotDiagnostics::new(
            spectrum, &estimate, None, false,
        ));
    }
    log::info!("Single-copy coverage read from the spectrum: {genomic_peak:?}");

    let kmers = countmaps_into_indexed_kmers::<IntT>(shards, minc);

    if let Some(p) = out_path {
        if do_fit {
            if let Some(diagnostics) = plot_diagnostics.as_mut() {
                if !diagnostics.fit_attempted {
                    let estimate = estimate_by_valley(spectrum);
                    let fit = fit_and_log(spectrum, &estimate);
                    diagnostics.record_fit(spectrum, &estimate, fit);
                }
            }
        }
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
    (kmers, histovec, minc, chosen_min_qual, genomic_peak)
}

/// Read fastq files, get the reads, get the k-mers, count them, filter them by count, and get some way of recovering the sequence later.
#[cfg(not(target_family = "wasm"))]
#[allow(clippy::too_many_arguments)]
pub fn preprocessing_standalone<IntT, I>(
    input_iters: &mut [I],
    k: usize,
    qual: &QualOpts,
    floors: Option<&[u8]>,
    timevec: &mut Option<&mut Vec<Instant>>,
    out_path: &mut Option<PathBuf>,
    csize: usize,
    do_bloom: bool,
    do_fit: bool,
) -> PreprocessedK<IntT>
where
    IntT: for<'a> UInt<'a>,
    I: Iterator<Item = (Vec<u8>, Option<Vec<u8>>)> + Send,
{
    log::info!("Starting preprocessing_standalone with k = {k}");

    if csize != 0 {
        log::warn!(
            "--chunk-size is ignored: k-mers are counted into a hash map, which buffers no \
             occurrences, so memory is bounded by the number of distinct k-mers instead."
        );
    }
    log::info!("Counting k-mers into a hash map, without sorting");
    let (shards, sketch) =
        bulk_preprocessing_standalone_cpu::<IntT, _>(input_iters, k, qual, floors, do_bloom);
    let (kmers, histovec, used_min_count, chosen_min_qual, genomic_peak) =
        finish_map_counter::<IntT>(shards, qual, floors, sketch, do_fit, do_bloom, out_path);

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

    PreprocessedK {
        k,
        kmers,
        histovec,
        used_min_count,
        chosen_min_qual,
        genomic_peak,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(target_family = "wasm"))]
    use crate::spectrum_fitter::{ErrorModel, ErrorParams, GenomeModel};
    use nohash_hasher::NoHashHasher;
    use std::{collections::HashMap, hash::BuildHasherDefault};

    #[test]
    fn legacy_histogram_range_remains_pinned() {
        assert_eq!(LEGACY_HISTO_RANGE, 500);
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn explicit_cutoff_diagnostics_do_not_contain_a_fit() {
        let spectrum = vec![0u32; MAXSIZEHISTO];
        let estimate = SpectrumEstimate {
            valley: 12,
            genomic_peak: 95,
            ..SpectrumEstimate::default()
        };
        let diagnostics = SpectrumPlotDiagnostics::new(&spectrum, &estimate, None, false);
        assert_eq!(diagnostics.valley, Some(12));
        assert_eq!(diagnostics.empirical_peak, Some(95));
        assert!(!diagnostics.fit_attempted);
        assert!(diagnostics.fit.is_none());
        assert!(diagnostics.shadow_floors.is_none());
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn native_plot_range_tracks_fit_data_and_markers() {
        let spectrum = vec![0u32; MAXSIZEHISTO];
        let estimate = SpectrumEstimate {
            genomic_peak: 95,
            ..SpectrumEstimate::default()
        };
        let diagnostics = SpectrumPlotDiagnostics::new(&spectrum, &estimate, None, true);
        assert_eq!(native_plot_end(spectrum.len(), &diagnostics, 12), 570);

        let estimate = SpectrumEstimate {
            genomic_peak: 15,
            ..SpectrumEstimate::default()
        };
        let diagnostics = SpectrumPlotDiagnostics::new(&spectrum, &estimate, None, true);
        assert_eq!(native_plot_end(spectrum.len(), &diagnostics, 2), 90);
        assert_eq!(native_plot_end(spectrum.len(), &diagnostics, 150), 150);

        let fit = NativeSpectrumFit::for_test(
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
            321,
        );
        let diagnostics = SpectrumPlotDiagnostics::new(&spectrum, &estimate, Some(fit), true);
        assert_eq!(native_plot_end(spectrum.len(), &diagnostics, 2), 321);

        let empty_diagnostics = SpectrumPlotDiagnostics::default();
        assert_eq!(
            native_plot_end(spectrum.len(), &empty_diagnostics, 2),
            MIN_NATIVE_PLOT_RANGE
        );
        assert_eq!(
            native_plot_end(20, &diagnostics, u16::MAX),
            19,
            "the saturating terminal histogram bin is not plotted"
        );
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn plot_markers_distinguish_cutoff_and_shadow_floors() {
        let spectrum = vec![0u32; MAXSIZEHISTO];
        let estimate = SpectrumEstimate {
            genomic_peak: 95,
            min_count: 12,
            ..SpectrumEstimate::default()
        };
        let fit = NativeSpectrumFit::for_test(
            ErrorModel::SingletonPareto,
            GenomeModel::NegativeBinomial,
            [0.9, 0.09, 0.01],
            95.0,
            6.0,
            ErrorParams {
                singleton_probability: 0.85,
                tail_exponent: 2.0,
                weibull_shape: None,
            },
            10.0e6,
            570,
        );
        let diagnostics = SpectrumPlotDiagnostics::new(&spectrum, &estimate, Some(fit), true);
        let markers = plot_markers(&diagnostics, 17);
        let cutoff = markers.last().expect("the cutoff marker is always last");
        assert_eq!(cutoff.x, 17.0);
        assert_eq!(cutoff.colour, BLACK);
        assert_eq!(cutoff.line_style, MarkerLineStyle::Solid);
        assert_eq!(cutoff.label, "Minimum count used = 17");

        let shadows: Vec<_> = markers
            .iter()
            .filter(|marker| marker.label.starts_with("Shadow error floor"))
            .collect();
        assert!(!shadows.is_empty());
        assert!(shadows.iter().all(|marker| {
            marker.colour == RGBColor(105, 105, 105) && marker.line_style == MarkerLineStyle::Dashed
        }));
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
        let fit = NativeSpectrumFit::for_test(
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
        let diagnostics = SpectrumPlotDiagnostics::new(&decision, &estimate, Some(fit), true);

        for (suffix, raw) in [("exact", None), ("bloom", Some(raw_bloom.as_slice()))] {
            let path = std::env::temp_dir().join(format!(
                "sparrowhawk-spectrum-{suffix}-{}.png",
                std::process::id()
            ));
            let svg_path = path.with_extension("svg");
            plot_kmer_histogram(&decision, raw, &diagnostics, 12, &path);
            assert!(std::fs::metadata(&path).is_ok_and(|metadata| metadata.len() > 0));
            assert!(std::fs::metadata(&svg_path).is_ok_and(|metadata| metadata.len() > 0));
            let svg = std::fs::read_to_string(&svg_path).expect("read spectrum SVG");
            assert!(svg.contains("<svg"));
            assert!(svg.contains("Counts"));
            assert!(svg.contains("k-mer frequency"));
            assert!(svg.contains("Minimum count used = 12"));
            std::fs::remove_file(path).expect("remove temporary spectrum plot");
            std::fs::remove_file(svg_path).expect("remove temporary vector spectrum plot");
        }
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn shadow_diagnostics_count_bins_without_changing_the_cutoff() {
        let mut spectrum = vec![0u32; MAXSIZEHISTO];
        spectrum[1] = 11;
        spectrum[2] = 7;
        spectrum[3] = 5;
        spectrum[4] = 3;
        let fit = NativeSpectrumFit::for_test(
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
        let diagnostics = shadow_floor_diagnostics(&spectrum, 2, &fit);
        for diagnostic in diagnostics {
            let end = usize::from(diagnostic.floor).min(spectrum.len() + 1);
            let expected = if end > 2 {
                spectrum[1..end - 1]
                    .iter()
                    .map(|&count| u64::from(count))
                    .sum()
            } else {
                0
            };
            assert_eq!(diagnostic.observed_distinct_removed, expected);
            assert!(diagnostic.fitted_error_removed >= 0.0);
            assert!(diagnostic.fitted_genome_removed >= 0.0);
        }

        let mut estimate = SpectrumEstimate {
            min_count: 2,
            verdict: Verdict::NoPeakAboveValley,
            ..SpectrumEstimate::default()
        };
        apply_hole_guard(&mut estimate, Some(&fit), &spectrum);
        assert_eq!(estimate.min_count, 2, "shadow floors must not be applied");
    }

    fn empty_countmap() -> HashMap<u64, (u32, u64, u8), BuildHasherDefault<NoHashHasher<u64>>> {
        HashMap::with_hasher(BuildHasherDefault::default())
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
        let (shards, _) = bulk_preprocessing_standalone_cpu::<u64, _>(
            &mut iters,
            k,
            &qual,
            Some(&ladder[..]),
            false,
        );

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
        let (shards, _) =
            bulk_preprocessing_standalone_cpu::<u64, _>(&mut iters, k, qual, Some(ladder), true);
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
    }

    /// Shard assignment is fixed and each shard replays in record order, so rayon's scheduling cannot
    /// change the answer.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn sharded_bloom_counting_is_reproducible() {
        let ladder = [0u8, 11, 25];
        let qual = QualOpts {
            min_count: 2,
            min_qual: 25,
        };
        let first = bloom_shards(counter_test_reads(), 11, &qual, &ladder);
        let second = bloom_shards(counter_test_reads(), 11, &qual, &ladder);
        assert_eq!(first, second);
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

    /// The shard count must be a power of two and within its bounds at every thread count. A
    /// non-power-of-two silently collapses `shard_of` onto the count below -- 6 would use 2, 12 would
    /// use 4 -- with no error and a balance report that still reads as healthy.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn the_shard_count_is_always_a_usable_power_of_two() {
        for threads in 1..=64usize {
            let n = (threads * SHARDS_PER_THREAD)
                .next_power_of_two()
                .clamp(MIN_COUNTMAP_SHARDS, MAX_COUNTMAP_SHARDS);
            assert!(n.is_power_of_two(), "{threads} threads gave {n} shards");
            assert!((MIN_COUNTMAP_SHARDS..=MAX_COUNTMAP_SHARDS).contains(&n));
            assert!(
                n >= threads.min(MAX_COUNTMAP_SHARDS),
                "{threads} threads starve at {n} shards"
            );
        }
        let shards_at = |t: usize| {
            (t * SHARDS_PER_THREAD)
                .next_power_of_two()
                .clamp(MIN_COUNTMAP_SHARDS, MAX_COUNTMAP_SHARDS)
        };
        // Every thread count the sweep or a workstation uses sits on the floor, which is where the
        // absorb measurement put the knee. Only a machine with more cores than that scales past it.
        assert_eq!(shards_at(4), MIN_COUNTMAP_SHARDS);
        assert_eq!(shards_at(64), MIN_COUNTMAP_SHARDS);
        assert_eq!(shards_at(128), 128);
        assert_eq!(shards_at(4096), MAX_COUNTMAP_SHARDS);
    }

    /// Whatever the count, every k-mer must land in the shard `shard_of` claims, or the lock-free
    /// accumulate is unsound.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn the_live_shard_count_partitions_the_key_space() {
        let n = countmap_shards();
        assert!(n.is_power_of_two() && n >= MIN_COUNTMAP_SHARDS);
        let mut seen = vec![0usize; n];
        for h in canonical_minimum_hashes(200_000) {
            let s = shard_of(h, n);
            assert!(s < n);
            seen[s] += 1;
        }
        assert!(
            seen.iter().all(|&c| c > 0),
            "some shard never receives a key"
        );
    }

    fn empty_themap() -> HashMap<u64, crate::HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>
    {
        HashMap::with_hasher(BuildHasherDefault::default())
    }

    fn empty_dict() -> HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>> {
        HashMap::with_hasher(BuildHasherDefault::default())
    }

    #[test]
    fn initial_bloom_min_count_is_shared_between_modes() {
        let qual = QualOpts {
            min_count: 7,
            min_qual: 0,
        };

        assert_eq!(initial_bloom_min_count(&qual, true), 2);
        assert_eq!(initial_bloom_min_count(&qual, false), 7);
    }

    #[test]
    fn histogram_zero_count_uses_first_bin() {
        let mut histovec = vec![0u32; MAXSIZEHISTO];
        add_to_histogram(&mut histovec, 0);
        assert_eq!(histovec[0], 1);
    }

    #[test]
    fn update_countmap_single_entry() {
        let input = vec![(100u64, 200u64, 1u8)];
        let mut cmap = empty_countmap();
        update_countmap(&input, &mut cmap);
        assert_eq!(cmap[&100].0, 1);
        assert_eq!(cmap[&100].1, 200u64);
        assert_eq!(cmap[&100].2, 1u8);
    }

    #[test]
    fn update_countmap_two_same_hash() {
        let input = vec![(100u64, 200u64, 1u8), (100u64, 200u64, 1u8)];
        let mut cmap = empty_countmap();
        update_countmap(&input, &mut cmap);
        assert_eq!(cmap[&100].0, 2);
    }

    #[test]
    fn update_countmap_two_different_hashes() {
        let input = vec![(100u64, 200u64, 1u8), (200u64, 100u64, 2u8)];
        let mut cmap = empty_countmap();
        update_countmap(&input, &mut cmap);
        assert_eq!(cmap[&100].0, 1);
        assert_eq!(cmap[&200].0, 1);
    }

    #[test]
    fn update_countmap_accumulates_across_calls() {
        // Two calls: first adds 2, second adds 1 → total 3
        let input1 = vec![(42u64, 0u64, 0u8), (42u64, 0u64, 0u8)];
        let input2 = vec![(42u64, 0u64, 0u8)];
        let mut cmap = empty_countmap();
        update_countmap(&input1, &mut cmap);
        update_countmap(&input2, &mut cmap);
        assert_eq!(cmap[&42].0, 3);
    }

    #[test]
    fn update_countmap_run_of_five() {
        let input: Vec<_> = (0..5).map(|_| (7u64, 8u64, 0u8)).collect();
        let mut cmap = empty_countmap();
        update_countmap(&input, &mut cmap);
        assert_eq!(cmap[&7].0, 5);
    }

    #[test]
    fn drain_countmap_above_threshold_included() {
        let mut cmap = empty_countmap();
        cmap.insert(1u64, (5u32, 2u64, 0u8)); // count=5 >= minc=3 → in themap
        let mut themap = empty_themap();
        let mut outdict = empty_dict();
        let mut minmaxdict = empty_dict();
        drain_countmap_into_themap::<u64>(
            &mut cmap,
            &mut themap,
            &mut outdict,
            &mut minmaxdict,
            3,
            None,
        );
        assert!(themap.contains_key(&1));
    }

    #[test]
    fn drain_countmap_below_threshold_excluded() {
        let mut cmap = empty_countmap();
        cmap.insert(2u64, (2u32, 3u64, 0u8)); // count=2 < minc=3 → removed from outdict
        let mut themap = empty_themap();
        let mut outdict = empty_dict();
        outdict.insert(2u64, 99u64); // should be removed
        let mut minmaxdict = empty_dict();
        minmaxdict.insert(3u64, 2u64); // hnc → hc, should be removed
        drain_countmap_into_themap::<u64>(
            &mut cmap,
            &mut themap,
            &mut outdict,
            &mut minmaxdict,
            3,
            None,
        );
        assert!(!themap.contains_key(&2));
        assert!(!outdict.contains_key(&2));
        assert!(!minmaxdict.contains_key(&3));
    }

    #[test]
    fn drain_countmap_boundary_equal_minc() {
        // count == minc → included (>= check)
        let mut cmap = empty_countmap();
        cmap.insert(5u64, (3u32, 0u64, 0u8));
        let mut themap = empty_themap();
        let mut outdict = empty_dict();
        let mut minmaxdict = empty_dict();
        drain_countmap_into_themap::<u64>(
            &mut cmap,
            &mut themap,
            &mut outdict,
            &mut minmaxdict,
            3,
            None,
        );
        assert!(themap.contains_key(&5));
    }

    #[test]
    fn drain_countmap_with_histogram() {
        let mut cmap = empty_countmap();
        cmap.insert(1u64, (5u32, 0u64, 0u8)); // count=5 → histovec[4]
        cmap.insert(2u64, (10u32, 0u64, 0u8)); // count=10 → histovec[9]
        let mut themap = empty_themap();
        let mut outdict = empty_dict();
        let mut minmaxdict = empty_dict();
        let mut histovec = vec![0u32; MAXSIZEHISTO];
        drain_countmap_into_themap::<u64>(
            &mut cmap,
            &mut themap,
            &mut outdict,
            &mut minmaxdict,
            1,
            Some(&mut histovec),
        );
        assert!(histovec[4] > 0, "count=5 should be at histovec[4]");
        assert!(histovec[9] > 0, "count=10 should be at histovec[9]");
    }

    /// A bimodal spectrum: a tall error peak at count 1-2 and the real single-copy peak further out.
    /// `coverage_peak` must return the *count* of the second peak, ignoring the first.
    #[test]
    fn coverage_peak_finds_the_mode_above_the_error_peak() {
        for expected in [3usize, 10, 30, 113, 498] {
            let mut h = vec![0u32; MAXSIZEHISTO];
            h[0] = 1_000_000; // count 1: errors, must be ignored
            h[1] = 200_000; // count 2: still errors
            h[expected - 1] = 50_000; // the genuine single-copy peak
            assert_eq!(
                coverage_peak(&h),
                expected,
                "genomic_peak planted at count {expected}"
            );
        }
    }

    /// Ties keep the lowest count, and a spectrum with nothing above the error peak falls back to 2 —
    /// so the peak-derived floor stays well defined even for a degenerate histogram.
    #[test]
    fn coverage_peak_is_deterministic_and_has_a_floor() {
        let mut h = vec![0u32; MAXSIZEHISTO];
        h[9] = 42;
        h[19] = 42; // equal height: the lower count wins
        assert_eq!(coverage_peak(&h), 10);

        let mut only_errors = vec![0u32; MAXSIZEHISTO];
        only_errors[0] = 99;
        only_errors[1] = 7;
        assert_eq!(coverage_peak(&only_errors), 2);
    }

    /// The last bin saturates (it absorbs every count >= MAXSIZEHISTO), so it must not be mistaken for
    /// a peak — the current mixture fit excludes it for the same reason.
    #[test]
    fn coverage_peak_ignores_the_edge_of_its_pinned_window() {
        let mut h = vec![0u32; MAXSIZEHISTO];
        h[LEGACY_HISTO_RANGE - 1] = 1_000_000;
        h[29] = 10;
        assert_eq!(coverage_peak(&h), 30);
    }

    /// Widening the histogram must not move the legacy readers: they see counts 1..=499, and
    /// `add_to_histogram` puts count `c` at index `c-1` regardless of how long the vector is.
    #[test]
    fn widening_the_histogram_leaves_the_pinned_window_alone() {
        let mut narrow = vec![0u32; LEGACY_HISTO_RANGE];
        let mut wide = vec![0u32; MAXSIZEHISTO];
        for c in [1u32, 2, 3, 47, 163, 499] {
            add_to_histogram(&mut narrow[..], c);
            add_to_histogram(&mut wide[..], c);
        }
        // Counts below the old ceiling land identically; only what used to saturate now moves.
        assert_eq!(
            narrow[..LEGACY_HISTO_RANGE - 1],
            wide[..LEGACY_HISTO_RANGE - 1]
        );
        assert_eq!(coverage_peak(&narrow), coverage_peak(&wide));
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

    /// Fraction of a Poisson(`lam`) genome deleted by cutting below `min_count`.
    fn genome_loss(lam: f64, min_count: u16) -> f64 {
        (0..min_count as usize).map(|c| poisson(c, lam)).sum()
    }

    /// The case that breaks today: at 500x the error lobe outvotes the genome lobe, so the global
    /// argmax returns ~3 and `mode/8` yields 2, while the valley finds the real genomic peak out at ~500.
    #[test]
    fn valley_estimator_survives_a_deep_library() {
        let h = synthetic_spectrum(500.0);
        assert!(
            coverage_peak(&h) < 10,
            "the old argmax should be fooled here; that is the bug being fixed"
        );
        let e = estimate_by_valley(&h);
        assert_eq!(e.verdict, Verdict::Ok);
        assert!(
            (450..550).contains(&e.genomic_peak),
            "genomic_peak was {}",
            e.genomic_peak
        );
        assert!(
            (10..40).contains(&e.min_count),
            "min_count was {}",
            e.min_count
        );
        assert!(genome_loss(500.0, e.min_count) < 1e-6);
    }

    /// On a smooth spectrum the walk stops at the valley and widening the search cannot move it. This
    /// does *not* hold on real data: where the error tail is noisy the walk stops at the first bump
    /// and the spectrum keeps dipping afterwards, which is the case this change exists to correct.
    #[test]
    fn the_valley_equals_the_walk_on_a_clean_spectrum() {
        for lam in [20.0, 60.0, 150.0, 500.0] {
            let h = synthetic_spectrum(lam);
            let valley_seed = find_valley_seed(&h).expect("a clean spectrum turns back up");
            let genomic_peak = find_genomic_peak(&h, valley_seed);
            assert_eq!(
                find_valley(&h, genomic_peak),
                valley_seed,
                "lam {lam}: valley moved (valley_seed {valley_seed}, genomic_peak {genomic_peak})"
            );
        }
    }

    #[test]
    fn a_seed_past_the_genome_lobe_still_finds_the_valley() {
        let h = synthetic_spectrum(500.0);
        let truth = find_valley_seed(&h).unwrap();
        for overshoot in [700, 1500, 4000] {
            assert_eq!(
                find_valley(&h, overshoot),
                truth,
                "a genomic_peak of {overshoot} moved the valley away from {truth}"
            );
        }
    }

    /// A library with no separable lobe must still bail: widening the search must not manufacture a
    /// valley where the spectrum is one monotone slide.
    #[test]
    fn a_flat_library_still_bails() {
        let h = synthetic_spectrum(4.0);
        let e = estimate_by_valley(&h);
        assert_ne!(e.verdict, Verdict::Ok, "min_count was {}", e.min_count);
        assert_eq!(e.min_count, UNRESOLVED_MINCOUNT);
    }

    /// A deep library is mostly error k-mers by *count* and mostly genome by *sequence*. Weighing
    /// distinct k-mers instead of instances would reject this healthy spectrum.
    #[test]
    fn a_deep_library_is_not_rejected_for_being_mostly_errors() {
        let h = synthetic_spectrum(500.0);
        let distinct_total: u64 = h.iter().map(|&n| n as u64).sum();
        let distinct_genome: u64 = h[100..].iter().map(|&n| n as u64).sum();
        assert!(
            distinct_genome * 50 < distinct_total,
            "the genome should be a small minority of distinct k-mers here"
        );
        assert_eq!(estimate_by_valley(&h).verdict, Verdict::Ok);
    }

    /// Where the lobes merge the cutoff must not eat the genome, whether the refusal comes from a
    /// verdict or from the loss guard clamping it. This is the invariant that matters.
    #[test]
    fn valley_estimator_does_not_cut_when_the_lobes_merge() {
        for lam in [3.0, 5.0, 8.0, 9.0] {
            let e = estimate_by_valley(&synthetic_spectrum(lam));
            assert_eq!(
                e.min_count, UNRESOLVED_MINCOUNT,
                "lambda {lam} cut at {} ({:?})",
                e.min_count, e.verdict
            );
        }
    }

    /// Every refusal must name itself, and the resolved case must still explain itself when the guard
    /// is what collapsed the cutoff — that clause is the one the old string could not express.
    #[test]
    fn every_verdict_has_a_distinct_reason() {
        let all = [
            Verdict::NeverTurnsUp,
            Verdict::NoPeakAboveValley,
            Verdict::TooFewCandidateKmers,
            Verdict::PeakNotClearOfValley,
            Verdict::Ok,
        ];
        for (i, a) in all.iter().enumerate() {
            assert!(!a.reason().is_empty(), "{a:?} has no reason");
            for b in &all[i + 1..] {
                assert_ne!(a.reason(), b.reason(), "{a:?} and {b:?} share a reason");
            }
        }
        assert_eq!(all.iter().filter(|v| v.is_ok()).count(), 1);
        assert!(!Verdict::default().is_ok(), "the default must fail closed");
    }

    /// Each way the lobes can merge must be named accurately: at 3-5x no genomic peak clears the valley at all,
    /// and at 6-7x there is a genomic peak but it does not stand clear of the valley.
    #[test]
    fn valley_estimator_names_the_reason_the_lobes_merged() {
        for lam in [3.0, 4.0, 5.0] {
            assert_eq!(
                estimate_by_valley(&synthetic_spectrum(lam)).verdict,
                Verdict::NoPeakAboveValley,
                "lambda {lam}"
            );
        }
        for lam in [6.0, 7.0] {
            assert_eq!(
                estimate_by_valley(&synthetic_spectrum(lam)).verdict,
                Verdict::PeakNotClearOfValley,
                "lambda {lam}"
            );
        }
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
        let e = estimate_by_valley(&h);
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
        h[0] = 1_000_000; // error spike at count 1
        h[5] = 100_000; // genome spike at count 6, empty valley between
        let e = estimate_by_valley(&h);
        assert_eq!(
            e.verdict,
            Verdict::Ok,
            "valley {} genomic_peak {} gp_to_v_ratio {:.1} valley_to_peak_xratio {:.3}",
            e.valley,
            e.genomic_peak,
            e.gp_to_v_ratio,
            e.valley_to_peak_xratio
        );
        assert!(
            e.valley_to_peak_xratio > 0.6,
            "the point of the case is that it is crowded"
        );
    }

    /// Through the working range the cutoff must clear the errors without eating the genome.
    #[test]
    fn valley_estimator_is_safe_through_the_working_range() {
        for lam in [10.0, 15.0, 20.0, 50.0, 100.0, 250.0] {
            let e = estimate_by_valley(&synthetic_spectrum(lam));
            assert_eq!(e.verdict, Verdict::Ok, "lambda {lam}");
            assert!(e.min_count >= 2, "lambda {lam}");
            assert!(
                genome_loss(lam, e.min_count) < 0.01,
                "lambda {lam} lost {:.2} % of the genome at min_count {}",
                genome_loss(lam, e.min_count) * 100.0,
                e.min_count
            );
        }
    }

    /// The reason for widening the histogram: a genomic peak past the old 500-bin ceiling must still be found.
    #[test]
    fn valley_estimator_finds_a_peak_beyond_the_old_ceiling() {
        let mut h = vec![0u32; MAXSIZEHISTO];
        for c in 1..40 {
            h[c - 1] = 1_000_000 / (c as u32 * c as u32); // a decaying error lobe
        }
        for c in 5500..6500 {
            h[c - 1] = 20_000; // a genome lobe far outside LEGACY_HISTO_RANGE
        }
        let e = estimate_by_valley(&h);
        assert_eq!(e.verdict, Verdict::Ok);
        assert!(
            e.genomic_peak >= LEGACY_HISTO_RANGE,
            "genomic_peak was {}",
            e.genomic_peak
        );
        assert!(coverage_peak(&h) < LEGACY_HISTO_RANGE);
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
            let valley = find_valley_seed(&h).unwrap();
            let genomic_peak = find_genomic_peak(&h, valley);
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

    /// Neither half is safe alone, so the guard takes the tighter: the Poisson bound binds at low
    /// coverage, where errors above the valley loosen the measured one, and the measured bound binds at
    /// depth, where the lobe is far too wide for a Poisson tail.
    #[test]
    fn loss_guard_takes_the_tighter_of_its_two_bounds() {
        // Low coverage: the Poisson bound is the strict one.
        let h = synthetic_spectrum(10.0);
        let valley = find_valley_seed(&h).unwrap();
        let genomic_peak = find_genomic_peak(&h, valley);
        assert!(poisson_guard(genomic_peak) < measured_guard(&h, valley, genomic_peak));
        assert_eq!(
            measured_guard(&h, valley, genomic_peak).min(poisson_guard(genomic_peak)),
            poisson_guard(genomic_peak)
        );

        // Depth, with a lobe as wide as the real libraries: the measured bound is the strict one.
        let mut wide = vec![0u32; MAXSIZEHISTO];
        for c in 1..MAXSIZEHISTO {
            let z = (c as f64 - 200.0) / 45.0;
            wide[c - 1] = (1_000_000.0 * (-0.5 * z * z).exp()) as u32;
        }
        assert!(measured_guard(&wide, 60, 200) < poisson_guard(200));
        assert_eq!(
            measured_guard(&wide, 60, 200).min(poisson_guard(200)),
            measured_guard(&wide, 60, 200)
        );
    }

    /// The reason the Poisson tail was dropped: on a lobe as wide as the real ones it permits a cutoff
    /// far above what the data can afford. Mean 200, sd ~45, against a Poisson sd of 14.
    #[test]
    fn loss_guard_is_tighter_than_a_poisson_tail_on_an_overdispersed_lobe() {
        let mut h = vec![0u32; MAXSIZEHISTO];
        let (mu, sd) = (200.0f64, 45.0f64);
        for c in 1..MAXSIZEHISTO {
            let z = (c as f64 - mu) / sd;
            h[c - 1] = (1_000_000.0 * (-0.5 * z * z).exp()) as u32;
        }
        let valley = 60;
        let m = measured_guard(&h, valley, 200).min(poisson_guard(200)) as usize;
        // A Poisson(200) tail would allow ~168; the measured 1 % quantile of this lobe is far lower.
        assert!(
            m < 150,
            "guard returned {m}, no tighter than a Poisson tail"
        );
        assert!(
            measured_loss(&h, valley, m) <= MAX_GENOME_LOSS,
            "guard {m} cost {:.4}",
            measured_loss(&h, valley, m)
        );
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

    /// The swap: on a deep library the valley and the mixture fit disagree, and it is the valley that
    /// must come out of `choose_min_count`.
    #[test]
    fn choose_min_count_returns_the_valley_not_the_fit() {
        let h = synthetic_spectrum(500.0);
        let e = estimate_by_valley(&h);
        assert_eq!(e.verdict, Verdict::Ok);
        assert_eq!(choose_min_count(&h), e.min_count);
        // ...and that is emphatically not what the old path would have produced.
        let floor = ((coverage_peak(&h) as f64 / 8.0).round() as u16).max(2);
        assert!(
            e.min_count > floor,
            "the old floor was {floor} and the valley {}; the swap changes nothing here",
            e.min_count
        );
    }

    // ---- the sketch -------------------------------------------------------------------------------

    /// A modest bimodal spectrum as (count, number of distinct k-mers) pairs: an error lobe decaying
    /// from count 1 and a genome lobe around `genomic_peak`.
    fn bimodal(genomic_peak: usize) -> Vec<(u32, usize)> {
        let mut out: Vec<(u32, usize)> = (1..=5).map(|c| (c as u32, 4000 / c)).collect();
        for c in (genomic_peak - 12)..=(genomic_peak + 12) {
            let d = (c as f64 - genomic_peak as f64) / 5.0;
            out.push((c as u32, (3000.0 * (-0.5 * d * d).exp()) as usize));
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
                lowest = lowest.min(NtHashIterator::new(&seq, k, true).curr_hash());
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
    fn shrinking_at_the_hash_floor_halves_the_sample_rather_than_erasing_it() {
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

    #[test]
    fn shrinking_keeps_survivors_exact() {
        let mut s = SpectrumSketch::new();
        // Both sit in the range the hash can actually reach: subsampling is of the interval above
        // [`HASH_FLOOR`], so a fixture below it would never be admitted in the first place.
        let low = HASH_FLOOR + 1;
        let high = u64::MAX - 1;
        for _ in 0..5 {
            s.observe(low, 0);
        }
        for _ in 0..9 {
            s.observe(high, 0);
        }
        assert_eq!(s.counts[&low][0], 5);
        s.shrink();
        assert_eq!(s.counts[&low][0], 5, "a survivor must keep its count");
        assert!(
            !s.counts.contains_key(&high),
            "a hash above the threshold must go"
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

        let (r, s) = (estimate_by_valley(&raw), estimate_by_valley(&scaled));
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
        const BASES: [u8; 4] = [b'A', b'C', b'G', b'T'];
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
        let (shards, sketch) = bulk_preprocessing_standalone_cpu::<u64, _>(
            &mut iters,
            k,
            qual,
            Some(ladder),
            do_bloom,
        );
        finish_map_counter::<u64>(
            shards,
            qual,
            Some(ladder),
            sketch,
            true,
            do_bloom,
            &mut None,
        )
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

    /// Doubling every bin doubles the distinct k-mers without moving the mode. `distinct_above` must see
    /// that and `genomic_peak` must not — the difference a depth criterion is blind to.
    #[test]
    fn distinct_above_sees_breadth_that_the_peak_does_not() {
        let narrow = histo(&bimodal(30));
        let wide: Vec<u32> = narrow.iter().map(|n| n * 2).collect();
        let (a, b) = (estimate_by_valley(&narrow), estimate_by_valley(&wide));
        assert_eq!(a.genomic_peak, b.genomic_peak, "depth is unchanged");
        assert_eq!(b.distinct_above, 2 * a.distinct_above, "breadth is not");
    }

    /// A clean separation at a genomic peak of 18 is still starvation: the floor, not the library, may be what
    /// made it shallow, so a materially deeper floor wins even though the strict one resolved.
    #[test]
    fn a_shallow_resolving_spectrum_still_loosens() {
        let shallow = histo(&bimodal(18));
        let strict = estimate_by_valley(&shallow);
        assert!(
            resolves(&strict),
            "the premise: the strict floor does resolve"
        );
        assert!(
            strict.genomic_peak < MIN_USEFUL_COVERAGE,
            "genomic_peak was {}",
            strict.genomic_peak
        );

        let sketch = sketch_with(1, &bimodal(40));
        let (_, floor, _, _) = choose_min_count_and_floor(&shallow, &sketch, &[0u8, 11, 25]);
        assert_eq!(floor, 11, "a 2.2x deeper spectrum justifies the recount");
    }

    /// The same shallow spectrum, but loosening barely moves the genomic peak: depth rather than the floor is
    /// the limit, so the second pass is not worth paying for and the strict cutoff stands.
    #[test]
    fn a_shallow_spectrum_with_no_gain_keeps_the_strict_floor() {
        let shallow = histo(&bimodal(18));
        let strict = estimate_by_valley(&shallow);
        // 19 against 18 is a gain of 1.06, under `MIN_COVERAGE_GAIN`.
        let sketch = sketch_with(1, &bimodal(19));
        let (minc, floor, _, _) = choose_min_count_and_floor(&shallow, &sketch, &[0u8, 11, 25]);
        assert_eq!(floor, 25, "no material gain, so no recount");
        assert_eq!(
            minc, strict.min_count,
            "and the strict cutoff is kept, not the fallback"
        );
        assert_ne!(minc, UNRESOLVED_MINCOUNT);
    }

    /// A sketch where the loose rung sees `extra` further occurrences of every k-mer, which is what
    /// dropping the floor physically does: the same k-mers, seen more often.
    fn sketch_deepening(spectrum: &[(u32, usize)], extra: u32) -> SpectrumSketch {
        let mut s = SpectrumSketch::new();
        let mut h = 1u64;
        for &(count, n) in spectrum {
            for _ in 0..n {
                let mut c = [0u32; MAX_GROUPS];
                c[1] = count;
                c[0] = count * extra;
                s.counts.insert(h, c);
                h += 1;
            }
        }
        s
    }

    /// The strict floor does not resolve at all and the middle rung does — but at a genomic peak of 15 it is
    /// still starved, so the walk must keep loosening instead of settling for the first rung that
    /// merely separates. This is art at k=71/81, which stopped at its B rung and stayed fragmented.
    #[test]
    fn a_resolving_but_starved_rung_keeps_loosening() {
        let flat = vec![1000u32; MAXSIZEHISTO];
        let sketch = sketch_deepening(&bimodal(15), 2);
        let (_, floor, _, _) = choose_min_count_and_floor(&flat, &sketch, &[0u8, 11, 25]);
        assert_eq!(floor, 0, "a 3x deeper rung is there and must be taken");
    }

    /// The same shape, but the loosest rung is barely deeper: no rung reaches useful depth, so the walk
    /// settles for the strictest one that separated rather than dropping the filter for nothing.
    #[test]
    fn a_starved_ladder_settles_for_the_strictest_that_separated() {
        let flat = vec![1000u32; MAXSIZEHISTO];
        let mut sketch = sketch_deepening(&bimodal(15), 0);
        // Group 0 sees a tenth again as many occurrences: real, but under `MIN_COVERAGE_GAIN`.
        for c in sketch.counts.values_mut() {
            c[0] = c[1] / 10;
        }
        let (minc, floor, _, _) = choose_min_count_and_floor(&flat, &sketch, &[0u8, 11, 25]);
        assert_eq!(
            floor, 11,
            "no material gain below it, so the walk stops here"
        );
        assert_ne!(
            minc, UNRESOLVED_MINCOUNT,
            "and keeps the cutoff that rung measured"
        );
    }

    /// The stranding case, from a starved library whose floors sit at peaks 15/17/18. Accepting the
    /// middle floor must not raise the bar the loosest one has to clear.
    #[test]
    fn a_middle_floor_does_not_block_a_looser_one() {
        let shallow = histo(&bimodal(15));
        let mut sketch = sketch_deepening(&bimodal(15), 0);
        for c in sketch.counts.values_mut() {
            let base = c[1];
            c[1] = base + base / 7; // the middle floor, a little deeper
            c[0] = base / 12; // the loosest, deeper still but only just
        }
        let (_, floor, _, _) = choose_min_count_and_floor(&shallow, &sketch, &[0u8, 15, 20]);
        assert_eq!(
            floor, 0,
            "the loosest qualifying floor wins when every floor is starved"
        );
    }

    /// Order must not matter: the floor chosen is a property of the candidates, not of the walk that
    /// visits them.
    #[test]
    fn a_deep_enough_floor_is_taken_at_its_strictest() {
        let flat = vec![1000u32; MAXSIZEHISTO];
        let sketch = sketch_with(1, &bimodal(40));
        let (_, floor, _, _) = choose_min_count_and_floor(&flat, &sketch, &[0u8, 11, 25]);
        assert_eq!(
            floor, 11,
            "both floors clear MIN_USEFUL_COVERAGE, so the stricter one wins"
        );
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
        match peak_of(&estimate_by_valley(&h), &h) {
            PeakSource::Fitted(p) => assert!(
                (100..=140).contains(&p),
                "fitted peak {p} should sit near the true 120x"
            ),
            other => panic!("expected a fitted peak, got {other:?}"),
        }
    }

    /// The case this fallback exists for. A merged spectrum must never report `Fitted`, because
    /// `genomic_peak` is then an argmax above a valley that was never found: the error lobe.
    #[test]
    fn an_unresolved_spectrum_falls_back_to_the_median() {
        let flat = vec![1000u32; MAXSIZEHISTO];
        assert!(
            !estimate_by_valley(&flat).verdict.is_ok(),
            "fixture must not resolve"
        );
        assert!(
            matches!(
                peak_of(&estimate_by_valley(&flat), &flat),
                PeakSource::Fallback(_)
            ),
            "a merged spectrum must fall back, never report a fitted peak"
        );
    }

    /// The median must land in the genomic lobe, not the error lobe, even though error k-mers vastly
    /// outnumber genomic ones: each carries only a few occurrences, so they hold little of the mass.
    ///
    /// This also pins the indexing. `histovec[c - 1]` is the count-`c` bin, and an estimator reading
    /// the index as the count would report one less here.
    #[test]
    fn the_occurrence_weighted_median_ignores_the_error_lobe() {
        let mut h = vec![0u32; MAXSIZEHISTO];
        h[0] = 5_000_000; // 5M singleton error k-mers: many, but one occurrence each
        h[39] = 1_000_000; // 1M genomic k-mers at 40x: fewer, but 40 occurrences each
        let median = occurrence_weighted_median(&h).expect("a non-empty spectrum has a median");
        assert_eq!(median, 40, "the median must name the genomic lobe's count");
    }

    /// A single bin, so the answer is forced and the off-by-one has nowhere to hide.
    #[test]
    fn the_median_of_one_bin_is_that_bins_count() {
        let mut h = vec![0u32; MAXSIZEHISTO];
        h[6] = 10; // ten k-mers seen 7 times each
        assert_eq!(occurrence_weighted_median(&h), Some(7));
    }

    /// Nothing counted at all: no coverage can be claimed, fitted or otherwise.
    #[test]
    fn an_empty_spectrum_reports_unknown() {
        let empty = vec![0u32; MAXSIZEHISTO];
        assert_eq!(occurrence_weighted_median(&empty), None);
        assert_eq!(
            peak_of(&estimate_by_valley(&empty), &empty),
            PeakSource::Unknown
        );
    }
}

#[cfg(target_family = "wasm")]
/// Main preprocessing function for wasm
pub fn preprocessing_wasm<IntT>(
    file1: &mut WebSysFile,
    file2: Option<&mut WebSysFile>,
    k: usize,
    qual: &QualOpts,
    csize: usize,
    do_bloom: bool,
    do_fit: bool,
) -> (
    HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
    Option<HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>>,
    HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
    Vec<u32>,
    u16,
)
where
    IntT: for<'a> UInt<'a>,
{
    if do_bloom {
        // Build indexes
        logw("Processing using a Bloom filter", Some("info"));

        let (thedict, maxmindict, themap, histovec, used_min_count) =
            bloom_filter_preprocessing_wasm::<IntT>(file1, file2, k, qual, do_fit);
        (themap, Some(thedict), maxmindict, histovec, used_min_count)
    } else {
        // "No chunking" is one unbounded chunk. The guard matters: `i_record >= 0` holds on every
        // record, so passing 0 through would sort and count after every single read.
        let csize = if csize == 0 { usize::MAX } else { csize };
        if csize == usize::MAX {
            logw("Counting k-mers by sorting, without chunking", Some("info"));
        } else {
            logw(
                format!("Counting k-mers by sorting, in chunks of {csize} records").as_str(),
                Some("info"),
            );
        }

        let mut tmpvec: Vec<(u64, u64, u8)> = Vec::new();
        let (thedict, maxmindict, themap, mut histovec, used_min_count) =
            chunked_processing_wasm::<IntT>(file1, file2, k, qual, &mut tmpvec, csize, do_fit);
        drop(tmpvec);
        histovec.shrink_to_fit();
        (themap, Some(thedict), maxmindict, histovec, used_min_count)
    }
}
