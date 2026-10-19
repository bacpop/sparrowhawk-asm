//! Read preprocessing orchestration and empirical coverage estimation.

pub(crate) mod bloom_filter;
pub(crate) mod filter;
pub mod kmer;
pub(crate) mod math_helpers;
pub(crate) mod models;
pub(crate) mod parsercounter;
pub(crate) mod plotter;
pub mod read_store;
pub(crate) mod sketch;
pub(crate) mod spectrum_fitter;

use crate::graph_works::indexed_kmers::IndexedKmers;
use crate::utils::bit_encoding::UInt;
use crate::utils::clock::Instant;
use crate::utils::reads::{MaybeSend, OwnedRecord};
use crate::QualOpts;
use libm::lgamma;
use std::{error::Error, fmt, path::PathBuf};

use self::filter::choose_cutoff_and_filter;
use self::models::SpectrumFit;
use self::parsercounter::count_kmers;
pub use self::parsercounter::PassInput;
use self::read_store::PackedSeqs;

/// Bins of the k-mer spectrum: bin `c - 1` holds the k-mers seen `c` times, and the last bin every
/// count from `MAXSIZEHISTO` up. For the fitting, we'll use actually MAXSIZEHISTO - 1.
pub(crate) const MAXSIZEHISTO: usize = 8000;

/// Settings for one preprocessing pass at exactly the requested quality floor.
#[derive(Clone, Copy, Debug)]
pub struct PreprocessingOptions {
    /// Numerical PHRED floor and fixed count (when fitting is disabled).
    pub quality: QualOpts,
    /// Use Bloom counting, with possible false-positive counts.
    pub use_bloom: bool,
    /// Fit the minimum count instead of using `quality.min_count`.
    pub fit_min_count: bool,
}

impl Default for PreprocessingOptions {
    fn default() -> Self {
        Self {
            quality: QualOpts {
                min_count: crate::cli::DEFAULT_MINCOUNT,
                min_qual: crate::cli::DEFAULT_MINQUAL,
            },
            use_bloom: true,
            fit_min_count: true,
        }
    }
}

/// Count and filter one k at one quality floor, without recounting or writing diagnostics.
/// Quality bytes are PHRED+33. For automatic quality selection or multiple k values use
/// [`crate::assemble`]. The chosen integer must hold at least `2 * k` bits.
pub fn preprocessing<IntT, I>(
    input: PassInput<'_, I>,
    k: usize,
    options: &PreprocessingOptions,
) -> Result<PreprocessedK<IntT>, crate::AssemblyError>
where
    IntT: for<'a> UInt<'a>,
    I: Iterator<Item = std::io::Result<OwnedRecord>> + MaybeSend,
{
    crate::api::validate_packed_k::<IntT>(k)?;
    crate::cli::validate_bloom_min_count(
        options.use_bloom,
        (!options.fit_min_count).then_some(options.quality.min_count),
    )
    .map_err(|message| crate::AssemblyError::InvalidOptions(message.into()))?;
    preprocessing_pass(
        input,
        k,
        &options.quality,
        None,
        &mut None,
        &mut None,
        options.use_bloom,
        options.fit_min_count,
        None,
        false,
        None,
    )
}

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

/// Genome k-mers the cutoff may delete: this share of those above the valley for the measured guard,
/// and of a Poisson genome for the Poisson one.
const MAX_GENOME_LOSS: f64 = 0.01;

/// Expected single-copy k-mers the cutoff may strand below itself.
const MAX_GENOME_HOLES: f64 = 1.0;

/// Everything the preprocessing of one k value produces. Named fields rather than a tuple, since the
/// backends and the public entry point order these differently.
pub struct PreprocessedK<IntT> {
    /// The k this was built with.
    pub k: usize,
    /// K-mer fields stored once in aligned vectors, plus their hash-to-index lookup.
    pub kmers: IndexedKmers<IntT>,
    /// 8,000-bin spectrum: index `c - 1` counts distinct k-mers seen `c` times.
    /// The last bin includes counts of 8,000 and above.
    pub histovec: Vec<u32>,
    /// the min-count actually applied (fitted, a fallback, or taken from the CLI)
    pub used_min_count: u16,
    /// Selected PHRED floor. The public single-pass component keeps the requested floor;
    /// automatic floor selection and recounting belong to the complete API.
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

/// Read the records, count their k-mers, choose the cutoff and filter by it. `carried`: a multi-k
/// step's previous contigs, put in once the cutoff is chosen. `debug_quality_floors` also evaluates
/// and plots every floor; `prior`: a recount's selection from the sketch, see
/// `filter::keep_refit_or_prior` for the refit policy.
#[allow(clippy::too_many_arguments)]
pub(crate) fn preprocessing_pass<IntT, I>(
    input: PassInput<'_, I>,
    k: usize,
    qual: &QualOpts,
    floors: Option<&[u8]>,
    timevec: &mut Option<&mut Vec<Instant>>,
    out_path: &mut Option<PathBuf>,
    do_bloom: bool,
    do_fit: bool,
    carried: Option<&PackedSeqs>,
    debug_quality_floors: bool,
    prior: Option<(u16, PeakSource)>,
) -> Result<PreprocessedK<IntT>, crate::AssemblyError>
where
    IntT: for<'a> UInt<'a>,
    I: Iterator<Item = std::io::Result<OwnedRecord>> + MaybeSend,
{
    crate::post_state("preprocess:bulk:start");
    log::info!("Starting preprocessing with k = {k}");
    log::info!("Counting k-mers into a hash map, without sorting");
    let min_class = match (&input, floors) {
        (PassInput::Stored(store), None) => {
            store.class_of_floor(qual.min_qual).ok_or_else(|| {
                crate::AssemblyError::InvalidOptions(
                    "floor must belong to the stored ladder".into(),
                )
            })?
        }
        _ => 0,
    };
    let (shards, sketch, floor_has_kmers) =
        count_kmers::<IntT, _>(input, k, qual, floors, min_class, do_bloom, do_fit)?;
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

#[cfg(test)]
mod tests {
    use super::filter::choose_min_count_and_peak;
    use super::*;
    use crate::preprocessing::test_support::*;

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

    /// The area's denominator leaves out singletons, its 5 % threshold is inclusive, and its mirrored
    /// shoulder may reach the last unsaturated count but not the saturating last bin.
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

        assert!(
            PeakAreaSupport::measure(&histogram, 3, 7).is_some(),
            "the right endpoint is count 11, the last unsaturated count"
        );
        assert_eq!(
            PeakAreaSupport::measure(&histogram, 4, 8),
            None,
            "the right endpoint is count 12, the saturating last bin"
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

    /// At extreme depth the genome lobe is real but under 5 % of the non-singleton bins. The primary pair
    /// clears every separation check, so it is the area rule that refuses it: no pair is published,
    /// nothing is fitted, and the cutoff is the sentinel.
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
            let mut primary = SpectrumEstimate::default();
            let (valley, peak, _) = primary
                .find_primary_pair(&h)
                .expect("the primary search finds the genome lobe");
            primary.set_pair(&h, valley, peak);
            assert_eq!(
                primary.verdict,
                Verdict::Ok,
                "lambda {lam}: the separation checks alone accept the pair"
            );
            let area = PeakAreaSupport::measure(&h, valley, peak).expect("the area is measurable");
            assert!(
                !area.passes(),
                "lambda {lam}: area {:.3} must fail the guard",
                area.fraction()
            );

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
        // The primary pair is judged before a refusal clears it, so its verdict is read from a copy.
        let mut primary = SpectrumEstimate::default();
        let (valley, peak, _) = primary
            .find_primary_pair(&h)
            .expect("the walk locks onto a noise crest");
        primary.set_pair(&h, valley, peak);
        assert_eq!(
            primary.verdict,
            Verdict::PeakNotClearOfValley,
            "valley {} ({}) genomic_peak {} ({}) gave gp_to_v_ratio {:.3}",
            primary.valley,
            primary.valley_n,
            primary.genomic_peak,
            primary.genomic_peak_n,
            primary.gp_to_v_ratio
        );
        assert!(primary.gp_to_v_ratio < MIN_GP_TO_V_RATIO);
        // The local fallback finds no lobe either, so nothing resolves.
        let e = SpectrumEstimate::by_valley(&h);
        assert_ne!(e.verdict, Verdict::Ok);
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

    /// The smoothed view averages each bin with [`SMOOTH`] bins either side, so a lone spike spreads
    /// evenly over five.
    #[test]
    fn smoothing_spreads_a_spike_over_five_bins() {
        let mut spike = vec![0u32; 40];
        spike[19] = 100;
        let smooth = smooth_histogram(&spike);
        assert_eq!(&smooth[17..=21], &[20; 5]);
        assert_eq!((smooth[16], smooth[22]), (0, 0));
    }

    /// Above the low-coverage bypass, the pair is found from the raw candidates and refined on the
    /// smoothed spectrum, with the valley under the peak.
    #[test]
    fn moderate_depth_pairs_come_from_raw_candidates_and_smoothed_refinement() {
        for lam in [20.0, 30.0, 60.0, 150.0] {
            let e = SpectrumEstimate::by_valley(&synthetic_spectrum(lam));
            assert!(
                e.raw_valley.is_some() && e.raw_genomic_peak.is_some(),
                "lambda {lam}"
            );
            assert_eq!(
                e.peak_search,
                PeakSearch::SmoothedRefinement,
                "lambda {lam}"
            );
            assert!(e.valley < e.genomic_peak, "lambda {lam}");
        }
    }

    /// Through moderate depth the lobes separate and the cutoff stays inside the genome-loss budget.
    #[test]
    fn the_working_range_resolves_within_the_loss_budget() {
        for lam in [10.0, 15.0, 20.0, 50.0, 100.0] {
            let e = SpectrumEstimate::by_valley(&synthetic_spectrum(lam));
            assert_eq!(e.verdict, Verdict::Ok, "lambda {lam}");
            assert!(
                e.min_count >= 2 && genome_loss(lam, e.min_count) < MAX_GENOME_LOSS,
                "lambda {lam} cut at {}",
                e.min_count
            );
        }
    }

    /// A peak beyond the old 500-bin ceiling is found inside its lobe when the histogram, at its full
    /// size, extends far enough to contain the lobe's mirrored shoulder.
    #[test]
    fn valley_estimator_finds_a_peak_beyond_the_old_ceiling() {
        let mut h = vec![0u32; MAXSIZEHISTO];
        for c in 1..40 {
            h[c - 1] = 1_000_000 / (c as u32 * c as u32); // a decaying error lobe
        }
        let lobe = 1_000..=2_000;
        for c in lobe.clone() {
            // A genome lobe far outside 500, centred on 1,500.
            let z = (c as f64 - 1_500.0) / 150.0;
            h[c - 1] = (20_000.0 * (-0.5 * z * z).exp()) as u32;
        }
        let e = SpectrumEstimate::by_valley(&h);
        assert_eq!(e.verdict, Verdict::Ok);
        assert!(
            lobe.contains(&e.genomic_peak),
            "genomic_peak was {}",
            e.genomic_peak
        );
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
        // The cutoff `set_pair` settles on: the valley, at least 2 and at most the tighter guard.
        let clamped = |e: &SpectrumEstimate| {
            (e.valley as u16).clamp(2, e.guard_measured.min(e.guard_poisson))
        };

        let h = synthetic_spectrum(10.0);
        let e = SpectrumEstimate::by_valley(&h);
        assert_eq!(e.verdict, Verdict::Ok);
        assert_eq!(e.guard_poisson, poisson_guard(e.genomic_peak));
        assert_eq!(
            e.guard_measured,
            measured_guard(&h, e.valley, e.genomic_peak)
        );
        assert!(e.guard_poisson < e.guard_measured);
        assert_eq!(e.min_count, clamped(&e));

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
        let mut e = SpectrumEstimate::default();
        e.set_pair(&wide, 60, 200);
        assert_eq!(e.verdict, Verdict::Ok);
        assert_eq!(
            (e.guard_measured, e.guard_poisson),
            (measured, poisson_guard(200))
        );
        assert_eq!(e.min_count, clamped(&e));
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

    /// The hole guard is a ceiling on a resolved cutoff: it lowers the valley to the fit's hole cutoff,
    /// never raises it, never goes below 2, and leaves a refused spectrum alone. `guarded_min_count`
    /// predicts what `apply_hole_guard` settles on. (The error floor's precedence over it is checked in
    /// `filter::tests::active_error_floor_remains_a_hard_minimum`.)
    #[test]
    fn the_hole_guard_only_lowers_a_resolved_cutoff_and_never_below_two() {
        let fit = test_fit(40, 0.0);
        let holes = fit
            .hole_cutoff(MAX_GENOME_HOLES)
            .expect("the lobe reaches low counts");
        assert_eq!(fit.error_floor_cutoff(), 2, "no error floor to interfere");
        assert!(
            holes > 3,
            "the fixture needs room below its hole cutoff, got {holes}"
        );
        // A fit whose single-copy lobe strands a k-mer already at count 1.
        let shallow = test_fit(3, 0.0);
        assert_eq!(shallow.hole_cutoff(MAX_GENOME_HOLES), Some(2));

        // (fit, verdict, min_count before, min_count after)
        for (fit, verdict, before, after) in [
            (fit, Verdict::Ok, holes + 4, holes),
            (fit, Verdict::Ok, holes - 1, holes - 1),
            (fit, Verdict::Ok, holes, holes),
            (fit, Verdict::Ok, 2, 2),
            (shallow, Verdict::Ok, 9, 2),
            (fit, Verdict::PeakNotClearOfValley, holes + 4, holes + 4),
        ] {
            let mut estimate = SpectrumEstimate {
                min_count: before,
                verdict,
                ..SpectrumEstimate::default()
            };
            let predicted = estimate.guarded_min_count(&fit);
            estimate.apply_hole_guard(&fit);
            assert_eq!(estimate.min_count, after, "{verdict:?} from {before}");
            assert_eq!(predicted, after, "{verdict:?} from {before}");
        }
    }

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
}

#[cfg(test)]
pub(super) mod test_support {
    use super::filter::{
        choose_cutoff_and_filter, choose_min_count_and_floor_with_debug, fit_and_log,
    };
    use super::models::{ErrorModel, ErrorParams, GenomeModel};
    use super::parsercounter::{count_kmers, CountMap, CountedRead, KmerWalk, PassInput};
    use super::plotter::SpectrumPlotDiagnostics;
    use super::read_store::PackedSeqs;
    use super::sketch::SpectrumSketch;
    use super::*;
    use crate::utils::qual_profile::MAX_GROUPS;
    use crate::QualOpts;
    use rayon::prelude::*;
    use std::collections::HashMap;

    /// Text adapter for the existing counter fixtures, including multiple input iterators.
    pub(super) fn count_text<IntT, I>(
        records: &mut [I],
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
        count_kmers(
            PassInput::Text(records.iter_mut().flatten().map(Ok)),
            k,
            qual,
            floors,
            0,
            do_bloom,
            do_fit,
        )
        .unwrap()
    }

    pub(super) fn packed_contigs(seqs: &[Vec<u8>]) -> PackedSeqs {
        let mut packed = PackedSeqs::new(true);
        for seq in seqs {
            assert!(seq
                .iter()
                .all(|&b| crate::utils::bit_encoding::valid_base(b)));
            packed.push_contig(
                &seq.iter()
                    .map(|&b| crate::utils::bit_encoding::encode_base(b))
                    .collect::<Vec<_>>(),
            );
        }
        packed.finish();
        packed
    }

    /// Reads for the counter tests. Rotating the backbone keeps most k-mers shared across replicates
    /// while making the seam k-mers unique, so the spectrum has a singleton lobe as well as a deep one.
    /// Each read is mostly Q37 with one dip through Q11 to Q2, moved along by replicate: the dip has to
    /// be short, or no window of k consecutive bases ever clears the strict floor.
    pub(super) fn counter_test_reads() -> Vec<OwnedRecord> {
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

    /// True occurrence count per k-mer above the strict floor, from the independent reference hasher.
    pub(super) fn true_counts(
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

    pub(super) fn bloom_shards(
        reads: Vec<OwnedRecord>,
        k: usize,
        qual: &QualOpts,
        ladder: &[u8],
    ) -> HashMap<u64, u32> {
        let mut iters = [reads.into_iter()];
        let (shards, _, _) = count_text::<u64, _>(&mut iters, k, qual, Some(ladder), true, true);
        let mut got: HashMap<u64, u32> = HashMap::default();
        for shard in &shards {
            for (hc, info) in shard {
                got.insert(*hc, info.count);
            }
        }
        got
    }

    /// Canonical-minimum hashes for the sharding tests: `min(fwd, rc)` has a triangular density, which
    /// is exactly the bias `shard_of` has to mix away.
    pub(super) fn canonical_minimum_hashes(n: u64) -> impl Iterator<Item = u64> {
        (0..n).map(|i| {
            let f = i.wrapping_mul(0x2545_F491_4F6C_DD1D);
            let r = (!i).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            f.min(r)
        })
    }

    pub(super) fn poisson(k: usize, lam: f64) -> f64 {
        (-lam + k as f64 * lam.ln() - libm::lgamma(k as f64 + 1.0)).exp()
    }

    /// A synthetic spectrum at k-mer coverage `lam`, with a 1 % per-base error rate at k = 51.
    pub(super) fn synthetic_spectrum(lam: f64) -> Vec<u32> {
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

    pub(super) fn spectrum_with_valley_and_peak(
        valley: usize,
        peak: usize,
        len: usize,
    ) -> Vec<u32> {
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

    /// Fraction of a Poisson(`lam`) genome deleted by cutting below `min_count`.
    pub(super) fn genome_loss(lam: f64, min_count: u16) -> f64 {
        (0..min_count as usize).map(|c| poisson(c, lam)).sum()
    }

    /// What cutting at `m` costs, as a fraction of the k-mers above `valley` — the quantity the guard
    /// budgets, measured the same way from the same histogram.
    pub(super) fn measured_loss(histovec: &[u32], valley: usize, m: usize) -> f64 {
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

    /// A modest bimodal spectrum as (count, number of distinct k-mers) pairs: an error lobe decaying
    /// from count 1 and a genome lobe around `genomic_peak`.
    pub(super) fn bimodal(genomic_peak: usize) -> Vec<(u32, usize)> {
        bimodal_with_genome_size(genomic_peak, 3_000.0)
    }

    pub(super) fn bimodal_with_genome_size(
        genomic_peak: usize,
        genome_size: f64,
    ) -> Vec<(u32, usize)> {
        let mut out: Vec<(u32, usize)> = (1..=5).map(|c| (c as u32, 4000 / c)).collect();
        for c in (genomic_peak - 12)..=(genomic_peak + 12) {
            let d = (c as f64 - genomic_peak as f64) / 5.0;
            out.push((c as u32, (genome_size * (-0.5 * d * d).exp()) as usize));
        }
        out
    }

    pub(super) fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// A deterministic genome sequenced at `depth` with 100 bp reads at a flat quality, plus a scatter
    /// of single-base errors, so the spectrum has a singleton lobe as well as a deep one.
    pub(super) fn deep_library(depth: usize, phred: u8) -> Vec<OwnedRecord> {
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
    pub(super) fn count_and_finish(
        reads: Vec<OwnedRecord>,
        k: usize,
        qual: &QualOpts,
        ladder: &[u8],
        do_bloom: bool,
    ) -> (IndexedKmers<u64>, Vec<u32>, u16, u8, PeakSource) {
        let mut iters = [reads.into_iter()];
        let (shards, sketch, floor_has_kmers) =
            count_text::<u64, _>(&mut iters, k, qual, Some(ladder), do_bloom, true);
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

    /// Build a histogram from a `(count, distinct)` spectrum, as the main table would.
    pub(super) fn histo(spectrum: &[(u32, usize)]) -> Vec<u32> {
        let mut out = vec![0u32; MAXSIZEHISTO];
        for &(count, n) in spectrum {
            for _ in 0..n {
                add_to_histogram(&mut out, count);
            }
        }
        out
    }

    pub(super) fn test_fit(mean: usize, error_weight: f64) -> SpectrumFit {
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

    /// Deterministic ACGT, so a 2 kb genome has no repeated 20-mer to branch on.
    pub(super) fn pseudo_genome(len: usize) -> Vec<u8> {
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

    /// The min-count, and the floor it was read at. Every candidate floor is evaluated and only then
    /// compared, all against one reference fixed beforehand, so the answer cannot depend on the order the
    /// floors happen to be visited in.
    pub(super) fn choose_min_count_and_floor(
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

    pub(super) fn choose_min_count_and_floor_with_fit<F>(
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
    pub(super) fn hash_batch<IntT>(
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
            .flat_map_iter(|read| {
                let mut local = Vec::new();
                let mut groups: Vec<u8> = Vec::new();
                let w = KmerWalk {
                    k,
                    min_qual,
                    min_class: 0,
                    floors,
                    keep,
                };
                read.walk::<IntT, _>(w, &mut groups, |hc, hnc, b, g, km| {
                    local.push((hc, hnc, b, g, km))
                });
                local
            })
            .collect()
    }
}
