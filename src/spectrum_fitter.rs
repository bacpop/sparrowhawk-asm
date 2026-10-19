//! Fits a coverage model to the k-mer spectrum, to choose a count cutoff.
//!
//! Three-component coverage mixture.
//!
//! ====================================================== Got from ska-rust!! =====
//!
//! [`fit_spectrum`] is the main interface.

use argmin::core::TerminationReason::MaxItersReached;
use argmin::{
    core::{CostFunction, Error, Executor, State, TerminationReason::SolverConverged},
    solver::neldermead::NelderMead,
};
use libm::lgamma;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{fmt, sync::Arc};

// Constants

// The fit window and the single-copy mode's band

/// Counts above this multiple of the peak are not fitted: conservative threshold.
const FIT_WINDOW_PEAK_MULT: usize = 6;

/// The fitted single-copy mode is held within peak × [MU_LO, MU_LO + MU_SPAN], 0.75 to 1.33.
const MU_LO: f64 = 0.75;
/// Width of that band, as a multiple of the peak.
const MU_SPAN: f64 = 0.58;

// These are consts that allow the mode to move a bit more from the empirical peaks,
// for cases of high coverage. In those situations, the distribution can be very dispersed
// and it makes total sense to allow for more "space" for the mode to be different from the
// empirical peak, as a good fit can be recovered even under those circumstances.
/// Largest ratio allowed between the fitted single-copy mode and the empirical peak, at low coverage...
const PRIMARY_MODE_RATIO_LOW: f64 = 1.25;
/// ...and at high coverage.
const PRIMARY_MODE_RATIO_HIGH: f64 = 1.40;
/// The allowed ratio starts rising from the low one at this empirical peak...
const PRIMARY_MODE_RATIO_START_PEAK: usize = 100;
/// ...and reaches the high one at this one.
const PRIMARY_MODE_RATIO_END_PEAK: usize = 200;

// The two-copy lobe

/// Ratio we allow the second lobe (repeats lobe) to float in
const REPEAT_RATIO: (f64, f64) = (1.5, 2.0);
/// Where the ratio starts: just under a double, so the simplex has room on both sides.
const REPEAT_RATIO_START: f64 = 1.9;

// The shelf

/// The shelf never holds more than this share of single-copy k-mers, so it cannot swallow the lobe.
const SHELF_MAX_SHARE: f64 = 0.5;
/// The shelf's k-mers keep at least this fraction of their coverage on average: below it the shelf
/// sat on the error tail, at counts 0-1, in the 2026_09_28 sweep.
const SHELF_MIN_RETAINED: f64 = 0.3;
/// Upper bound on the shelf's a + b: above it the shelf is a spike.
const SHELF_MAX_CONCENTRATION: f64 = 20.0;
/// Starting (retained fraction, concentration): a broad bump, and one ramping up to the lobe. Both sit
/// inside the bounds, so the simplex can move either way.
const SHELF_SHAPES: [(f64, f64); 2] = [(0.5, 3.0), (0.75, 4.0)];

// The search

/// The error and genome models fitted from every start; the lowest BIC among the valid fits wins.
const CONFIGURATIONS: [(ErrorModel, GenomeModel); 4] = [
    (ErrorModel::SingletonPareto, GenomeModel::NegativeBinomial),
    (
        ErrorModel::FreeSingletonWeibull,
        GenomeModel::NegativeBinomial,
    ),
    (
        ErrorModel::SingletonPareto,
        GenomeModel::ShelfNegativeBinomial,
    ),
    (
        ErrorModel::FreeSingletonWeibull,
        GenomeModel::ShelfNegativeBinomial,
    ),
];
/// Dispersions to restart the fit from, alongside the valley estimator's own guess.
/// A single start is NOT enough.
const DISPERSION_SEEDS: [f64; 3] = [2.0, 8.0, 15.0];
/// Offset applied to one coordinate at a time to build the initial simplex.
const SIMPLEX_STEP: f64 = 0.5;
/// Argmin defaults to machine epsilon, which is needlessly strict for likelihoods around 1e7 and
/// makes an otherwise stationary simplex run to [`MAX_ITERS`].
const SD_TOLERANCE: f64 = 1e-4;
/// Maximum number of fit iterations (per fit)
const MAX_ITERS: u64 = 1_000;
/// Three more dimensions need more simplex steps than the NB fit's cap.
const SHELF_MAX_ITERS: u64 = 2_000;

// Validation

/// Allow modest fit/sketch error, but not component totals larger than the whole spectrum.
const COMPONENT_MASS_MAX_RATIO: f64 = 1.10;
/// A log window mass is a probability and therefore cannot exceed one, apart from round-off.
const WINDOW_LOG_MASS_TOLERANCE: f64 = 1e-12;
/// Require the NB mode encoded by mean and dispersion to retain half-count precision.
const NB_MODE_ROUNDTRIP_TOLERANCE: f64 = 0.5;

// Maths helpers

/// Log-sum-exp of two terms, shifted by the larger so the exponentials stay in range.
#[inline]
fn lse2(a: f64, b: f64) -> f64 {
    let maximum = a.max(b);
    if !maximum.is_finite() {
        return maximum;
    }
    maximum + ((a - maximum).exp() + (b - maximum).exp()).ln()
}

/// Log-sum-exp of three terms, shifted by the largest so the exponentials stay in range.
#[inline]
fn lse3(a: f64, b: f64, c: f64) -> f64 {
    let m = a.max(b).max(c);
    if !m.is_finite() {
        return m;
    }
    m + ((a - m).exp() + (b - m).exp() + (c - m).exp()).ln()
}

/// Log of the three softmax weights, computed in log space so no exponential can overflow.
#[inline]
fn ln_softmax3(a: f64, b: f64, c: f64) -> (f64, f64, f64) {
    let norm = lse3(a, b, c);
    (a - norm, b - norm, c - norm)
}

/// ln(e^high − e^low) for `high >= low`, computed as high + ln(−expm1(low − high)), so nothing
/// underflows. Minus infinity when `high` is minus infinity, and NaN when `low > high`.
#[inline]
fn ln_sub_exp(high: f64, low: f64) -> f64 {
    if high == f64::NEG_INFINITY {
        return high;
    }
    if low > high || low.is_nan() {
        return f64::NAN;
    }
    high + (-(low - high).exp_m1()).ln()
}

/// The logistic function, evaluated on the side that cannot overflow.
#[inline]
fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let scaled = value.exp();
        scaled / (1.0 + scaled)
    }
}

/// ln Σ p(count) over counts 1 to `top`, from ln p(1) and the log ratio of consecutive terms.
fn log_recurrence_mass<F>(mut log_probability: f64, top: usize, log_ratio: F) -> f64
where
    F: Fn(usize) -> f64,
{
    let mut total = f64::NEG_INFINITY;
    for count in 1..=top {
        total = lse2(total, log_probability);
        log_probability += log_ratio(count);
    }
    total
}

// Error model

/// The error component's family: both give count 1 a free probability, and differ in the tail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ErrorModel {
    /// A free count-1 probability, then the Weibull-like tail.
    FreeSingletonWeibull,
    /// A free count-1 probability, then a Pareto tail.
    SingletonPareto,
}

impl fmt::Display for ErrorModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::FreeSingletonWeibull => "free_singleton_weibull",
            Self::SingletonPareto => "singleton_pareto",
        })
    }
}

impl ErrorModel {
    /// Coordinates of θ before the genome model's own: the two free weights, the mode, the dispersion,
    /// and this model's error parameters.
    const fn parameter_count(self) -> usize {
        match self {
            Self::FreeSingletonWeibull => 7,
            Self::SingletonPareto => 6,
        }
    }
}

/// The error component's parameters. Count 1 has probability p₁ (`singleton_probability`); from
/// count 2 on, the tail has survival S(x) = exp(−λ((x − 1)^β − 1)/β), with λ the `tail_exponent` and
/// β the `weibull_shape`. It is Weibull-like in x − 1, and tends to the Pareto (x − 1)^−λ as β → 0,
/// which `SingletonPareto` fixes. `c = scale * beta` keeps the conditional Weibull tail
/// well-conditioned as beta approaches zero.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ErrorParams {
    /// p₁, the probability of count 1.
    pub(crate) singleton_probability: f64,
    /// λ, the tail's scale: its exponent in the Pareto limit.
    pub(crate) tail_exponent: f64,
    /// β, or `None` for the Pareto tail.
    pub(crate) weibull_shape: Option<f64>,
}

impl ErrorParams {
    /// ln S(count): the log probability that an error k-mer of count 2 or more reaches `count`.
    #[inline]
    fn log_survival_from_two(self, count: f64) -> f64 {
        let log_offset = (count - 1.0).ln();
        let factor = self.weibull_shape.map_or(1.0, |beta| {
            let value = beta * log_offset;
            if value == 0.0 {
                1.0
            } else {
                value.exp_m1() / value
            }
        });
        -self.tail_exponent * log_offset * factor
    }

    /// ln p(count): p₁ at count 1, and (1 − p₁)(S(count) − S(count + 1)) beyond.
    #[inline]
    fn log_probability(self, count: f64) -> f64 {
        if count == 1.0 {
            return self.singleton_probability.ln();
        }
        (1.0 - self.singleton_probability).ln()
            + ln_sub_exp(
                self.log_survival_from_two(count),
                self.log_survival_from_two(count + 1.0),
            )
    }

    /// Log of the mass on the fit window, [1, top]: ln(1 − (1 − p₁) S(top + 1)).
    fn log_window_mass(self, top: usize) -> f64 {
        let log_beyond =
            (1.0 - self.singleton_probability).ln() + self.log_survival_from_two(top as f64 + 1.0);
        (-log_beyond.exp_m1()).ln()
    }
}

// Genome model

/// The genome lobes' family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GenomeModel {
    /// Two NB lobes, one main, one secondary to fit potential repeats.
    NegativeBinomial,
    /// Negative-binomial lobes, plus a beta-binomial shelf under the single-copy one to accommodate strong left genomic lobe tails.
    ShelfNegativeBinomial,
}

impl fmt::Display for GenomeModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NegativeBinomial => "negative_binomial",
            Self::ShelfNegativeBinomial => "betabinomial_shelf+negative_binomial",
        })
    }
}

impl GenomeModel {
    /// Parameters beyond the lobes' mean and dispersion: the two-copy lobe's ratio, and for a shelf its
    /// share, a and b.
    const fn extra_parameters(self) -> usize {
        match self {
            Self::ShelfNegativeBinomial => 4,
            Self::NegativeBinomial => 1,
        }
    }
}

/// Natural log of the Poisson density.
#[inline]
fn ln_dpois(x: f64, lambda: f64) -> f64 {
    x * lambda.ln() - lgamma(x + 1.0) - lambda
}

/// Natural log of the negative-binomial density, by mean and dispersion (`variance = dispersion *
/// mean`), so `r = mean / (dispersion - 1)` and `p = 1 / dispersion`. Tends to the Poisson as the
/// dispersion approaches 1, which is where the genome lobe of a clean simulated library sits.
#[inline]
fn ln_dnbinom(x: f64, mean: f64, dispersion: f64) -> f64 {
    if dispersion <= 1.0 + 1e-9 {
        return ln_dpois(x, mean);
    }
    let r = mean / (dispersion - 1.0);
    lgamma(x + r) - lgamma(x + 1.0) - lgamma(r)
        + r * (-dispersion.ln())
        + x * (1.0 - 1.0 / dispersion).ln()
}

/// The negative binomial's mode, as a count: floor(mean − (dispersion − 1)), or 0.
fn discrete_nb_mode(mean: f64, dispersion: f64) -> usize {
    let mode = mean - (dispersion - 1.0);
    if mode.is_finite() && mode > 0.0 {
        mode.floor().min(usize::MAX as f64) as usize
    } else {
        0
    }
}

/// Log of the negative binomial's mass on [1, top].
fn log_nb_mass(mean: f64, dispersion: f64, top: usize) -> f64 {
    if dispersion <= 1.0 + 1e-9 {
        return log_recurrence_mass(ln_dpois(1.0, mean), top, |count| {
            mean.ln() - ((count + 1) as f64).ln()
        });
    }
    let r = mean / (dispersion - 1.0);
    let log_q = (1.0 - 1.0 / dispersion).ln();
    log_recurrence_mass(ln_dnbinom(1.0, mean, dispersion), top, |count| {
        ((count as f64) + r).ln() - ((count + 1) as f64).ln() + log_q
    })
}

// The shelf, inside the single-copy lobe

/// Struct to hold a beta-binomial distribution to model cases where some k-mers lost part of their reads or weren't
/// as sequenced as most of the total kmers. This could be the case of e.g. very GC-enriched genome areas, so it has
/// a biological reason to exist. Also we could have systematic errors :/
/// The idea is that this could explain a strong left shoulder of the main genomic peak, and/or a strong valley that would not
/// be explained by the error or genomic peaks themselves. Additionally, it could explain even an extra minipeak at the left
/// of the clearly main one. It is really versatile, but I had to set some heuristics to constraint it so that it doesn't get
/// an absurd shape, or size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Shelf {
    /// Share of the single-copy component the shelf holds.
    pub(crate) share: f64,
    /// The beta-binomial's a...
    pub(crate) alpha: f64,
    /// ...and b.
    pub(crate) beta: f64,
}

impl Shelf {
    /// From the fit's bounded coordinates: `a = m * kappa`, `b = (1 - m) * kappa`.
    fn from_retained(share: f64, retained: f64, concentration: f64) -> Self {
        Self {
            share,
            alpha: retained * concentration,
            beta: (1.0 - retained) * concentration,
        }
    }

    /// Mean fraction of coverage the shelf's k-mers keep, `a / (a + b)`.
    pub(crate) fn retained(self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// How narrow the shelf is, `a + b`: the variance of the retained fraction is `m(1 - m)/(kappa + 1)`.
    pub(crate) fn concentration(self) -> f64 {
        self.alpha + self.beta
    }
}

/// Bounds on a + b for a retained fraction `m`. The lower one is a = m * (a + b) >= 1: below it the
/// density diverges at count 0 and its mass at counts 1-2 pulled the hole guard down (S. aureus, k = 71).
fn shelf_concentration_bounds(retained: f64) -> (f64, f64) {
    (1.0 / retained, SHELF_MAX_CONCENTRATION)
}

/// The shelf's trials: the single-copy mean, rounded, as the beta-binomial needs a whole number.
fn shelf_trials(mean: f64) -> f64 {
    mean.round().max(1.0)
}

/// A share inside (0, [`SHELF_MAX_SHARE`]), and finite shapes with a >= 1 within the retained-fraction
/// and concentration bounds.
fn shelf_is_valid(shelf: Shelf) -> bool {
    // Rounding in `a / (a + b)` may land a hair outside a bound the mapping itself respects.
    let slack = 1e-9;
    shelf.share > 0.0
        && shelf.share < SHELF_MAX_SHARE
        && shelf.alpha.is_finite()
        && shelf.alpha >= 1.0 - slack
        && shelf.beta.is_finite()
        && shelf.beta > 0.0
        && shelf.retained() >= SHELF_MIN_RETAINED - slack
        && shelf.concentration() <= SHELF_MAX_CONCENTRATION + slack
}

/// Natural log of the beta-binomial density over `trials` trials; minus infinity above them.
fn ln_dbetabinom(x: f64, trials: f64, alpha: f64, beta: f64) -> f64 {
    if x > trials {
        return f64::NEG_INFINITY;
    }
    lgamma(trials + 1.0) - lgamma(x + 1.0) - lgamma(trials - x + 1.0)
        + lgamma(x + alpha)
        + lgamma(trials - x + beta)
        - lgamma(trials + alpha + beta)
        - (lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta))
}

/// The shelf's log density at every count `0..=min(trials, top)`, from the ratio of consecutive terms,
/// so the fit's hot loop pays four logs per count instead of eight log-gammas. Only counts inside the
/// fit window are ever read, and mid-search the mean (so the trials) is unbounded.
fn shelf_log_table(mean: f64, shelf: Shelf, top: usize) -> Vec<f64> {
    let trials = shelf_trials(mean);
    let last = trials.min(top as f64) as usize;
    let (alpha, beta) = (shelf.alpha, shelf.beta);
    let mut ln =
        lgamma(trials + beta) + lgamma(alpha + beta) - lgamma(trials + alpha + beta) - lgamma(beta);
    let mut shelf_table = Vec::with_capacity(last + 1);
    shelf_table.push(ln);
    for x in 0..last {
        let x = x as f64;
        ln +=
            (trials - x).ln() + (x + alpha).ln() - (x + 1.0).ln() - (trials - x - 1.0 + beta).ln();
        shelf_table.push(ln);
    }
    shelf_table
}

/// Log density of the single-copy component: its lobe, plus the shelf when the model has one.
fn ln_single_copy(count: f64, mean: f64, dispersion: f64, shelf: Option<Shelf>) -> f64 {
    let lobe = ln_dnbinom(count, mean, dispersion);
    shelf.map_or(lobe, |s| {
        lse2(
            (1.0 - s.share).ln() + lobe,
            s.share.ln() + ln_dbetabinom(count, shelf_trials(mean), s.alpha, s.beta),
        )
    })
}

/// The single-copy log density from its lobe's, adding the shelf read from its table.
#[inline]
fn mix_single_copy(lobe: f64, count: f64, shelf: Option<(Shelf, &[f64])>) -> f64 {
    shelf.map_or(lobe, |(s, shelf_table)| {
        let at = shelf_table
            .get(count as usize)
            .copied()
            .unwrap_or(f64::NEG_INFINITY);
        lse2((1.0 - s.share).ln() + lobe, s.share.ln() + at)
    })
}

/// Log of the single-copy component's mass inside `[1, top]`, the shelf read from its table.
fn single_copy_log_mass(
    mean: f64,
    dispersion: f64,
    shelf: Option<(Shelf, &[f64])>,
    top: usize,
) -> f64 {
    let lobe = log_nb_mass(mean, dispersion, top);
    shelf.map_or(lobe, |(s, shelf_table)| {
        let shelf_mass = shelf_table
            .iter()
            .take(top + 1)
            .skip(1)
            .fold(f64::NEG_INFINITY, |acc, &ln| lse2(acc, ln));
        lse2((1.0 - s.share).ln() + lobe, s.share.ln() + shelf_mass)
    })
}

// Fit result

/// Why a converged fit is not used: the first validation check it fails.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FitRejection {
    OptimisationFailed,
    NonFiniteLikelihood,
    InvalidComponents,
    InvalidWindowMass,
    ComponentMassExceeded,
    UnstableModeParameterisation,
    GenomeLobeOutsideFitWindow,
    ErrorModePastValley,
    PrimaryModeOutsideBand,
    RepeatModeTooEarly,
    PositiveLogLikelihood,
}

impl fmt::Display for FitRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::OptimisationFailed => "optimisation_failed",
            Self::NonFiniteLikelihood => "non_finite_likelihood",
            Self::InvalidComponents => "invalid_components",
            Self::InvalidWindowMass => "invalid_window_mass",
            Self::ComponentMassExceeded => "component_mass_exceeded",
            Self::UnstableModeParameterisation => "unstable_mode_parameterisation",
            Self::GenomeLobeOutsideFitWindow => "genome_lobe_outside_fit_window",
            Self::ErrorModePastValley => "error_mode_past_valley",
            Self::PrimaryModeOutsideBand => "primary_mode_outside_band",
            Self::RepeatModeTooEarly => "repeat_mode_too_early",
            Self::PositiveLogLikelihood => "positive_loglikelihood",
        })
    }
}

/// A converged fit, as the crate reads it: the model's parameters, and the totals, likelihood and
/// masses derived from them.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SpectrumFit {
    /// The error component's family.
    pub(crate) error_model: ErrorModel,
    /// The genome lobes' family.
    pub(crate) genome_model: GenomeModel,
    /// Mixture weight of the error component, inside the fit window.
    pub(crate) w_error: f64,
    /// Mixture weight of the single-copy component, inside the fit window.
    pub(crate) w_single: f64,
    /// Mixture weight of the two-copy component, inside the fit window.
    pub(crate) w_repeat: f64,
    /// Mean of the single-copy lobe.
    pub(crate) mean: f64,
    /// Variance over mean, shared by both lobes.
    pub(crate) dispersion: f64,
    /// The continuous mode used to derive the NB mean, retained to detect cancellation.
    target_mode: f64,
    /// The error component's parameters.
    pub(crate) error_params: ErrorParams,
    /// The shelf under the single-copy lobe, when the genome model has one.
    pub(crate) shelf: Option<Shelf>,
    /// The two-copy lobe's mean as a multiple of the single-copy mean.
    pub(crate) repeat_ratio: f64,
    /// Distinct k-mers in the single-copy component, extrapolated over every count, at library scale.
    pub(crate) genome_kmers: f64,
    /// Distinct k-mers in the error component, likewise.
    pub(crate) error_kmers: f64,
    /// Distinct k-mers in the two-copy component, likewise.
    pub(crate) repeat_kmers: f64,
    /// Log-likelihood of the fit window, at sample scale.
    pub(crate) log_likelihood: f64,
    /// Bayesian information criterion: the lowest valid one is selected.
    pub(crate) bic: f64,
    /// The window's deviance, 2 Σ obs·ln(obs/expected), at sample scale.
    pub(crate) deviance: f64,
    /// Last count fitted.
    pub(crate) fit_window_end: usize,
    /// Simplex iterations of the converged start.
    pub(crate) best_iterations: u64,
    /// Each component's log mass inside the fit window, [1, `fit_window_end`]: its density is divided
    /// by it, so the mixture is a distribution over the counts actually fitted.
    pub(crate) component_log_window_masses: [f64; 3],
    /// Each component's log total of distinct k-mers, extrapolated over every count, at library scale.
    component_log_totals: [f64; 3],
    /// Distinct k-mers inside the fit window, at library scale.
    observed_total: f64,
    /// Distinct k-mers in the whole spectrum, at library scale.
    full_distinct_total: f64,
}

impl SpectrumFit {
    /// The components' extrapolated totals over the spectrum's distinct k-mers; validation tests the
    /// same in log space.
    pub(crate) fn component_total_ratio(&self) -> f64 {
        let log_total = lse3(
            self.component_log_totals[0],
            self.component_log_totals[1],
            self.component_log_totals[2],
        );
        (log_total - self.full_distinct_total.ln()).exp()
    }

    /// How far the mode recovered from the mean, mean − (dispersion − 1), lands from the mode the fit
    /// searched over. At a huge dispersion the subtraction cancels, and validation rejects the fit.
    pub(crate) fn mode_roundtrip_error(&self) -> f64 {
        (self.mean - (self.dispersion - 1.0) - self.target_mode).abs()
    }

    /// The single-copy lobe's mode.
    pub(crate) fn primary_mode(&self) -> usize {
        discrete_nb_mode(self.mean, self.dispersion)
    }

    /// The two-copy lobe's mode.
    pub(crate) fn repeat_mode(&self) -> usize {
        discrete_nb_mode(self.repeat_ratio * self.mean, self.dispersion)
    }

    /// Count where the error component is highest in the window; validation needs it under the valley.
    pub(crate) fn error_mode(&self) -> usize {
        (1..=self.fit_window_end)
            .max_by(|&a, &b| {
                self.error_params
                    .log_probability(a as f64)
                    .total_cmp(&self.error_params.log_probability(b as f64))
            })
            .unwrap_or(1)
    }

    /// Expected in-window component heights at one observed abundance.
    pub(crate) fn component_heights(&self, count: usize) -> [f64; 3] {
        let count = count as f64;
        let logs = [
            self.error_params.log_probability(count),
            ln_single_copy(count, self.mean, self.dispersion, self.shelf),
            ln_dnbinom(count, self.repeat_ratio * self.mean, self.dispersion),
        ];
        [
            self.observed_total
                * self.w_error
                * (logs[0] - self.component_log_window_masses[0]).exp(),
            self.observed_total
                * self.w_single
                * (logs[1] - self.component_log_window_masses[1]).exp(),
            self.observed_total
                * self.w_repeat
                * (logs[2] - self.component_log_window_masses[2]).exp(),
        ]
    }

    /// Expected in-window height of the shelf alone, the part of the single-copy component it holds.
    /// Read only by the native plots.
    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn shelf_height(&self, count: usize) -> Option<f64> {
        self.shelf.map(|s| {
            let ln = s.share.ln()
                + ln_dbetabinom(count as f64, shelf_trials(self.mean), s.alpha, s.beta);
            self.observed_total * self.w_single * (ln - self.component_log_window_masses[1]).exp()
        })
    }

    /// Very important heuristic: this helps us reduce the valley avoiding losing kmers from the genome parts whenever
    /// we see in the fit that the genomic part extends significantly to the left of the empirical valley. This recovers,
    /// in most situations, a good portion of contiguity.
    ///
    /// It is the largest cutoff whose expected removed single-copy k-mers (those below it, the unobserved
    /// count 0 included) stay within `budget`, and at least 2; `None` when every cutoff up to the mean
    /// stays within it.
    pub(crate) fn hole_cutoff(&self, budget: f64) -> Option<u16> {
        if self.genome_kmers <= 0.0
            || !self.mean.is_finite()
            || self.mean < 2.0
            || self.fit_window_end < 2
        {
            return None;
        }
        let allowed = budget / self.genome_kmers;
        let ceiling = (self.mean.round() as usize).min(self.fit_window_end);
        let single = |count: f64| {
            ln_single_copy(count, self.mean, self.dispersion, self.shelf).exp() // we un-log it
        };
        let mut cdf = single(0.0);
        let mut cutoff = 1usize;
        while cutoff < ceiling {
            let next = cdf + single(cutoff as f64);
            if next > allowed {
                return Some((cutoff.min(u16::MAX as usize) as u16).max(2));
            }
            cdf = next;
            cutoff += 1;
        }
        None
    }

    /// Where error and genomic parts cross. This would be, ideally, equivalent to the empiric valley.
    /// It is the first count where the single-copy component outgrows the error one; only the logs and
    /// the plots read it.
    pub(crate) fn crossover(&self) -> Option<u16> {
        if !self.mean.is_finite() || self.mean < 2.0 || self.fit_window_end < 2 {
            return None;
        }
        let ceiling = (self.mean.round().max(2.0) as usize).min(self.fit_window_end);
        for count in 1..ceiling {
            let heights = self.component_heights(count);
            if heights[1] > heights[0] {
                return Some((count.min(u16::MAX as usize) as u16).max(2));
            }
        }
        None
    }

    /// One past the last count, within the fit window, where the error component is at least as high
    /// as the single-copy lobe at its own mode; at least 2. Below it, errors alone fill a bin to the
    /// genome peak's height, so the cutoff never goes under it (the logs' "100% error floor").
    pub(crate) fn error_floor_cutoff(&self) -> u16 {
        let reference = self.component_heights(self.primary_mode())[1];
        let mut last_bad = 0usize;
        for count in 1..=self.fit_window_end {
            if self.component_heights(count)[0] >= reference {
                last_bad = count;
            }
        }
        last_bad.saturating_add(1).min(u16::MAX as usize).max(2) as u16
    }
}

/// One configuration's outcome: its best valid fit, or its best rejected one for the logs.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FitAttempt {
    /// The configuration's error model...
    pub(crate) error_model: ErrorModel,
    /// ...and genome model.
    pub(crate) genome_model: GenomeModel,
    /// The best valid fit, or else the lowest-cost rejected one; `None` when no start converged.
    pub(crate) candidate: Option<SpectrumFit>,
    /// Why `candidate` was rejected, or `None` when it is valid.
    pub(crate) rejection: Option<FitRejection>,
    /// Simplex iterations over every start.
    pub(crate) total_iterations: u64,
    /// Starts that hit their iteration cap.
    pub(crate) capped_starts: u32,
}

/// The whole search: the selected fit, if any is valid, and every configuration's attempt.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FitSearchResult {
    /// The valid fit with the lowest BIC.
    pub(crate) selected: Option<SpectrumFit>,
    /// One attempt per entry of [`CONFIGURATIONS`], in its order.
    pub(crate) attempts: [FitAttempt; 4],
}

// Optimisation problem

/// One simplex point decoded into the model's parameters.
#[derive(Clone, Copy)]
struct Params {
    /// Log mixture weights.
    ln_error: f64,
    ln_single: f64,
    ln_repeat: f64,
    /// The single-copy lobe's mode, the coordinate the simplex moves.
    target_mode: f64,
    /// The single-copy lobe's mean, from the mode and the dispersion.
    mean: f64,
    dispersion: f64,
    error_params: ErrorParams,
    shelf: Option<Shelf>,
    repeat_ratio: f64,
}

/// The problem argmin minimises: one configuration over the fit window.
#[derive(Clone)]
struct MixtureFit {
    /// (count, distinct k-mers) of every non-empty bin in the window, at sample scale.
    observed: Arc<[(f64, f64)]>,
    /// Their total.
    observed_total: f64,
    /// Distinct k-mers in the whole spectrum, at library scale.
    full_distinct_total: f64,
    /// The empirical single-copy peak the mode is held around.
    peak: f64,
    /// Last count fitted.
    fit_window_end: usize,
    error_model: ErrorModel,
    genome_model: GenomeModel,
    /// Library k-mers each observed one stands for: 1/fraction for a sketch, 1 for an exact table.
    sample_scale: f64,
}

impl MixtureFit {
    /// Decode θ: softmax weights with the error's logit fixed at 0, and sigmoid or exponential maps
    /// into each parameter's range.
    fn params(&self, theta: &[f64]) -> Params {
        let (ln_error, ln_single, ln_repeat) = ln_softmax3(0.0, theta[0], theta[1]);
        let target_mode = self.peak * (MU_LO + MU_SPAN / (1.0 + (-theta[2]).exp()));
        let dispersion = 1.0 + theta[3].exp();
        let error_params = ErrorParams {
            tail_exponent: theta[4].exp(),
            singleton_probability: sigmoid(theta[5]),
            weibull_shape: match self.error_model {
                ErrorModel::FreeSingletonWeibull => Some(theta[6].exp()),
                ErrorModel::SingletonPareto => None,
            },
        };
        let base = self.error_model.parameter_count();
        let (ratio_low, ratio_high) = REPEAT_RATIO;
        let repeat_ratio = ratio_low + (ratio_high - ratio_low) * sigmoid(theta[base]);
        let shelf_at = base + 1;
        let shelf = (self.genome_model == GenomeModel::ShelfNegativeBinomial).then(|| {
            let retained =
                SHELF_MIN_RETAINED + (1.0 - SHELF_MIN_RETAINED) * sigmoid(theta[shelf_at + 1]);
            let (low, high) = shelf_concentration_bounds(retained);
            Shelf::from_retained(
                SHELF_MAX_SHARE * sigmoid(theta[shelf_at]),
                retained,
                low + (high - low) * sigmoid(theta[shelf_at + 2]),
            )
        });
        Params {
            ln_error,
            ln_single,
            ln_repeat,
            target_mode,
            mean: target_mode + dispersion - 1.0,
            dispersion,
            error_params,
            shelf,
            repeat_ratio,
        }
    }

    /// Each component's log mass inside the window, [1, `fit_window_end`].
    fn log_window_masses(&self, params: Params, shelf_table: Option<&[f64]>) -> [f64; 3] {
        [
            params.error_params.log_window_mass(self.fit_window_end),
            single_copy_log_mass(
                params.mean,
                params.dispersion,
                params.shelf.zip(shelf_table),
                self.fit_window_end,
            ),
            log_nb_mass(
                params.repeat_ratio * params.mean,
                params.dispersion,
                self.fit_window_end,
            ),
        ]
    }

    /// A converged θ as the crate reads it.
    fn unpack(&self, theta: &[f64], cost: f64, iterations: u64) -> SpectrumFit {
        let params = self.params(theta);
        let shelf_table = params
            .shelf
            .map(|s| shelf_log_table(params.mean, s, self.fit_window_end));
        let window_masses = self.log_window_masses(params, shelf_table.as_deref());
        let w_error = params.ln_error.exp();
        let w_single = params.ln_single.exp();
        let w_repeat = params.ln_repeat.exp();
        // Totals and heights at library scale, for the hole guard's absolute budget and the plots;
        // BIC's penalty counts the sample that was actually observed.
        let library_total = self.observed_total * self.sample_scale;
        let component_log_totals = [
            params.ln_error + library_total.ln() - window_masses[0],
            params.ln_single + library_total.ln() - window_masses[1],
            params.ln_repeat + library_total.ln() - window_masses[2],
        ];
        let mut fit = SpectrumFit {
            error_model: self.error_model,
            genome_model: self.genome_model,
            w_error,
            w_single,
            w_repeat,
            mean: params.mean,
            dispersion: params.dispersion,
            target_mode: params.target_mode,
            error_params: params.error_params,
            shelf: params.shelf,
            repeat_ratio: params.repeat_ratio,
            genome_kmers: component_log_totals[1].exp(),
            error_kmers: component_log_totals[0].exp(),
            repeat_kmers: component_log_totals[2].exp(),
            log_likelihood: -cost,
            bic: 2.0 * cost
                + (self.error_model.parameter_count() + self.genome_model.extra_parameters())
                    as f64
                    * self.observed_total.ln(),
            deviance: 0.0,
            fit_window_end: self.fit_window_end,
            best_iterations: iterations,
            component_log_window_masses: window_masses,
            component_log_totals,
            observed_total: library_total,
            full_distinct_total: self.full_distinct_total,
        };
        fit.deviance = self.conditional_deviance(&fit);
        fit
    }

    /// 2 Σ obs·ln(obs/expected) over the window, in the sample's units.
    fn conditional_deviance(&self, fit: &SpectrumFit) -> f64 {
        self.observed
            .iter()
            .map(|&(count, observed)| {
                // Heights are at library scale; the deviance compares in the sample's own units.
                let expected = fit
                    .component_heights(count as usize)
                    .into_iter()
                    .sum::<f64>()
                    / self.sample_scale;
                if expected > 0.0 {
                    2.0 * observed * (observed / expected).ln()
                } else {
                    f64::INFINITY
                }
            })
            .sum()
    }
}

/// This implements the CostFunction trait for our mixture of distributions! This comes from the argmin crate
impl CostFunction for MixtureFit {
    type Param = Vec<f64>;
    type Output = f64;

    /// The negative log-likelihood of the window, or infinity for parameters the model cannot use.
    fn cost(&self, theta: &Self::Param) -> Result<Self::Output, Error> {
        let params = self.params(theta);

        // If the parameters don't make any sense, the cost is infinite
        if !(params.mean.is_finite()
            && params.dispersion.is_finite()
            && params.target_mode.is_finite()
            && params.error_params.tail_exponent.is_finite()
            && params.error_params.singleton_probability.is_finite()
            && params.error_params.weibull_shape.is_none_or(f64::is_finite))
            || params.mean <= 0.0
            || params.error_params.tail_exponent <= 0.0
            || !(0.0..1.0).contains(&params.error_params.singleton_probability)
            || params.shelf.is_some_and(|s| !shelf_is_valid(s))
        {
            return Ok(f64::INFINITY);
        }
        // Also for the NB mode
        if (params.mean - (params.dispersion - 1.0) - params.target_mode).abs()
            > NB_MODE_ROUNDTRIP_TOLERANCE
        {
            return Ok(f64::INFINITY);
        }

        // Built once per evaluation, then read for the window mass and at every observed count.
        let shelf_table = params
            .shelf
            .map(|s| shelf_log_table(params.mean, s, self.fit_window_end));

        let window_masses = self.log_window_masses(params, shelf_table.as_deref());
        if window_masses
            .iter()
            .any(|value| !value.is_finite() || *value > WINDOW_LOG_MASS_TOLERANCE)
        {
            return Ok(f64::INFINITY);
        }

        let shelf = params.shelf.zip(shelf_table.as_deref());

        let loglikelihood = self
            .observed
            .iter()
            .map(|&(count, weight)| {
                // Error lobe
                let error =
                    params.ln_error + params.error_params.log_probability(count) - window_masses[0];

                // Genomic lobe
                let lobe = ln_dnbinom(count, params.mean, params.dispersion);
                // Genomic lobe + shelf
                let single =
                    params.ln_single + mix_single_copy(lobe, count, shelf) - window_masses[1];

                // Second genomic lobe (repeat lobe)
                let repeat = params.ln_repeat
                    + ln_dnbinom(count, params.repeat_ratio * params.mean, params.dispersion)
                    - window_masses[2];
                weight * lse3(error, single, repeat)
            })
            .sum::<f64>();

        Ok(if loglikelihood.is_finite() && loglikelihood <= 0.0 {
            -loglikelihood
        } else {
            f64::INFINITY
        })
    }
}

// Search

/// Main function for doing the fits, after the empirical values have been extracted
/// `histovec` is at library scale; `sample_scale` is how many library k-mers each observed one stands
/// for. The likelihood and BIC count the sample itself, so a sketch is not mistaken for millions of
/// independent k-mers; totals and heights are reported back at library scale.
pub(crate) fn fit_spectrum(
    histovec: &[u32],
    valley: usize,
    peak: usize,
    dispersion_hint: f64,
    sample_scale: f64,
) -> Result<FitSearchResult, Error> {
    if peak < 2 || valley < 2 || histovec.len() < 2 {
        return Err(Error::msg("no usable genome peak or valley to fit around"));
    }

    let sample_scale = if sample_scale.is_finite() && sample_scale >= 1.0 {
        sample_scale
    } else {
        1.0
    };

    // A rescaled sketch bin is floor(sampled * scale), so dividing and rounding recovers the sample.
    let sample: Vec<u32> = histovec
        .iter()
        .map(|&n| (f64::from(n) / sample_scale).round() as u32)
        .collect();

    // `histovec` is already at library scale and spans the full count range. Keep this separate from
    // `observed_total`, which is restricted to the fit window and sample scale.
    let full_distinct_total = histovec.iter().map(|&n| f64::from(n)).sum::<f64>();
    if !full_distinct_total.is_finite() || full_distinct_total <= 0.0 {
        return Err(Error::msg("no distinct k-mers in the full spectrum"));
    }

    let top = fit_window_end(sample.len(), peak);
    let observed: Arc<[(f64, f64)]> = sample[..top]
        .iter()
        .enumerate()
        .filter(|(_, count)| **count > 0)
        .map(|(index, &count)| ((index + 1) as f64, f64::from(count)))
        .collect::<Vec<_>>()
        .into();
    let observed_total = observed.iter().map(|&(_, count)| count).sum::<f64>();
    if observed_total <= 0.0 {
        return Err(Error::msg("no counts inside the fit window"));
    }
    let hint = if dispersion_hint.is_finite() && dispersion_hint > 1.0 {
        dispersion_hint
    } else {
        2.0
    };

    let shelf_share = empirical_shelf_share(&sample, valley, peak);

    // We set the configurations up
    let problems: Vec<MixtureFit> = CONFIGURATIONS
        .iter()
        .map(|&(error_model, genome_model)| MixtureFit {
            observed: Arc::clone(&observed),
            observed_total,
            full_distinct_total,
            peak: peak as f64,
            fit_window_end: top,
            error_model,
            genome_model,
            sample_scale,
        })
        .collect();

    // All different starts/tries that we'll need
    let jobs: Vec<(usize, Vec<f64>)> = problems
        .iter()
        .enumerate()
        .flat_map(|(index, problem)| {
            fit_starts(problem, hint, shelf_share)
                .into_iter()
                .map(move |start| (index, start))
        })
        .collect();

    // Here WE RUN THE FITS!
    let outcomes: Vec<(usize, RunOutcome)> = jobs
        .into_par_iter()
        .filter_map(|(index, start)| {
            let problem = &problems[index];
            // The shelf's three extra dimensions need more simplex steps than the NB fit.
            let max_iters = if problem.genome_model == GenomeModel::ShelfNegativeBinomial {
                SHELF_MAX_ITERS
            } else {
                MAX_ITERS
            };
            run_one(problem.clone(), start, max_iters)
                .ok()
                .map(|outcome| (index, outcome))
        })
        .collect();

    // We collect nicely the outcomes per distribution configuration
    let mut per_problem: Vec<Vec<RunOutcome>> = problems.iter().map(|_| Vec::new()).collect();
    for (index, outcome) in outcomes {
        per_problem[index].push(outcome);
    }

    let attempts: Vec<FitAttempt> = problems
        .iter()
        .zip(per_problem)
        .map(|(problem, outcomes)| attempt_from(problem, outcomes, valley, peak))
        .collect();

    let attempts: [FitAttempt; 4] = attempts.try_into().expect("one attempt per configuration");
    let selected = select_fit(&attempts);
    Ok(FitSearchResult { selected, attempts })
}

/// Last count fitted: [`FIT_WINDOW_PEAK_MULT`] times the peak, within the histogram.
pub(crate) fn fit_window_end(histogram_len: usize, peak: usize) -> usize {
    (FIT_WINDOW_PEAK_MULT * peak).min(histogram_len.saturating_sub(1))
}

/// Starting shelf share, read off the histogram: k-mers between the valley and half the peak, where
/// the NB lobe is thin, doubled for the shelf's upper half, over those up to twice the peak.
fn empirical_shelf_share(histovec: &[u32], valley: usize, peak: usize) -> f64 {
    // `histovec[i]` holds count `i + 1`, so this sums counts `from..=to`.
    let mass = |from: usize, to: usize| -> f64 {
        histovec
            .iter()
            .take(to.min(histovec.len()))
            .skip(from.saturating_sub(1))
            .map(|&c| f64::from(c))
            .sum()
    };
    let lobe = mass(valley, 2 * peak);
    if lobe <= 0.0 {
        return 0.05;
    }
    (2.0 * mass(valley, peak / 2) / lobe).clamp(0.01, 0.4)
}

/// Every start for one candidate: dispersion seeds times error starts, and for the shelf, each of
/// those with the histogram's plateau share and each starting shape.
fn fit_starts(problem: &MixtureFit, dispersion_hint: f64, shelf_share: f64) -> Vec<Vec<f64>> {
    let mut starts = Vec::new();
    for seed in DISPERSION_SEEDS
        .iter()
        .copied()
        .chain(std::iter::once(dispersion_hint))
    {
        let error_starts: &[(f64, f64, f64)] = match problem.error_model {
            ErrorModel::SingletonPareto => &[(1.5, 0.8, 0.0), (2.5, 0.95, 0.0)],
            ErrorModel::FreeSingletonWeibull => {
                &[(1.5, 0.8, 0.05), (2.5, 0.9, 0.5), (2.0, 0.95, 1.0)]
            }
        };
        for &(tail_exponent, singleton_probability, shape) in error_starts {
            let mut start = vec![
                0.0,
                -3.0,
                0.0,
                (seed - 1.0).max(1e-3).ln(),
                tail_exponent.ln(),
                (singleton_probability / (1.0 - singleton_probability)).ln(),
            ];
            if problem.error_model == ErrorModel::FreeSingletonWeibull {
                start.push(shape.ln());
            }
            let logit = |p: f64| (p / (1.0 - p)).ln();
            // Starts near a two-copy lobe; the simplex moves it if the second lobe sits lower.
            let (ratio_low, ratio_high) = REPEAT_RATIO;
            start.push(logit(
                (REPEAT_RATIO_START - ratio_low) / (ratio_high - ratio_low),
            ));
            if problem.genome_model == GenomeModel::ShelfNegativeBinomial {
                for &(retained, concentration) in &SHELF_SHAPES {
                    let (low, high) = shelf_concentration_bounds(retained);
                    let mut shelf_start = start.clone();
                    shelf_start.extend([
                        logit(shelf_share / SHELF_MAX_SHARE),
                        logit((retained - SHELF_MIN_RETAINED) / (1.0 - SHELF_MIN_RETAINED)),
                        logit((concentration - low) / (high - low)),
                    ]);
                    starts.push(shelf_start);
                }
            } else {
                starts.push(start);
            }
        }
    }
    starts
}

/// One Nelder–Mead run: its best point when it converged, and how long it ran.
struct RunOutcome {
    /// (cost, θ) of the best point, when the run converged to a finite cost.
    converged: Option<(f64, Vec<f64>)>,
    iterations: u64,
    /// Whether the run stopped at its iteration cap.
    hit_cap: bool,
}

/// Actual fitting method
fn run_one(problem: MixtureFit, start: Vec<f64>, max_iters: u64) -> Result<RunOutcome, Error> {
    // Simplex initialisation
    let mut simplex = vec![start.clone()];
    for index in 0..start.len() {
        let mut vertex = start.clone();
        vertex[index] += SIMPLEX_STEP;
        simplex.push(vertex);
    }

    // Setting up optimiser
    let solver = NelderMead::new(simplex).with_sd_tolerance(SD_TOLERANCE)?;

    // Run!
    // No timer: argmin's reads the `instant` crate, whose browser clock covers wasm32 only, and nothing
    // here reads the time it measures.
    let result = Executor::new(problem, solver)
        .configure(|state| state.max_iters(max_iters))
        .timer(false)
        .run()?;

    let state = result.state();
    let termination = state.get_termination_reason();

    // We say that it converged if... it converged, plus if the cost function is finite
    let converged = if matches!(termination, Some(reason) if *reason == SolverConverged)
        && state.get_best_cost().is_finite()
        && state.get_best_cost() > 0.0
    {
        state
            .get_best_param()
            .map(|params| (state.get_best_cost(), params.clone()))
    } else {
        None
    };
    Ok(RunOutcome {
        converged,
        iterations: state.get_iter(),
        hit_cap: matches!(termination, Some(reason) if *reason == MaxItersReached),
    })
}

/// Validates every converged start before choosing the best fit for one configuration. A pathological
/// low-cost start must not hide a slightly higher-cost valid one. If none are valid, retain the
/// lowest-cost rejected candidate for diagnostics; `OptimisationFailed` means none converged.
fn attempt_from(
    problem: &MixtureFit,
    outcomes: Vec<RunOutcome>,
    valley: usize,
    empirical_peak: usize,
) -> FitAttempt {
    let mut total_iterations = 0u64;
    let mut capped_starts = 0u32;
    let mut converged = Vec::new();
    for outcome in outcomes {
        total_iterations = total_iterations.saturating_add(outcome.iterations);
        capped_starts += u32::from(outcome.hit_cap);
        let Some((cost, params)) = outcome.converged else {
            continue;
        };
        converged.push((cost, params, outcome.iterations));
    }

    // The first valid entry in cost order is the best valid fit. If all are rejected, the first
    // rejected entry is also the best one to retain for diagnostics. Stable sorting preserves start
    // order for equal costs, as before.
    converged.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mut best_rejected = None;
    for (cost, params, iterations) in converged {
        let fit = problem.unpack(&params, cost, iterations);
        match validate_fit(&fit, valley, empirical_peak) {
            None => {
                return FitAttempt {
                    error_model: problem.error_model,
                    genome_model: problem.genome_model,
                    candidate: Some(fit),
                    rejection: None,
                    total_iterations,
                    capped_starts,
                };
            }
            Some(rejection) => {
                if best_rejected.is_none() {
                    best_rejected = Some((fit, rejection));
                }
            }
        }
    }

    if let Some((fit, rejection)) = best_rejected {
        return FitAttempt {
            error_model: problem.error_model,
            genome_model: problem.genome_model,
            candidate: Some(fit),
            rejection: Some(rejection),
            total_iterations,
            capped_starts,
        };
    }

    FitAttempt {
        error_model: problem.error_model,
        genome_model: problem.genome_model,
        candidate: None,
        rejection: Some(FitRejection::OptimisationFailed),
        total_iterations,
        capped_starts,
    }
}

/// The largest mode-to-peak ratio a fit may have: [`PRIMARY_MODE_RATIO_LOW`] up to an empirical peak
/// of [`PRIMARY_MODE_RATIO_START_PEAK`], rising linearly to [`PRIMARY_MODE_RATIO_HIGH`] at
/// [`PRIMARY_MODE_RATIO_END_PEAK`].
fn primary_mode_ratio_limit(empirical_peak: usize) -> f64 {
    let progress = (empirical_peak.saturating_sub(PRIMARY_MODE_RATIO_START_PEAK) as f64
        / (PRIMARY_MODE_RATIO_END_PEAK - PRIMARY_MODE_RATIO_START_PEAK) as f64)
        .clamp(0.0, 1.0);

    PRIMARY_MODE_RATIO_LOW + (PRIMARY_MODE_RATIO_HIGH - PRIMARY_MODE_RATIO_LOW) * progress
}

/// Checker method reviewing the parameters and details of a fit result: the first check it fails, or None.
fn validate_fit(fit: &SpectrumFit, valley: usize, empirical_peak: usize) -> Option<FitRejection> {
    if !fit.log_likelihood.is_finite() || !fit.deviance.is_finite() || !fit.bic.is_finite() {
        return Some(FitRejection::NonFiniteLikelihood);
    }
    if fit.log_likelihood > 0.0 {
        return Some(FitRejection::PositiveLogLikelihood);
    }
    let weight_sum = fit.w_error + fit.w_single + fit.w_repeat;
    if !weight_sum.is_finite()
        || (weight_sum - 1.0).abs() > 1e-8
        || fit
            .component_log_totals
            .iter()
            .any(|value| value.is_nan() || *value == f64::INFINITY)
        || !fit.full_distinct_total.is_finite()
        || fit.full_distinct_total <= 0.0
    {
        return Some(FitRejection::InvalidComponents);
    }
    if !fit.mean.is_finite()
        || fit.mean <= 0.0
        || !fit.dispersion.is_finite()
        || fit.dispersion <= 1.0
        || !fit.target_mode.is_finite()
        || fit.shelf.is_some_and(|s| !shelf_is_valid(s))
    {
        return Some(FitRejection::InvalidComponents);
    }
    if fit
        .component_log_window_masses
        .iter()
        .any(|value| !value.is_finite() || *value > WINDOW_LOG_MASS_TOLERANCE)
    {
        return Some(FitRejection::InvalidWindowMass);
    }
    let log_component_total = lse3(
        fit.component_log_totals[0],
        fit.component_log_totals[1],
        fit.component_log_totals[2],
    );
    let log_component_limit =
        fit.full_distinct_total.ln() + COMPONENT_MASS_MAX_RATIO.ln() + WINDOW_LOG_MASS_TOLERANCE;
    if log_component_total > log_component_limit {
        return Some(FitRejection::ComponentMassExceeded);
    }
    if !fit.genome_kmers.is_finite()
        || !fit.error_kmers.is_finite()
        || !fit.repeat_kmers.is_finite()
    {
        return Some(FitRejection::InvalidComponents);
    }
    if fit.mode_roundtrip_error() > NB_MODE_ROUNDTRIP_TOLERANCE {
        return Some(FitRejection::UnstableModeParameterisation);
    }
    if fit.primary_mode() > fit.fit_window_end || fit.repeat_mode() > fit.fit_window_end {
        return Some(FitRejection::GenomeLobeOutsideFitWindow);
    }
    if fit.error_mode() >= valley {
        return Some(FitRejection::ErrorModePastValley);
    }
    let mode = fit.primary_mode() as f64;
    let peak = empirical_peak as f64;
    let ratio_limit = primary_mode_ratio_limit(empirical_peak);
    if mode <= 0.0 || empirical_peak == 0 || (mode / peak).max(peak / mode) > ratio_limit {
        return Some(FitRejection::PrimaryModeOutsideBand);
    }
    if (fit.repeat_mode() as f64) < 1.4 * mode {
        return Some(FitRejection::RepeatModeTooEarly);
    }
    None
}

/// The valid candidate with the lowest BIC; a tie within round-off keeps the earlier configuration.
fn select_fit(attempts: &[FitAttempt]) -> Option<SpectrumFit> {
    let mut selected: Option<SpectrumFit> = None;
    for attempt in attempts {
        let Some(candidate) = attempt
            .rejection
            .is_none()
            .then_some(attempt.candidate)
            .flatten()
        else {
            continue;
        };
        if selected.is_none_or(|best| {
            let tolerance = 1e-9 * (1.0 + best.bic.abs().max(candidate.bic.abs()));
            candidate.bic < best.bic - tolerance
        }) {
            selected = Some(candidate);
        }
    }
    selected
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

    /// Genome lobes modelled: single-copy and two-copy
    const REPEAT_LOBE_COPIES: f64 = 2.0;

    #[test]
    fn fit_matches_poisson_when_dispersion_is_one() {
        for x in [0.0, 1.0, 5.0, 40.0] {
            let nb = ln_dnbinom(x, 20.0, 1.0);
            let po = ln_dpois(x, 20.0);
            assert!((nb - po).abs() < 1e-9, "x={x}: {nb} vs {po}");
        }
        // Just past the switch the two must still be close, or the limit is discontinuous.
        assert!((ln_dnbinom(5.0, 20.0, 1.001) - ln_dpois(5.0, 20.0)).abs() < 0.05);
    }

    #[cfg(not(target_family = "wasm"))]
    fn synthetic_histogram(error: ErrorParams) -> Vec<u32> {
        let mut histogram = vec![0u32; 601];
        let mean = 53.0;
        for (index, bin) in histogram.iter_mut().enumerate() {
            let count = (index + 1) as f64;
            let error_height = 400_000.0 * error.log_probability(count).exp();
            let single = 200_000.0 * ln_dnbinom(count, mean, 4.0).exp();
            let repeat = 15_000.0 * ln_dnbinom(count, 2.0 * mean, 4.0).exp();
            *bin = (error_height + single + repeat).round() as u32;
        }
        histogram
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn capped_starts_count_iterations_but_are_not_candidates() {
        let histogram = synthetic_histogram(ErrorParams {
            singleton_probability: 0.85,
            tail_exponent: 1.8,
            weibull_shape: None,
        });
        let observed: Arc<[(f64, f64)]> = histogram[..300]
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(index, &count)| ((index + 1) as f64, f64::from(count)))
            .collect::<Vec<_>>()
            .into();
        let problem = MixtureFit {
            observed_total: observed.iter().map(|&(_, count)| count).sum(),
            full_distinct_total: histogram.iter().map(|&n| f64::from(n)).sum(),
            observed,
            peak: 50.0,
            fit_window_end: 300,
            error_model: ErrorModel::SingletonPareto,
            genome_model: GenomeModel::NegativeBinomial,
            sample_scale: 1.0,
        };
        let attempt = fit_candidate(problem, 12, 50, 4.0, 0.05, 1);
        assert_eq!(attempt.capped_starts, 8);
        assert_eq!(attempt.total_iterations, 8);
        assert!(attempt.candidate.is_none());
        assert_eq!(attempt.rejection, Some(FitRejection::OptimisationFailed));
    }

    /// A(1, 1) beta-binomial is uniform over its trials, and any beta-binomial sums to one.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn the_beta_binomial_is_normalised_and_uniform_at_one_one() {
        for x in 0..=50 {
            assert!((ln_dbetabinom(f64::from(x), 50.0, 1.0, 1.0).exp() - 1.0 / 51.0).abs() < 1e-12);
        }
        let total: f64 = (0..=60)
            .map(|x| ln_dbetabinom(f64::from(x), 60.0, 2.5, 0.7).exp())
            .sum();
        assert!((total - 1.0).abs() < 1e-10);
        assert_eq!(ln_dbetabinom(61.0, 60.0, 2.5, 0.7), f64::NEG_INFINITY);
    }

    /// The fit's recurrence table agrees with the direct log-gamma density at every count.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn the_shelf_table_matches_the_direct_density() {
        for (alpha, beta) in [(1.0, 1.0), (2.5, 0.7), (0.4, 0.8), (3.0, 1.0)] {
            let shelf = Shelf {
                share: 0.1,
                alpha,
                beta,
            };
            let shelf_table = shelf_log_table(147.4, shelf, 1_000);
            assert_eq!(shelf_table.len(), 148);
            for (count, &ln) in shelf_table.iter().enumerate() {
                let direct = ln_dbetabinom(count as f64, 147.0, alpha, beta);
                assert!((ln - direct).abs() < 1e-9, "a={alpha} b={beta} x={count}");
            }
            // A window shorter than the trials cuts the table without changing its values.
            let short = shelf_log_table(147.4, shelf, 60);
            assert_eq!(short.len(), 61);
            assert_eq!(short[..], shelf_table[..61]);
        }
    }

    /// The shelf model with a Pareto tail, over the NB-only synthetic spectrum's first 300 counts.
    #[cfg(not(target_family = "wasm"))]
    fn shelf_problem() -> MixtureFit {
        let histogram = synthetic_histogram(ErrorParams {
            singleton_probability: 0.85,
            tail_exponent: 1.8,
            weibull_shape: None,
        });
        let observed: Arc<[(f64, f64)]> = histogram[..300]
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(index, &count)| ((index + 1) as f64, f64::from(count)))
            .collect::<Vec<_>>()
            .into();
        MixtureFit {
            observed_total: observed.iter().map(|&(_, count)| count).sum(),
            full_distinct_total: histogram.iter().map(|&n| f64::from(n)).sum(),
            observed,
            peak: 50.0,
            fit_window_end: 300,
            error_model: ErrorModel::SingletonPareto,
            genome_model: GenomeModel::ShelfNegativeBinomial,
            sample_scale: 1.0,
        }
    }

    #[cfg(not(target_family = "wasm"))]
    fn valid_fit_for_validation_tests() -> SpectrumFit {
        SpectrumFit::for_test(
            ErrorModel::SingletonPareto,
            GenomeModel::NegativeBinomial,
            [0.7, 0.29, 0.01],
            106.0,
            7.0,
            ErrorParams {
                singleton_probability: 0.85,
                tail_exponent: 2.0,
                weibull_shape: None,
            },
            100.0,
            600,
        )
    }

    /// Each validation limit accepts its boundary and rejects just past it: a window log-mass above round-off,
    /// an NB mode off by more than half a count, and component totals above 1.10x the *full* spectrum.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn validation_limits_are_inclusive_at_their_boundaries() {
        let mut fit = valid_fit_for_validation_tests();
        fit.component_log_window_masses[0] = WINDOW_LOG_MASS_TOLERANCE;
        assert_eq!(validate_fit(&fit, 12, 100), None);
        fit.component_log_window_masses[0] = WINDOW_LOG_MASS_TOLERANCE + 1e-15;
        assert_eq!(
            validate_fit(&fit, 12, 100),
            Some(FitRejection::InvalidWindowMass)
        );

        let mut fit = valid_fit_for_validation_tests();
        fit.target_mode += NB_MODE_ROUNDTRIP_TOLERANCE;
        assert_eq!(validate_fit(&fit, 12, 100), None);
        fit.target_mode += 1e-6;
        assert_eq!(
            validate_fit(&fit, 12, 100),
            Some(FitRejection::UnstableModeParameterisation)
        );

        // The full spectrum holds 10 k-mers; the fit window only 7, which would reject 10.9.
        let mut histogram = vec![0u32; 300];
        histogram[0] = 4;
        histogram[99] = 3;
        histogram[299] = 3;
        assert_eq!(
            histogram[..220].iter().map(|&n| u64::from(n)).sum::<u64>(),
            7
        );
        let mut fit = valid_fit_for_validation_tests();
        fit.fit_window_end = 220;
        fit.full_distinct_total = histogram.iter().map(|&n| f64::from(n)).sum();
        assert_eq!(fit.full_distinct_total, 10.0);
        // (repeat k-mers, accepted); error and genome hold 5 each
        for (repeat, accepted) in [(0.9_f64, true), (1.0, true), (1.001, false)] {
            fit.component_log_totals = [5.0_f64.ln(), 5.0_f64.ln(), repeat.ln()];
            fit.error_kmers = 5.0;
            fit.genome_kmers = 5.0;
            fit.repeat_kmers = repeat;
            let expected = (!accepted).then_some(FitRejection::ComponentMassExceeded);
            assert_eq!(
                validate_fit(&fit, 12, 100),
                expected,
                "repeat k-mers {repeat}"
            );
        }
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn a_lower_cost_rejected_start_does_not_hide_a_valid_start() {
        let error = ErrorParams {
            singleton_probability: 0.85,
            tail_exponent: 1.8,
            weibull_shape: None,
        };
        let histogram = synthetic_histogram(error);
        let observed: Arc<[(f64, f64)]> = histogram[..300]
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(index, &count)| ((index + 1) as f64, f64::from(count)))
            .collect::<Vec<_>>()
            .into();
        let observed_total = observed.iter().map(|&(_, count)| count).sum::<f64>();
        let problem = MixtureFit {
            observed_total,
            full_distinct_total: observed_total * 5.0,
            observed,
            peak: 50.0,
            fit_window_end: 300,
            error_model: ErrorModel::SingletonPareto,
            genome_model: GenomeModel::NegativeBinomial,
            sample_scale: 1.0,
        };
        let logit = |probability: f64| (probability / (1.0 - probability)).ln();
        let valid_params = vec![
            0.0,
            -3.0,
            0.0,
            3.0_f64.ln(),
            1.8_f64.ln(),
            logit(0.85),
            logit(0.8),
        ];
        let mut excessive_mass_params = valid_params.clone();
        excessive_mass_params[4] = 0.01_f64.ln();
        excessive_mass_params[5] = logit(0.01);

        let attempt = attempt_from(
            &problem,
            vec![
                RunOutcome {
                    converged: Some((1.0, excessive_mass_params.clone())),
                    iterations: 10,
                    hit_cap: false,
                },
                RunOutcome {
                    converged: Some((2.0, valid_params)),
                    iterations: 12,
                    hit_cap: false,
                },
            ],
            12,
            50,
        );
        assert_eq!(attempt.rejection, None);
        assert_eq!(attempt.candidate.unwrap().primary_mode(), 52);

        let all_rejected = attempt_from(
            &problem,
            vec![RunOutcome {
                converged: Some((1.0, excessive_mass_params)),
                iterations: 10,
                hit_cap: false,
            }],
            12,
            50,
        );
        assert!(all_rejected.candidate.is_some());
        assert_eq!(
            all_rejected.rejection,
            Some(FitRejection::ComponentMassExceeded)
        );
    }

    /// However far the simplex pushes its coordinates, the shelf stays off the error tail, keeps a >= 1
    /// and never becomes a spike, and the two-copy lobe stays within its ratio bounds.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn simplex_coordinates_map_inside_their_bounds() {
        let problem = shelf_problem();
        let extremes = [-50.0, 0.0, 50.0];
        for share in extremes {
            for retained in extremes {
                for concentration in extremes {
                    let theta = [
                        0.0,
                        -3.0,
                        0.0,
                        1.0,
                        0.5,
                        1.7,
                        0.0,
                        share,
                        retained,
                        concentration,
                    ];
                    let shelf = problem.params(&theta).shelf.expect("the shelf model");
                    assert!((0.0..=SHELF_MAX_SHARE).contains(&shelf.share), "{shelf:?}");
                    assert!(
                        (SHELF_MIN_RETAINED - 1e-12..=1.0).contains(&shelf.retained()),
                        "{shelf:?}"
                    );
                    assert!(shelf.alpha >= 1.0 - 1e-12, "{shelf:?}");
                    assert!(
                        shelf.concentration() <= SHELF_MAX_CONCENTRATION + 1e-12,
                        "{shelf:?}"
                    );
                }
            }
        }
        let (low, high) = REPEAT_RATIO;
        for ratio in extremes {
            let theta = [0.0, -3.0, 0.0, 1.0, 0.5, 1.7, ratio, 0.0, 0.0, 0.0];
            let fitted = problem.params(&theta).repeat_ratio;
            assert!((low - 1e-12..=high + 1e-12).contains(&fitted), "{fitted}");
        }
    }

    /// A second lobe at 1.7 times the single-copy mean, not 2, is followed by the fit.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn a_second_lobe_off_the_double_is_followed() {
        let error = ErrorParams {
            singleton_probability: 0.85,
            tail_exponent: 1.8,
            weibull_shape: None,
        };
        let (mean, ratio) = (53.0, 1.7);
        let mut histogram = vec![0u32; 601];
        for (index, bin) in histogram.iter_mut().enumerate() {
            let count = (index + 1) as f64;
            let error_height = 400_000.0 * error.log_probability(count).exp();
            let single = 200_000.0 * ln_dnbinom(count, mean, 4.0).exp();
            let second = 60_000.0 * ln_dnbinom(count, ratio * mean, 4.0).exp();
            *bin = (error_height + single + second).round() as u32;
        }
        let selected = fit_spectrum(&histogram, 12, 50, 4.0, 1.0)
            .expect("the fit runs")
            .selected
            .expect("one candidate survives");
        assert!((1.6..=1.8).contains(&selected.repeat_ratio), "{selected:?}");
    }

    /// A simplex probe at an absurd dispersion (a mean near 1e13) returns promptly, as it did before
    /// the table: no allocation proportional to the mean.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn a_shelf_cost_at_an_extreme_dispersion_stays_bounded() {
        let problem = shelf_problem();
        for log_dispersion in [30.0, 60.0, 700.0] {
            let theta = vec![
                0.0,
                -3.0,
                0.0,
                log_dispersion,
                1.8f64.ln(),
                1.7,
                0.0,
                -1.0,
                0.0,
                0.0,
            ];
            let started = std::time::Instant::now();
            let cost = problem.cost(&theta).expect("the cost never errors");
            assert!(cost.is_finite() || cost == f64::INFINITY);
            assert!(
                started.elapsed().as_secs() < 5,
                "log dispersion {log_dispersion}"
            );
        }
        // The table behind that cost stops at the fit window, however large the mean.
        let shelf = Shelf {
            share: 0.1,
            alpha: 1.0,
            beta: 1.0,
        };
        for mean in [1e6, 1e12, 1e19, 1e300] {
            assert_eq!(shelf_log_table(mean, shelf, 300).len(), 301, "mean {mean}");
        }
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn single_copy_mass_mixes_the_lobe_and_the_shelf() {
        let shelf = Shelf {
            share: 0.12,
            alpha: 2.0,
            beta: 0.8,
        };
        for shelf in [None, Some(shelf)] {
            let shelf_table = shelf.map(|s| shelf_log_table(53.0, s, 300));
            let direct = (1..=300).fold(f64::NEG_INFINITY, |acc, count| {
                lse2(acc, ln_single_copy(f64::from(count), 53.0, 4.0, shelf))
            });
            let mass = single_copy_log_mass(53.0, 4.0, shelf.zip(shelf_table.as_deref()), 300);
            assert!(
                (mass - direct).abs() < 1e-9,
                "{shelf:?}: {mass} != {direct}"
            );
        }
    }

    /// A spectrum with a tenth of its single-copy k-mers in a uniform shelf selects the shelf model,
    /// and the hole guard then counts the shelf's low counts as genome.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn the_shelf_model_recovers_a_synthetic_shelf() {
        let error = ErrorParams {
            singleton_probability: 0.85,
            tail_exponent: 1.8,
            weibull_shape: None,
        };
        let (mean, share) = (53.0, 0.1);
        let mut histogram = vec![0u32; 601];
        for (index, bin) in histogram.iter_mut().enumerate() {
            let count = (index + 1) as f64;
            let error_height = 400_000.0 * error.log_probability(count).exp();
            let single = 200_000.0
                * ((1.0 - share) * ln_dnbinom(count, mean, 4.0).exp()
                    + share * ln_dbetabinom(count, mean, 1.0, 1.0).exp());
            let repeat = 15_000.0 * ln_dnbinom(count, 2.0 * mean, 4.0).exp();
            *bin = (error_height + single + repeat).round() as u32;
        }
        let result = fit_spectrum(&histogram, 12, 50, 4.0, 1.0).expect("synthetic fit should run");
        let selected = result.selected.expect("one candidate should survive");
        assert_eq!(
            selected.genome_model,
            GenomeModel::ShelfNegativeBinomial,
            "{result:?}"
        );
        let fitted = selected.shelf.expect("the shelf model carries a shelf");
        assert!((0.05..=0.15).contains(&fitted.share), "{fitted:?}");
        assert!(fitted.retained() >= SHELF_MIN_RETAINED, "{fitted:?}");
        assert!(shelf_is_valid(fitted), "{fitted:?}");
        assert_eq!(selected.hole_cutoff(1.0), Some(2));
    }

    /// A spectrum rescaled from a sample fits exactly as the sample does, with genome k-mers, heights
    /// and so the hole guard at library scale.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn a_rescaled_sample_fits_as_the_sample_itself() {
        let error = ErrorParams {
            singleton_probability: 0.85,
            tail_exponent: 1.8,
            weibull_shape: None,
        };
        let sample = synthetic_histogram(error);
        let scale = 228.0;
        let rescaled: Vec<u32> = sample
            .iter()
            .map(|&n| (f64::from(n) * scale) as u32)
            .collect();
        let raw = fit_spectrum(&sample, 12, 50, 4.0, 1.0)
            .unwrap()
            .selected
            .unwrap();
        let fit = fit_spectrum(&rescaled, 12, 50, 4.0, scale)
            .unwrap()
            .selected
            .unwrap();
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * a.abs().max(b.abs());
        assert_eq!(fit.genome_model, raw.genome_model);
        assert!(
            close(fit.bic, raw.bic) && close(fit.mean, raw.mean),
            "{fit:?} {raw:?}"
        );
        assert!(close(fit.genome_kmers, scale * raw.genome_kmers));
        assert!(close(
            fit.component_heights(30)[1],
            scale * raw.component_heights(30)[1]
        ));
        assert_eq!(fit.hole_cutoff(scale), raw.hole_cutoff(1.0));
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn free_singleton_weibull_nests_ordinary_weibull_geometric_and_one_inflated() {
        for (scale, shape) in [(1.2_f64, 0.6_f64), (0.7, 1.0)] {
            let base_singleton = -(-scale).exp_m1();
            for inflation in [0.0, 0.3] {
                let error = ErrorParams {
                    singleton_probability: base_singleton + inflation * (1.0 - base_singleton),
                    tail_exponent: scale * shape,
                    weibull_shape: Some(shape),
                };
                for count in 1..=30 {
                    let count = count as f64;
                    let base = if count == 1.0 {
                        base_singleton
                    } else {
                        (-scale * (count - 1.0).powf(shape)).exp()
                            - (-scale * count.powf(shape)).exp()
                    };
                    let expected = if count == 1.0 {
                        inflation + (1.0 - inflation) * base
                    } else {
                        (1.0 - inflation) * base
                    };
                    assert!((error.log_probability(count).exp() - expected).abs() < 1e-12);
                    if shape == 1.0 && inflation == 0.0 {
                        let geometric = base_singleton * (-scale * (count - 1.0)).exp();
                        assert!((expected - geometric).abs() < 1e-12);
                    }
                }
            }
        }
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn pareto_is_the_stable_weibull_boundary() {
        let pareto = ErrorParams {
            singleton_probability: 0.85,
            tail_exponent: 1.8,
            weibull_shape: None,
        };
        let weibull = ErrorParams {
            weibull_shape: Some(1e-9),
            ..pareto
        };
        for count in 1..=1000 {
            let difference = (pareto.log_probability(count as f64)
                - weibull.log_probability(count as f64))
            .abs();
            assert!(difference < 1e-7, "count={count} difference={difference}");
        }
        for error in [
            pareto,
            ErrorParams {
                weibull_shape: Some(0.5),
                ..pareto
            },
        ] {
            let top = 570;
            let sum = (1..=top)
                .map(|count| error.log_probability(count as f64).exp())
                .sum::<f64>();
            assert!((sum - error.log_window_mass(top).exp()).abs() < 1e-10);
        }
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn fit_recovers_the_pareto_tail_with_negative_binomial_genomes() {
        let error = ErrorParams {
            singleton_probability: 0.85,
            tail_exponent: 1.8,
            weibull_shape: None,
        };
        let result = fit_spectrum(&synthetic_histogram(error), 12, 50, 4.0, 1.0)
            .expect("synthetic fit should run");
        let configurations: Vec<_> = result
            .attempts
            .iter()
            .map(|attempt| (attempt.error_model, attempt.genome_model))
            .collect();
        assert_eq!(
            configurations,
            [
                (ErrorModel::SingletonPareto, GenomeModel::NegativeBinomial),
                (
                    ErrorModel::FreeSingletonWeibull,
                    GenomeModel::NegativeBinomial
                ),
                (
                    ErrorModel::SingletonPareto,
                    GenomeModel::ShelfNegativeBinomial
                ),
                (
                    ErrorModel::FreeSingletonWeibull,
                    GenomeModel::ShelfNegativeBinomial
                ),
            ]
        );
        // No shelf in the spectrum, so BIC must keep the plain NB.
        let selected = result.selected.expect("one candidate should survive");
        assert_eq!(selected.genome_model, GenomeModel::NegativeBinomial);
        assert_eq!(
            selected.error_model,
            ErrorModel::SingletonPareto,
            "{result:?}"
        );
        assert!(selected.primary_mode().abs_diff(50) <= 3, "{result:?}");
        assert!(
            (selected.error_params.tail_exponent - 1.8).abs() < 0.5,
            "{result:?}"
        );
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn fit_recovers_the_weibull_tail_with_negative_binomial_genomes() {
        let error = ErrorParams {
            singleton_probability: 0.78,
            tail_exponent: 1.5,
            weibull_shape: Some(0.8),
        };
        let result = fit_spectrum(&synthetic_histogram(error), 12, 50, 4.0, 1.0)
            .expect("synthetic fit should run");
        let selected = result.selected.expect("one candidate should survive");
        assert_eq!(
            selected.error_model,
            ErrorModel::FreeSingletonWeibull,
            "{result:?}"
        );
        assert!(selected.primary_mode().abs_diff(50) <= 3, "{result:?}");
        assert!(selected.error_params.weibull_shape.is_some());
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn bic_ties_prefer_pareto_and_lower_bic_selects_weibull() {
        let error = ErrorParams {
            singleton_probability: 0.85,
            tail_exponent: 2.0,
            weibull_shape: None,
        };
        let accepted = |fit: SpectrumFit| FitAttempt {
            error_model: fit.error_model,
            genome_model: fit.genome_model,
            candidate: Some(fit),
            rejection: None,
            total_iterations: 1,
            capped_starts: 0,
        };
        let make = |error_model, genome_model| {
            let error_params = ErrorParams {
                weibull_shape: (error_model == ErrorModel::FreeSingletonWeibull).then_some(0.5),
                ..error
            };
            let mut fit = SpectrumFit::for_test(
                error_model,
                genome_model,
                [0.7, 0.29, 0.01],
                100.0,
                6.0,
                error_params,
                10.0e6,
                570,
            );
            fit.bic = 1.0e8;
            accepted(fit)
        };
        let mut attempts = [
            make(ErrorModel::SingletonPareto, GenomeModel::NegativeBinomial),
            make(
                ErrorModel::FreeSingletonWeibull,
                GenomeModel::NegativeBinomial,
            ),
        ];
        let selected = select_fit(&attempts).expect("a fit should be selected");
        assert_eq!(
            (selected.error_model, selected.genome_model),
            (ErrorModel::SingletonPareto, GenomeModel::NegativeBinomial)
        );
        attempts[1].candidate.as_mut().unwrap().bic -= 2.0;
        let selected = select_fit(&attempts).expect("a fit should be selected");
        assert_eq!(
            (selected.error_model, selected.genome_model),
            (
                ErrorModel::FreeSingletonWeibull,
                GenomeModel::NegativeBinomial
            )
        );
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn bad_srr_like_mode_is_rejected() {
        let fit = SpectrumFit::for_test(
            ErrorModel::SingletonPareto,
            GenomeModel::NegativeBinomial,
            [0.877, 0.123, 0.0],
            417.5,
            139.22,
            ErrorParams {
                singleton_probability: 0.85,
                tail_exponent: 2.0,
                weibull_shape: None,
            },
            18.0e6,
            2478,
        );
        assert_eq!(fit.primary_mode(), 279);
        assert_eq!(
            validate_fit(&fit, 12, 413),
            Some(FitRejection::PrimaryModeOutsideBand)
        );
    }

    /// The primary mode must lie within a ratio of the empirical peak, ramping from 1.25 (peak <= 100) to
    /// 1.40 (peak >= 200), on both sides and inclusively.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn the_primary_mode_band_ramps_with_the_peak_and_is_inclusive() {
        for (peak, limit) in [
            (0, 1.25),
            (99, 1.25),
            (100, 1.25),
            (150, 1.325),
            (200, 1.40),
            (201, 1.40),
        ] {
            assert_eq!(primary_mode_ratio_limit(peak), limit, "peak {peak}");
        }
        let make_fit = |mode: usize| {
            SpectrumFit::for_test(
                ErrorModel::SingletonPareto,
                GenomeModel::NegativeBinomial,
                [0.7, 0.29, 0.01],
                mode as f64 + 6.0,
                7.0,
                ErrorParams {
                    singleton_probability: 0.85,
                    tail_exponent: 2.0,
                    weibull_shape: None,
                },
                10.0e6,
                1_000,
            )
        };
        // (empirical peak, fitted mode, accepted)
        for (peak, mode, accepted) in [
            (100, 125, true),
            (100, 126, false),
            (150, 198, true),
            (150, 199, false),
            (150, 114, true),
            (150, 113, false),
            (200, 280, true),
            (200, 281, false),
            (200, 143, true),
            (200, 142, false),
            (250, 280, true),
            (0, 125, false),
        ] {
            let fit = make_fit(mode);
            assert_eq!(fit.primary_mode(), mode);
            let expected = (!accepted).then_some(FitRejection::PrimaryModeOutsideBand);
            assert_eq!(
                validate_fit(&fit, 12, peak),
                expected,
                "peak {peak}, mode {mode}"
            );
        }
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn degenerate_genome_parameters_and_modes_outside_fit_window_are_rejected() {
        let params = ErrorParams {
            singleton_probability: 0.85,
            tail_exponent: 2.0,
            weibull_shape: None,
        };
        for (mean, dispersion) in [(0.0, 4.0), (100.0, 1.0), (f64::INFINITY, 4.0)] {
            let fit = SpectrumFit::for_test(
                ErrorModel::SingletonPareto,
                GenomeModel::NegativeBinomial,
                [0.7, 0.29, 0.01],
                mean,
                dispersion,
                params,
                1.0e6,
                120,
            );
            assert_eq!(
                validate_fit(&fit, 12, 100),
                Some(FitRejection::InvalidComponents),
                "mean={mean}, dispersion={dispersion}"
            );
        }

        let outside = SpectrumFit::for_test(
            ErrorModel::SingletonPareto,
            GenomeModel::NegativeBinomial,
            [0.7, 0.29, 0.01],
            100.0,
            5.0,
            params,
            1.0e6,
            120,
        );
        assert!(outside.primary_mode() <= outside.fit_window_end);
        assert!(outside.repeat_mode() > outside.fit_window_end);
        assert_eq!(
            validate_fit(&outside, 12, outside.primary_mode()),
            Some(FitRejection::GenomeLobeOutsideFitWindow)
        );
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn cutoff_searches_stop_at_the_fit_window() {
        let fit = SpectrumFit::for_test(
            ErrorModel::SingletonPareto,
            GenomeModel::NegativeBinomial,
            [0.999_999_999, 0.000_000_001, 0.0],
            100.0,
            5.0,
            ErrorParams {
                singleton_probability: 0.85,
                tail_exponent: 2.0,
                weibull_shape: None,
            },
            1.0e6,
            20,
        );
        assert_eq!(fit.crossover(), None);
        assert_eq!(fit.hole_cutoff(f64::INFINITY), None);
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn components_are_conditioned_on_the_fit_window() {
        for model in [
            ErrorModel::SingletonPareto,
            ErrorModel::FreeSingletonWeibull,
        ] {
            let error = ErrorParams {
                singleton_probability: 0.85,
                tail_exponent: 2.0,
                weibull_shape: (model == ErrorModel::FreeSingletonWeibull).then_some(0.5),
            };
            let fit = SpectrumFit::for_test(
                model,
                GenomeModel::NegativeBinomial,
                [0.7, 0.29, 0.01],
                100.0,
                6.0,
                error,
                10.0e6,
                570,
            );
            let sums = (1..=fit.fit_window_end).fold([0.0; 3], |mut sums, count| {
                let heights = fit.component_heights(count);
                for index in 0..3 {
                    sums[index] += heights[index] / fit.observed_total;
                }
                sums
            });
            for (actual, expected) in sums.into_iter().zip([0.7, 0.29, 0.01]) {
                assert!((actual - expected).abs() < 1e-8, "{actual} != {expected}");
            }
            assert!((fit.repeat_mode() as f64) >= 1.4 * fit.primary_mode() as f64);
        }
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn active_error_floor_is_one_past_the_last_overlapping_count() {
        for model in [
            ErrorModel::SingletonPareto,
            ErrorModel::FreeSingletonWeibull,
        ] {
            let error = ErrorParams {
                singleton_probability: 0.85,
                tail_exponent: 2.0,
                weibull_shape: (model == ErrorModel::FreeSingletonWeibull).then_some(0.5),
            };
            let fit = SpectrumFit::for_test(
                model,
                GenomeModel::NegativeBinomial,
                [0.8, 0.19, 0.01],
                100.0,
                6.0,
                error,
                10.0e6,
                570,
            );
            let floor = fit.error_floor_cutoff();
            if floor > 2 {
                let reference = fit.component_heights(fit.primary_mode())[1];
                assert!(fit.component_heights(usize::from(floor - 1))[0] >= reference);
                if usize::from(floor) <= fit.fit_window_end {
                    assert!(fit.component_heights(usize::from(floor))[0] < reference);
                }
            }
        }
    }

    #[cfg(not(target_family = "wasm"))]
    impl SpectrumFit {
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn for_test(
            error_model: ErrorModel,
            genome_model: GenomeModel,
            weights: [f64; 3],
            mean: f64,
            dispersion: f64,
            error_params: ErrorParams,
            observed_total: f64,
            fit_window_end: usize,
        ) -> Self {
            let window_masses = [
                error_params.log_window_mass(fit_window_end),
                single_copy_log_mass(mean, dispersion, None, fit_window_end),
                log_nb_mass(REPEAT_LOBE_COPIES * mean, dispersion, fit_window_end),
            ];
            let target_mode = mean - (dispersion - 1.0);
            let log_observed_total = observed_total.ln();
            let component_log_totals = [
                weights[0].ln() + log_observed_total - window_masses[0],
                weights[1].ln() + log_observed_total - window_masses[1],
                weights[2].ln() + log_observed_total - window_masses[2],
            ];
            Self {
                error_model,
                genome_model,
                w_error: weights[0],
                w_single: weights[1],
                w_repeat: weights[2],
                mean,
                dispersion,
                target_mode,
                error_params,
                shelf: None,
                repeat_ratio: REPEAT_LOBE_COPIES,
                genome_kmers: component_log_totals[1].exp(),
                error_kmers: component_log_totals[0].exp(),
                repeat_kmers: component_log_totals[2].exp(),
                log_likelihood: -1.0,
                bic: 2.0
                    + (error_model.parameter_count() + genome_model.extra_parameters()) as f64
                        * observed_total.ln(),
                deviance: 1.0,
                fit_window_end,
                best_iterations: 1,
                component_log_window_masses: window_masses,
                component_log_totals,
                observed_total,
                // Existing hand-built test fits focus on their specific guard. Dedicated mass tests set
                // this to a realistic whole-spectrum total.
                full_distinct_total: f64::MAX,
            }
        }
    }

    /// One candidate's starts run in turn, as the whole search once did.
    #[cfg(not(target_family = "wasm"))]
    fn fit_candidate(
        problem: MixtureFit,
        valley: usize,
        empirical_peak: usize,
        dispersion_hint: f64,
        shelf_share: f64,
        max_iters: u64,
    ) -> FitAttempt {
        let outcomes = fit_starts(&problem, dispersion_hint, shelf_share)
            .into_iter()
            .filter_map(|start| run_one(problem.clone(), start, max_iters).ok())
            .collect();
        attempt_from(&problem, outcomes, valley, empirical_peak)
    }
}
