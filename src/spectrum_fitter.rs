//! Fits a coverage model to the k-mer spectrum, to choose a count cutoff.
//!
//! Three-component coverage mixture. Native builds compare two singleton-aware error tails against
//! negative-binomial and Normal genomic lobes; WASM retains its legacy Poisson/NB mixture.
//!
//! ====================================================== Got from ska-rust!! =====
//!
//! [`fit_spectrum`] is the main interface.

#[cfg(not(target_family = "wasm"))]
use argmin::core::TerminationReason::MaxItersReached;
use argmin::{
    core::{CostFunction, Error, Executor, State, TerminationReason::SolverConverged},
    solver::neldermead::NelderMead,
};
use libm::lgamma;
#[cfg(not(target_family = "wasm"))]
use std::{fmt, sync::Arc};

/// Genome lobes modelled: single-copy and two-copy. The two-copy lobe carries little weight but
/// absorbs the right tail, which otherwise inflates the dispersion by 20-60 %. A third lobe was
/// measured at a fitted weight of 0.000-0.009 and is not worth its parameter.
const REPEAT_LOBE_COPIES: f64 = 2.0;

/// The fitted mean is pinned to this window around the peak the valley estimator found. Unpinned, the
/// model relabels the observed peak as a repeat lobe and puts a near-empty single-copy lobe at half or
/// a third of the coverage — an identifiability failure that costs nothing in likelihood.
const MU_LO: f64 = 0.75;
const MU_SPAN: f64 = 0.58;

/// Dispersions to restart the fit from, alongside the valley estimator's own guess.
///
/// A single start is not enough: `dispersion = 1 + exp(theta)` flattens as `theta` falls, so the
/// simplex can crawl out to the Poisson boundary and stop there while still reporting convergence.
/// Measured, that happened in half the spectra tried, once costing 6.7e6 in log-likelihood.
const DISPERSION_SEEDS: [f64; 3] = [2.0, 8.0, 15.0];

/// Counts above this multiple of the peak are not fitted: they are repeats beyond the two-copy lobe.
const FIT_WINDOW_PEAK_MULT: usize = 6;

pub(crate) fn fit_window_end(histogram_len: usize, peak: usize) -> usize {
    (FIT_WINDOW_PEAK_MULT * peak).min(histogram_len.saturating_sub(1))
}

pub(crate) fn fitted_distinct_total(histovec: &[u32], peak: usize) -> f64 {
    let top = fit_window_end(histovec.len(), peak);
    histovec[..top].iter().map(|&n| f64::from(n)).sum()
}

const MAX_ITERS: u64 = 5_000;
#[cfg(not(target_family = "wasm"))]
const NATIVE_MAX_ITERS: u64 = 500;
/// Offset applied to one coordinate at a time to build the initial simplex.
const SIMPLEX_STEP: f64 = 0.5;
/// Argmin defaults to machine epsilon, which is needlessly strict for likelihoods around 1e7 and
/// makes an otherwise stationary simplex run to [`NATIVE_MAX_ITERS`].
#[cfg(not(target_family = "wasm"))]
const NATIVE_SD_TOLERANCE: f64 = 1e-4;

/// A fitted coverage model.
#[derive(Clone, Copy, Debug)]
pub struct SpectrumFit {
    /// Share of distinct k-mers that are sequencing errors.
    pub w_error: f64,
    /// Share that are single-copy genome.
    pub w_single: f64,
    /// Share that are two-copy repeat.
    pub w_repeat: f64,
    /// Single-copy coverage: the mean of the genome lobe.
    pub mean: f64,
    /// Variance-to-mean ratio of the genome lobe; 1.0 is Poisson.
    pub dispersion: f64,
    /// Mean of the Poisson error lobe. Measured at 1.0-1.15 on real libraries.
    pub error_mean: f64,
    /// Distinct single-copy k-mers the fit implies, i.e. an estimate of genome size.
    pub genome_kmers: f64,
}

impl SpectrumFit {
    /// Weighted probabilities of the error, single-copy and two-copy components at one abundance.
    #[cfg(not(target_family = "wasm"))]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn component_probabilities(&self, count: f64) -> [f64; 3] {
        [
            self.w_error * ln_dpois(count, self.error_mean).exp(),
            self.w_single * ln_dnbinom(count, self.mean, self.dispersion).exp(),
            self.w_repeat
                * ln_dnbinom(count, REPEAT_LOBE_COPIES * self.mean, self.dispersion).exp(),
        ]
    }

    /// Upper mode of the fitted single-copy negative binomial.
    ///
    /// With `variance = dispersion * mean`, the mode is
    /// `floor(mean - (dispersion - 1))`, bounded below by zero. When two adjacent integer modes tie,
    /// this convention returns the upper one.
    #[cfg(not(target_family = "wasm"))]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn single_copy_mode(&self) -> u32 {
        let mode = self.mean - (self.dispersion - 1.0);
        if mode.is_finite() && mode > 0.0 {
            mode.floor().min(f64::from(u32::MAX)) as u32
        } else {
            0
        }
    }

    /// Largest cutoff expected to strand at most `budget` single-copy k-mers below itself.
    ///
    /// Each stranded k-mer is a hole the graph cannot bridge, so it severs a contig. This bounds their
    /// number rather than their share: a share far under 1 % still hides tens of thousands of breaks.
    pub fn hole_cutoff(&self, budget: f64) -> u16 {
        if self.genome_kmers <= 0.0 || !self.mean.is_finite() || self.mean < 2.0 {
            return 2;
        }
        let allowed = budget / self.genome_kmers;
        let ceiling = self.mean.round() as usize;

        let mut cdf = ln_dnbinom(0.0, self.mean, self.dispersion).exp();
        let mut m = 1usize;
        while m < ceiling {
            let next = cdf + ln_dnbinom(m as f64, self.mean, self.dispersion).exp();
            if next > allowed {
                break;
            }
            cdf = next;
            m += 1;
        }
        (m as u16).max(2)
    }

    /// The count at which the genome lobe overtakes the error lobe: the mixture's own decision
    /// boundary. Logged for comparison; it sits near the valley and is the more conservative choice.
    pub fn crossover(&self) -> u16 {
        let ceiling = (self.mean.round() as usize).max(2);
        for c in 1..ceiling {
            let err = self.w_error.ln() + ln_dpois(c as f64, self.error_mean);
            let gen = self.w_single.ln() + ln_dnbinom(c as f64, self.mean, self.dispersion);
            if gen > err {
                return (c as u16).max(2);
            }
        }
        (ceiling as u16).max(2)
    }
}

/// Fit the coverage model to a spectrum, seeded from the valley estimator's peak and dispersion.
///
/// `histovec[c - 1]` is the number of distinct k-mers seen `c` times. Returns `Err` if no start
/// converged, which the caller should treat as "keep the estimator's own answer".
pub fn fit_spectrum(
    histovec: &[u32],
    peak: usize,
    dispersion_hint: f64,
) -> Result<SpectrumFit, Error> {
    if peak < 2 || histovec.len() < 2 {
        return Err(Error::msg("no usable genome peak to fit around"));
    }
    // Counts and weights inside the window, skipping empty bins: they contribute nothing to the
    // likelihood and every one would cost three density evaluations per iteration.
    let top = fit_window_end(histovec.len(), peak);
    let observed: Vec<(f64, f64)> = histovec[..top]
        .iter()
        .enumerate()
        .filter(|(_, &n)| n > 0)
        .map(|(i, &n)| ((i + 1) as f64, f64::from(n)))
        .collect();
    if observed.is_empty() {
        return Err(Error::msg("no counts inside the fit window"));
    }
    let total = fitted_distinct_total(histovec, peak);

    let problem = MixtureFit {
        observed,
        peak: peak as f64,
    };

    let hint = if dispersion_hint.is_finite() && dispersion_hint > 1.0 {
        dispersion_hint
    } else {
        2.0
    };

    let mut best: Option<(f64, Vec<f64>)> = None;
    for seed in DISPERSION_SEEDS
        .iter()
        .copied()
        .chain(std::iter::once(hint))
    {
        let start = vec![0.0, 0.0, 0.0, (seed - 1.0).max(1e-3).ln(), 0.0];
        let Ok(res) = run_one(problem.clone(), start) else {
            continue;
        };
        if best.as_ref().is_none_or(|(cost, _)| res.0 < *cost) {
            best = Some(res);
        }
    }

    let Some((_, theta)) = best else {
        return Err(Error::msg("no start converged"));
    };
    Ok(problem.unpack(&theta, total))
}

/// One Nelder-Mead run. The simplex is the start point plus one offset coordinate each.
fn run_one(problem: MixtureFit, start: Vec<f64>) -> Result<(f64, Vec<f64>), Error> {
    let mut simplex = vec![start.clone()];
    for i in 0..start.len() {
        let mut v = start.clone();
        v[i] += SIMPLEX_STEP;
        simplex.push(v);
    }

    let res = Executor::new(problem, NelderMead::new(simplex))
        .configure(|state| state.max_iters(MAX_ITERS))
        .run()?;

    match res.state().get_termination_reason() {
        Some(reason) if *reason == SolverConverged => {
            let cost = res.state().get_best_cost();
            let param = res
                .state()
                .get_best_param()
                .ok_or_else(|| Error::msg("converged without a parameter vector"))?;
            Ok((cost, param.clone()))
        }
        _ => Err(Error::msg("did not converge")),
    }
}

/// The spectrum being fitted, and the peak the mean is pinned around.
#[derive(Clone)]
struct MixtureFit {
    observed: Vec<(f64, f64)>,
    peak: f64,
}

impl MixtureFit {
    /// Map the unconstrained parameters onto the model.
    ///
    /// Every constraint is structural rather than a penalty, so the optimiser cannot propose an
    /// invalid model and never meets a cliff. The error component is softmax's reference category:
    /// adding a constant to every weight leaves the mixture unchanged, so fixing one removes a flat
    /// ridge and a simplex dimension.
    fn params(&self, theta: &[f64]) -> Params {
        let (ln_error, ln_single, ln_repeat) = ln_softmax3(0.0, theta[0], theta[1]);
        Params {
            ln_error,
            ln_single,
            ln_repeat,
            mean: self.peak * (MU_LO + MU_SPAN / (1.0 + (-theta[2]).exp())),
            dispersion: 1.0 + theta[3].exp(),
            error_mean: theta[4].exp(),
        }
    }

    fn unpack(&self, theta: &[f64], total: f64) -> SpectrumFit {
        let p = self.params(theta);
        let w_single = p.ln_single.exp();
        SpectrumFit {
            w_error: p.ln_error.exp(),
            w_single,
            w_repeat: p.ln_repeat.exp(),
            mean: p.mean,
            dispersion: p.dispersion,
            error_mean: p.error_mean,
            genome_kmers: w_single * total,
        }
    }
}

/// The model on its natural scale, as the cost function reads it.
struct Params {
    ln_error: f64,
    ln_single: f64,
    ln_repeat: f64,
    mean: f64,
    dispersion: f64,
    error_mean: f64,
}

impl CostFunction for MixtureFit {
    type Param = Vec<f64>;
    type Output = f64;

    /// Negative log-likelihood of the spectrum under the mixture, each bin weighted by its height.
    fn cost(&self, theta: &Self::Param) -> Result<Self::Output, Error> {
        let p = self.params(theta);
        if !(p.mean.is_finite() && p.dispersion.is_finite() && p.error_mean.is_finite())
            || p.error_mean <= 0.0
        {
            return Ok(f64::INFINITY);
        }

        let mut ll = 0.0;
        for &(count, weight) in &self.observed {
            let a = p.ln_error + ln_dpois(count, p.error_mean);
            let b = p.ln_single + ln_dnbinom(count, p.mean, p.dispersion);
            let c = p.ln_repeat + ln_dnbinom(count, REPEAT_LOBE_COPIES * p.mean, p.dispersion);
            ll += weight * lse3(a, b, c);
        }
        Ok(-ll)
    }
}

/// Log of the three softmax weights, computed in log space so no exponential can overflow.
fn ln_softmax3(a: f64, b: f64, c: f64) -> (f64, f64, f64) {
    let norm = lse3(a, b, c);
    (a - norm, b - norm, c - norm)
}

/// Log-sum-exp of three terms, shifted by the largest so the exponentials stay in range.
fn lse3(a: f64, b: f64, c: f64) -> f64 {
    let m = a.max(b).max(c);
    if !m.is_finite() {
        return m;
    }
    m + ((a - m).exp() + (b - m).exp() + (c - m).exp()).ln()
}

/// Natural log of the Poisson density.
fn ln_dpois(x: f64, lambda: f64) -> f64 {
    x * lambda.ln() - lgamma(x + 1.0) - lambda
}

/// Natural log of the negative-binomial density, by mean and dispersion (`variance = dispersion *
/// mean`), so `r = mean / (dispersion - 1)` and `p = 1 / dispersion`. Tends to the Poisson as the
/// dispersion approaches 1, which is where the genome lobe of a clean simulated library sits.
fn ln_dnbinom(x: f64, mean: f64, dispersion: f64) -> f64 {
    if dispersion <= 1.0 + 1e-9 {
        return ln_dpois(x, mean);
    }
    let r = mean / (dispersion - 1.0);
    lgamma(x + r) - lgamma(x + 1.0) - lgamma(r)
        + r * (-dispersion.ln())
        + x * (1.0 - 1.0 / dispersion).ln()
}

// The browser deliberately keeps using the legacy Poisson-only fit above. Native models live
// separately so their likelihood, validation, and diagnostics cannot alter the WASM cutoff.
#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ErrorModel {
    FreeSingletonWeibull,
    SingletonPareto,
}

#[cfg(not(target_family = "wasm"))]
impl fmt::Display for ErrorModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::FreeSingletonWeibull => "free_singleton_weibull",
            Self::SingletonPareto => "singleton_pareto",
        })
    }
}

#[cfg(not(target_family = "wasm"))]
impl ErrorModel {
    const fn parameter_count(self) -> usize {
        match self {
            Self::FreeSingletonWeibull => 7,
            Self::SingletonPareto => 6,
        }
    }
}

#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GenomeModel {
    NegativeBinomial,
    Normal,
}

#[cfg(not(target_family = "wasm"))]
impl fmt::Display for GenomeModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NegativeBinomial => "negative_binomial",
            Self::Normal => "normal",
        })
    }
}

/// `c = scale * beta` keeps the conditional Weibull tail well-conditioned as beta approaches zero.
#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct ErrorParams {
    pub(crate) singleton_probability: f64,
    pub(crate) tail_exponent: f64,
    pub(crate) weibull_shape: Option<f64>,
}

#[cfg(not(target_family = "wasm"))]
impl ErrorParams {
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

    fn log_window_mass(self, top: usize) -> f64 {
        let log_beyond =
            (1.0 - self.singleton_probability).ln() + self.log_survival_from_two(top as f64 + 1.0);
        (-log_beyond.exp_m1()).ln()
    }
}

#[cfg(not(target_family = "wasm"))]
fn ln_sub_exp(high: f64, low: f64) -> f64 {
    if high == f64::NEG_INFINITY {
        return high;
    }
    if low > high || low.is_nan() {
        return f64::NAN;
    }
    high + (-(low - high).exp_m1()).ln()
}

#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FitRejection {
    OptimisationFailed,
    NonFiniteLikelihood,
    InvalidComponents,
    ErrorModePastValley,
    PrimaryModeOutsideBand,
    RepeatModeTooEarly,
}

#[cfg(not(target_family = "wasm"))]
impl fmt::Display for FitRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::OptimisationFailed => "optimisation_failed",
            Self::NonFiniteLikelihood => "non_finite_likelihood",
            Self::InvalidComponents => "invalid_components",
            Self::ErrorModePastValley => "error_mode_past_valley",
            Self::PrimaryModeOutsideBand => "primary_mode_outside_band",
            Self::RepeatModeTooEarly => "repeat_mode_too_early",
        })
    }
}

#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct NativeSpectrumFit {
    pub(crate) error_model: ErrorModel,
    pub(crate) genome_model: GenomeModel,
    pub(crate) w_error: f64,
    pub(crate) w_single: f64,
    pub(crate) w_repeat: f64,
    pub(crate) mean: f64,
    pub(crate) dispersion: f64,
    pub(crate) error_params: ErrorParams,
    pub(crate) genome_kmers: f64,
    pub(crate) error_kmers: f64,
    pub(crate) log_likelihood: f64,
    pub(crate) bic: f64,
    pub(crate) deviance: f64,
    pub(crate) fit_window_end: usize,
    pub(crate) best_iterations: u64,
    component_log_normalisers: [f64; 3],
    observed_total: f64,
}

#[cfg(not(target_family = "wasm"))]
impl NativeSpectrumFit {
    #[cfg(test)]
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
        let normalisers = [
            error_params.log_window_mass(fit_window_end),
            native_log_genome_mass(mean, dispersion, genome_model, fit_window_end),
            native_log_genome_mass(
                REPEAT_LOBE_COPIES * mean,
                dispersion,
                genome_model,
                fit_window_end,
            ),
        ];
        Self {
            error_model,
            genome_model,
            w_error: weights[0],
            w_single: weights[1],
            w_repeat: weights[2],
            mean,
            dispersion,
            error_params,
            genome_kmers: weights[1] * observed_total / normalisers[1].exp(),
            error_kmers: weights[0] * observed_total / normalisers[0].exp(),
            log_likelihood: -1.0,
            bic: 2.0 + error_model.parameter_count() as f64 * observed_total.ln(),
            deviance: 1.0,
            fit_window_end,
            best_iterations: 1,
            component_log_normalisers: normalisers,
            observed_total,
        }
    }

    pub(crate) fn primary_mode(&self) -> usize {
        native_genome_mode(self.mean, self.dispersion, self.genome_model)
    }

    pub(crate) fn repeat_mode(&self) -> usize {
        native_genome_mode(
            REPEAT_LOBE_COPIES * self.mean,
            self.dispersion,
            self.genome_model,
        )
    }

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
            native_ln_genome(count, self.mean, self.dispersion, self.genome_model),
            native_ln_genome(
                count,
                REPEAT_LOBE_COPIES * self.mean,
                self.dispersion,
                self.genome_model,
            ),
        ];
        [
            self.observed_total
                * self.w_error
                * (logs[0] - self.component_log_normalisers[0]).exp(),
            self.observed_total
                * self.w_single
                * (logs[1] - self.component_log_normalisers[1]).exp(),
            self.observed_total
                * self.w_repeat
                * (logs[2] - self.component_log_normalisers[2]).exp(),
        ]
    }

    pub(crate) fn hole_cutoff(&self, budget: f64) -> u16 {
        if self.genome_kmers <= 0.0 || !self.mean.is_finite() || self.mean < 2.0 {
            return 2;
        }
        let allowed = budget / self.genome_kmers;
        let ceiling = self.mean.round() as usize;
        let mut cdf = match self.genome_model {
            GenomeModel::NegativeBinomial => ln_dnbinom(0.0, self.mean, self.dispersion).exp(),
            GenomeModel::Normal => {
                normal_cdf((0.5 - self.mean) / (self.dispersion * self.mean).sqrt())
            }
        };
        let mut cutoff = 1usize;
        while cutoff < ceiling {
            let next = cdf
                + native_ln_genome(cutoff as f64, self.mean, self.dispersion, self.genome_model)
                    .exp();
            if next > allowed {
                break;
            }
            cdf = next;
            cutoff += 1;
        }
        (cutoff.min(u16::MAX as usize) as u16).max(2)
    }

    pub(crate) fn crossover(&self) -> u16 {
        let ceiling = self.mean.round().max(2.0) as usize;
        for count in 1..ceiling {
            let heights = self.component_heights(count);
            if heights[1] > heights[0] {
                return (count.min(u16::MAX as usize) as u16).max(2);
            }
        }
        (ceiling.min(u16::MAX as usize) as u16).max(2)
    }

    pub(crate) fn shadow_error_floor(&self, reference_fraction: f64) -> u16 {
        if !reference_fraction.is_finite() || reference_fraction <= 0.0 {
            return 2;
        }
        let reference = reference_fraction * self.component_heights(self.primary_mode())[1];
        let mut last_bad = 0usize;
        for count in 1..=self.fit_window_end {
            if self.component_heights(count)[0] >= reference {
                last_bad = count;
            }
        }
        last_bad.saturating_add(1).min(u16::MAX as usize).max(2) as u16
    }

    pub(crate) fn expected_error_between(&self, from: u16, to: u16) -> f64 {
        expected_between(from, to, self.error_kmers, |count| {
            self.error_params.log_probability(count)
        })
    }

    pub(crate) fn expected_genome_between(&self, from: u16, to: u16) -> f64 {
        expected_between(from, to, self.genome_kmers, |count| {
            native_ln_genome(count, self.mean, self.dispersion, self.genome_model)
        })
    }
}

#[cfg(not(target_family = "wasm"))]
fn expected_between<F>(from: u16, to: u16, population: f64, log_pmf: F) -> f64
where
    F: Fn(f64) -> f64,
{
    if to <= from || !population.is_finite() || population <= 0.0 {
        return 0.0;
    }
    population
        * (from..to)
            .map(|count| log_pmf(f64::from(count)).exp())
            .sum::<f64>()
}

#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct FitAttempt {
    pub(crate) error_model: ErrorModel,
    pub(crate) genome_model: GenomeModel,
    pub(crate) candidate: Option<NativeSpectrumFit>,
    pub(crate) rejection: Option<FitRejection>,
    pub(crate) total_iterations: u64,
    pub(crate) capped_starts: u32,
}

#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct FitSearchResult {
    pub(crate) selected: Option<NativeSpectrumFit>,
    pub(crate) attempts: [FitAttempt; 4],
}

#[cfg(not(target_family = "wasm"))]
#[derive(Clone)]
struct NativeMixtureFit {
    observed: Arc<[(f64, f64)]>,
    observed_total: f64,
    peak: f64,
    fit_window_end: usize,
    error_model: ErrorModel,
    genome_model: GenomeModel,
}

#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy)]
struct NativeParams {
    ln_error: f64,
    ln_single: f64,
    ln_repeat: f64,
    mean: f64,
    dispersion: f64,
    error_params: ErrorParams,
}

#[cfg(not(target_family = "wasm"))]
impl NativeMixtureFit {
    fn params(&self, theta: &[f64]) -> NativeParams {
        let (ln_error, ln_single, ln_repeat) = ln_softmax3(0.0, theta[0], theta[1]);
        let mode = self.peak * (MU_LO + MU_SPAN / (1.0 + (-theta[2]).exp()));
        let dispersion = 1.0 + theta[3].exp();
        let error_params = ErrorParams {
            tail_exponent: theta[4].exp(),
            singleton_probability: sigmoid(theta[5]),
            weibull_shape: match self.error_model {
                ErrorModel::FreeSingletonWeibull => Some(theta[6].exp()),
                ErrorModel::SingletonPareto => None,
            },
        };
        NativeParams {
            ln_error,
            ln_single,
            ln_repeat,
            mean: match self.genome_model {
                GenomeModel::NegativeBinomial => mode + dispersion - 1.0,
                GenomeModel::Normal => mode,
            },
            dispersion,
            error_params,
        }
    }

    fn log_normalisers(&self, params: NativeParams) -> [f64; 3] {
        [
            params.error_params.log_window_mass(self.fit_window_end),
            native_log_genome_mass(
                params.mean,
                params.dispersion,
                self.genome_model,
                self.fit_window_end,
            ),
            native_log_genome_mass(
                REPEAT_LOBE_COPIES * params.mean,
                params.dispersion,
                self.genome_model,
                self.fit_window_end,
            ),
        ]
    }

    fn unpack(&self, theta: &[f64], cost: f64, iterations: u64) -> NativeSpectrumFit {
        let params = self.params(theta);
        let normalisers = self.log_normalisers(params);
        let w_error = params.ln_error.exp();
        let w_single = params.ln_single.exp();
        let w_repeat = params.ln_repeat.exp();
        let mut fit = NativeSpectrumFit {
            error_model: self.error_model,
            genome_model: self.genome_model,
            w_error,
            w_single,
            w_repeat,
            mean: params.mean,
            dispersion: params.dispersion,
            error_params: params.error_params,
            genome_kmers: w_single * self.observed_total / normalisers[1].exp(),
            error_kmers: w_error * self.observed_total / normalisers[0].exp(),
            log_likelihood: -cost,
            bic: 2.0 * cost + self.error_model.parameter_count() as f64 * self.observed_total.ln(),
            deviance: 0.0,
            fit_window_end: self.fit_window_end,
            best_iterations: iterations,
            component_log_normalisers: normalisers,
            observed_total: self.observed_total,
        };
        fit.deviance = self.conditional_deviance(&fit);
        fit
    }

    fn conditional_deviance(&self, fit: &NativeSpectrumFit) -> f64 {
        self.observed
            .iter()
            .map(|&(count, observed)| {
                let expected = fit
                    .component_heights(count as usize)
                    .into_iter()
                    .sum::<f64>();
                if expected > 0.0 {
                    2.0 * observed * (observed / expected).ln()
                } else {
                    f64::INFINITY
                }
            })
            .sum()
    }
}

#[cfg(not(target_family = "wasm"))]
impl CostFunction for NativeMixtureFit {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, theta: &Self::Param) -> Result<Self::Output, Error> {
        let params = self.params(theta);
        if !(params.mean.is_finite()
            && params.dispersion.is_finite()
            && params.error_params.tail_exponent.is_finite()
            && params.error_params.singleton_probability.is_finite()
            && params.error_params.weibull_shape.is_none_or(f64::is_finite))
            || params.mean <= 0.0
            || params.error_params.tail_exponent <= 0.0
            || !(0.0..1.0).contains(&params.error_params.singleton_probability)
        {
            return Ok(f64::INFINITY);
        }
        let normalisers = self.log_normalisers(params);
        if normalisers.iter().any(|value| !value.is_finite()) {
            return Ok(f64::INFINITY);
        }

        let likelihood = self
            .observed
            .iter()
            .map(|&(count, weight)| {
                let error =
                    params.ln_error + params.error_params.log_probability(count) - normalisers[0];
                let single = params.ln_single
                    + native_ln_genome(count, params.mean, params.dispersion, self.genome_model)
                    - normalisers[1];
                let repeat = params.ln_repeat
                    + native_ln_genome(
                        count,
                        REPEAT_LOBE_COPIES * params.mean,
                        params.dispersion,
                        self.genome_model,
                    )
                    - normalisers[2];
                weight * lse3(error, single, repeat)
            })
            .sum::<f64>();
        Ok(if likelihood.is_finite() {
            -likelihood
        } else {
            f64::INFINITY
        })
    }
}

#[cfg(not(target_family = "wasm"))]
pub(crate) fn fit_native_spectrum(
    histovec: &[u32],
    valley: usize,
    peak: usize,
    dispersion_hint: f64,
) -> Result<FitSearchResult, Error> {
    if peak < 2 || valley < 2 || histovec.len() < 2 {
        return Err(Error::msg("no usable genome peak or valley to fit around"));
    }
    let top = fit_window_end(histovec.len(), peak);
    let observed: Arc<[(f64, f64)]> = histovec[..top]
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

    let run = |error_model, genome_model| {
        fit_native_candidate(
            NativeMixtureFit {
                observed: Arc::clone(&observed),
                observed_total,
                peak: peak as f64,
                fit_window_end: top,
                error_model,
                genome_model,
            },
            valley,
            peak,
            hint,
            NATIVE_MAX_ITERS,
        )
    };
    let ((pareto_nb, pareto_normal), (weibull_nb, weibull_normal)) = rayon::join(
        || {
            rayon::join(
                || run(ErrorModel::SingletonPareto, GenomeModel::NegativeBinomial),
                || run(ErrorModel::SingletonPareto, GenomeModel::Normal),
            )
        },
        || {
            rayon::join(
                || {
                    run(
                        ErrorModel::FreeSingletonWeibull,
                        GenomeModel::NegativeBinomial,
                    )
                },
                || run(ErrorModel::FreeSingletonWeibull, GenomeModel::Normal),
            )
        },
    );
    let attempts = [pareto_nb, pareto_normal, weibull_nb, weibull_normal];
    let selected = select_native_fit(&attempts);
    Ok(FitSearchResult { selected, attempts })
}

#[cfg(not(target_family = "wasm"))]
fn fit_native_candidate(
    problem: NativeMixtureFit,
    valley: usize,
    empirical_peak: usize,
    dispersion_hint: f64,
    max_iters: u64,
) -> FitAttempt {
    let mut best: Option<(f64, Vec<f64>, u64)> = None;
    let mut total_iterations = 0u64;
    let mut capped_starts = 0u32;
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
            let Ok(outcome) = run_native_one(problem.clone(), start, max_iters) else {
                continue;
            };
            total_iterations = total_iterations.saturating_add(outcome.iterations);
            capped_starts += u32::from(outcome.hit_cap);
            let Some((cost, params)) = outcome.converged else {
                continue;
            };
            if best
                .as_ref()
                .is_none_or(|(best_cost, _, _)| cost < *best_cost)
            {
                best = Some((cost, params, outcome.iterations));
            }
        }
    }

    let Some((cost, params, best_iterations)) = best else {
        return FitAttempt {
            error_model: problem.error_model,
            genome_model: problem.genome_model,
            candidate: None,
            rejection: Some(FitRejection::OptimisationFailed),
            total_iterations,
            capped_starts,
        };
    };
    let fit = problem.unpack(&params, cost, best_iterations);
    let rejection = validate_native_fit(&fit, valley, empirical_peak);
    FitAttempt {
        error_model: problem.error_model,
        genome_model: problem.genome_model,
        candidate: Some(fit),
        rejection,
        total_iterations,
        capped_starts,
    }
}

#[cfg(not(target_family = "wasm"))]
struct NativeRunOutcome {
    converged: Option<(f64, Vec<f64>)>,
    iterations: u64,
    hit_cap: bool,
}

#[cfg(not(target_family = "wasm"))]
fn run_native_one(
    problem: NativeMixtureFit,
    start: Vec<f64>,
    max_iters: u64,
) -> Result<NativeRunOutcome, Error> {
    let mut simplex = vec![start.clone()];
    for index in 0..start.len() {
        let mut vertex = start.clone();
        vertex[index] += SIMPLEX_STEP;
        simplex.push(vertex);
    }
    let solver = NelderMead::new(simplex).with_sd_tolerance(NATIVE_SD_TOLERANCE)?;
    let result = Executor::new(problem, solver)
        .configure(|state| state.max_iters(max_iters))
        .run()?;
    let state = result.state();
    let termination = state.get_termination_reason();
    let converged = if matches!(termination, Some(reason) if *reason == SolverConverged)
        && state.get_best_cost().is_finite()
    {
        state
            .get_best_param()
            .map(|params| (state.get_best_cost(), params.clone()))
    } else {
        None
    };
    Ok(NativeRunOutcome {
        converged,
        iterations: state.get_iter(),
        hit_cap: matches!(termination, Some(reason) if *reason == MaxItersReached),
    })
}

#[cfg(not(target_family = "wasm"))]
fn validate_native_fit(
    fit: &NativeSpectrumFit,
    valley: usize,
    empirical_peak: usize,
) -> Option<FitRejection> {
    if !fit.log_likelihood.is_finite() || !fit.deviance.is_finite() || !fit.bic.is_finite() {
        return Some(FitRejection::NonFiniteLikelihood);
    }
    let weight_sum = fit.w_error + fit.w_single + fit.w_repeat;
    if !weight_sum.is_finite()
        || (weight_sum - 1.0).abs() > 1e-8
        || fit.component_log_normalisers.iter().any(|x| !x.is_finite())
        || !fit.genome_kmers.is_finite()
        || !fit.error_kmers.is_finite()
    {
        return Some(FitRejection::InvalidComponents);
    }
    if fit.error_mode() >= valley {
        return Some(FitRejection::ErrorModePastValley);
    }
    let mode = fit.primary_mode() as f64;
    let peak = empirical_peak as f64;
    if mode <= 0.0 || (mode / peak).max(peak / mode) > 1.25 {
        return Some(FitRejection::PrimaryModeOutsideBand);
    }
    if (fit.repeat_mode() as f64) < 1.4 * mode {
        return Some(FitRejection::RepeatModeTooEarly);
    }
    None
}

#[cfg(not(target_family = "wasm"))]
fn select_native_fit(attempts: &[FitAttempt; 4]) -> Option<NativeSpectrumFit> {
    let mut selected: Option<NativeSpectrumFit> = None;
    // Normal lobes are useful fit diagnostics, but their unconstrained left tail can imply more
    // than the one-k-mer hole budget even below count 1, forcing the cutoff down to 2. Keep
    // their attempts and logs, but never let them determine the cutoff.
    for attempt in attempts
        .iter()
        .filter(|attempt| attempt.genome_model == GenomeModel::NegativeBinomial)
    {
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

#[cfg(not(target_family = "wasm"))]
fn discrete_nb_mode(mean: f64, dispersion: f64) -> usize {
    let mode = mean - (dispersion - 1.0);
    if mode.is_finite() && mode > 0.0 {
        mode.floor().min(usize::MAX as f64) as usize
    } else {
        0
    }
}

#[cfg(not(target_family = "wasm"))]
fn native_genome_mode(mean: f64, dispersion: f64, model: GenomeModel) -> usize {
    match model {
        GenomeModel::NegativeBinomial => discrete_nb_mode(mean, dispersion),
        GenomeModel::Normal => (mean + 0.5).floor().max(0.0) as usize,
    }
}

#[cfg(not(target_family = "wasm"))]
fn native_ln_genome(count: f64, mean: f64, dispersion: f64, model: GenomeModel) -> f64 {
    match model {
        GenomeModel::NegativeBinomial => ln_dnbinom(count, mean, dispersion),
        GenomeModel::Normal => {
            ln_normal_between(count - 0.5, count + 0.5, mean, (dispersion * mean).sqrt())
        }
    }
}

#[cfg(not(target_family = "wasm"))]
fn native_log_genome_mass(mean: f64, dispersion: f64, model: GenomeModel, top: usize) -> f64 {
    match model {
        GenomeModel::NegativeBinomial => native_log_nb_mass(mean, dispersion, top),
        GenomeModel::Normal => {
            ln_normal_between(0.5, top as f64 + 0.5, mean, (dispersion * mean).sqrt())
        }
    }
}

#[cfg(not(target_family = "wasm"))]
fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let scaled = value.exp();
        scaled / (1.0 + scaled)
    }
}

#[cfg(not(target_family = "wasm"))]
fn normal_cdf(value: f64) -> f64 {
    0.5 * libm::erfc(-value / std::f64::consts::SQRT_2)
}

/// Log(erfc(z)) for z >= 0, including the range where libm's direct result underflows.
#[cfg(not(target_family = "wasm"))]
fn ln_erfc_positive(value: f64) -> f64 {
    if value < 25.0 {
        libm::erfc(value).ln()
    } else {
        let inverse_square = 1.0 / (value * value);
        -value * value - value.ln() - 0.5 * std::f64::consts::PI.ln()
            + (1.0 - 0.5 * inverse_square + 0.75 * inverse_square * inverse_square).ln()
    }
}

/// Probability that a continuous normal lands in [lower, upper], evaluated in log space.
#[cfg(not(target_family = "wasm"))]
fn ln_normal_between(lower: f64, upper: f64, mean: f64, sigma: f64) -> f64 {
    if !sigma.is_finite() || sigma <= 0.0 || upper <= lower {
        return f64::NAN;
    }
    let lower_z = (lower - mean) / (sigma * std::f64::consts::SQRT_2);
    let upper_z = (upper - mean) / (sigma * std::f64::consts::SQRT_2);
    if upper_z <= 0.0 {
        ln_sub_exp(ln_erfc_positive(-upper_z), ln_erfc_positive(-lower_z)) - std::f64::consts::LN_2
    } else if lower_z >= 0.0 {
        ln_sub_exp(ln_erfc_positive(lower_z), ln_erfc_positive(upper_z)) - std::f64::consts::LN_2
    } else {
        (0.5 * (libm::erf(upper_z) - libm::erf(lower_z))).ln()
    }
}

#[cfg(not(target_family = "wasm"))]
fn native_log_nb_mass(mean: f64, dispersion: f64, top: usize) -> f64 {
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

#[cfg(not(target_family = "wasm"))]
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

#[cfg(not(target_family = "wasm"))]
fn lse2(a: f64, b: f64) -> f64 {
    let maximum = a.max(b);
    if !maximum.is_finite() {
        return maximum;
    }
    maximum + ((a - maximum).exp() + (b - maximum).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A spectrum drawn from the model: Poisson errors, then genome lobes at `mu` and `2 mu`.
    fn synthetic(
        mean: f64,
        dispersion: f64,
        genome: f64,
        err_mass: f64,
        repeat_frac: f64,
    ) -> Vec<u32> {
        let mut h = vec![0u32; 8000];
        for (i, slot) in h.iter_mut().enumerate() {
            let c = (i + 1) as f64;
            let single = genome * (1.0 - repeat_frac) * ln_dnbinom(c, mean, dispersion).exp();
            let repeat = genome * repeat_frac * ln_dnbinom(c, 2.0 * mean, dispersion).exp();
            let err = genome * err_mass * ln_dpois(c, 1.0).exp();
            *slot = (single + repeat + err) as u32;
        }
        h
    }

    #[test]
    fn fit_recovers_a_known_mixture() {
        let h = synthetic(95.0, 6.6, 4.6e6, 3.0, 0.03);
        let f = fit_spectrum(&h, 95, 6.6).expect("should converge");
        assert!(
            (f.mean - 95.0).abs() < 5.0 && (f.dispersion - 6.6).abs() < 1.5,
            "genome lobe: mean {} (want 95), dispersion {} (want 6.6)",
            f.mean,
            f.dispersion
        );
        // The error lobe absorbs part of the genome lobe's left tail, which biases its mean upward;
        // the wider the genome lobe, the more so. Real spectra measured 1.00-1.15, this fixture 1.6.
        assert!(
            (1.0..2.0).contains(&f.error_mean),
            "error mean {}",
            f.error_mean
        );
        let expected = 4.6e6 * 0.97;
        assert!(
            (f.genome_kmers - expected).abs() / expected < 0.2,
            "genome k-mers {}",
            f.genome_kmers
        );
    }

    /// The failure that made multi-start mandatory: seeded at the true dispersion alone, the simplex
    /// can walk to the Poisson boundary and stop. The multi-start must not end up there.
    #[test]
    fn multi_start_escapes_the_dispersion_boundary() {
        let h = synthetic(95.0, 6.6, 4.6e6, 3.0, 0.03);
        let f = fit_spectrum(&h, 95, 6.6).expect("should converge");
        assert!(
            f.dispersion > 1.5,
            "collapsed to the Poisson boundary: {}",
            f.dispersion
        );
    }

    /// Unpinned, the model relabels the peak as a repeat lobe and halves the mean. The window must
    /// hold it at the observed peak.
    #[test]
    fn a_constrained_mean_cannot_relabel_the_peak() {
        let h = synthetic(95.0, 6.6, 4.6e6, 3.0, 0.10);
        let f = fit_spectrum(&h, 95, 6.6).expect("should converge");
        assert!(
            f.mean >= 95.0 * MU_LO && f.mean <= 95.0 * (MU_LO + MU_SPAN),
            "mean {} escaped the window around the peak",
            f.mean
        );
    }

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

    /// The k=81 simulation case: shallow coverage, lobes nearly merged, dispersion near the boundary.
    #[test]
    fn merged_lobes_still_recover_the_genome_mean() {
        let h = synthetic(19.0, 1.2, 4.6e6, 2.0, 0.02);
        let f = fit_spectrum(&h, 19, 1.2).expect("should converge");
        assert!((f.mean - 19.0).abs() < 2.0, "mean {}", f.mean);
    }

    #[test]
    fn a_degenerate_spectrum_does_not_converge() {
        assert!(fit_spectrum(&vec![0u32; 8000], 95, 6.6).is_err());
        assert!(fit_spectrum(&synthetic(95.0, 6.6, 4.6e6, 3.0, 0.03), 1, 6.6).is_err());
    }

    #[test]
    fn the_fit_window_and_its_total_use_the_same_bins() {
        let mut h = vec![0u32; 8000];
        h[0] = 3;
        h[2477] = 5;
        h[2478] = 11;
        assert_eq!(fit_window_end(h.len(), 413), 2478);
        assert_eq!(fitted_distinct_total(&h, 413), 8.0);
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn fitted_mode_uses_the_negative_binomial_not_the_mean() {
        let fit = SpectrumFit {
            w_error: 0.877,
            w_single: 0.123,
            w_repeat: 0.0,
            mean: 417.5,
            dispersion: 139.22,
            error_mean: 1.49,
            genome_kmers: 2.184e6,
        };
        assert_eq!(fit.single_copy_mode(), 279);

        let probabilities = fit.component_probabilities(413.0);
        assert!(probabilities.iter().all(|p| p.is_finite() && *p >= 0.0));
        let mixture_probability = probabilities.iter().sum::<f64>();
        assert!(mixture_probability > 0.0 && mixture_probability <= 1.0);
        assert!(fit.component_probabilities(279.0)[1] >= fit.component_probabilities(278.0)[1]);
        assert!(fit.component_probabilities(279.0)[1] >= fit.component_probabilities(280.0)[1]);
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn fitted_mode_handles_the_poisson_boundary_and_zero_mode() {
        let mut fit = SpectrumFit {
            w_error: 0.0,
            w_single: 1.0,
            w_repeat: 0.0,
            mean: 95.0,
            dispersion: 1.0,
            error_mean: 1.0,
            genome_kmers: 1.0,
        };
        assert_eq!(fit.single_copy_mode(), 95);

        fit.mean = 5.0;
        fit.dispersion = 8.0;
        assert_eq!(fit.single_copy_mode(), 0);

        fit.mean = 20.0;
        fit.dispersion = 6.0;
        assert_eq!(
            fit.single_copy_mode(),
            15,
            "the upper tied mode is reported"
        );
    }

    /// The cutoff bounds holes, so it can only loosen as the budget grows, and never falls below 2.
    #[test]
    fn the_hole_cutoff_is_monotone_in_the_budget() {
        let f = SpectrumFit {
            w_error: 0.7,
            w_single: 0.29,
            w_repeat: 0.01,
            mean: 95.0,
            dispersion: 6.6,
            error_mean: 1.0,
            genome_kmers: 4.6e6,
        };
        let mut last = 0u16;
        for budget in [0.5, 1.0, 5.0, 50.0] {
            let c = f.hole_cutoff(budget);
            assert!(c >= last, "budget {budget} tightened the cutoff");
            assert!(c >= 2);
            last = c;
        }
    }

    #[cfg(not(target_family = "wasm"))]
    fn native_synthetic(error: ErrorParams, genome_model: GenomeModel) -> Vec<u32> {
        let mut histogram = vec![0u32; 601];
        let mean = if genome_model == GenomeModel::NegativeBinomial {
            53.0
        } else {
            50.0
        };
        for (index, bin) in histogram.iter_mut().enumerate() {
            let count = (index + 1) as f64;
            let error_height = 400_000.0 * error.log_probability(count).exp();
            let single = 200_000.0 * native_ln_genome(count, mean, 4.0, genome_model).exp();
            let repeat = 15_000.0 * native_ln_genome(count, 2.0 * mean, 4.0, genome_model).exp();
            *bin = (error_height + single + repeat).round() as u32;
        }
        histogram
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn capped_native_starts_count_iterations_but_are_not_candidates() {
        let histogram = native_synthetic(
            ErrorParams {
                singleton_probability: 0.85,
                tail_exponent: 1.8,
                weibull_shape: None,
            },
            GenomeModel::NegativeBinomial,
        );
        let observed: Arc<[(f64, f64)]> = histogram[..300]
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(index, &count)| ((index + 1) as f64, f64::from(count)))
            .collect::<Vec<_>>()
            .into();
        let problem = NativeMixtureFit {
            observed_total: observed.iter().map(|&(_, count)| count).sum(),
            observed,
            peak: 50.0,
            fit_window_end: 300,
            error_model: ErrorModel::SingletonPareto,
            genome_model: GenomeModel::NegativeBinomial,
        };
        let attempt = fit_native_candidate(problem, 12, 50, 4.0, 1);
        assert_eq!(attempt.capped_starts, 8);
        assert_eq!(attempt.total_iterations, 8);
        assert!(attempt.candidate.is_none());
        assert_eq!(attempt.rejection, Some(FitRejection::OptimisationFailed));
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
    fn discretised_normal_mass_and_left_tail_agree() {
        let mean = 50.0_f64;
        let dispersion = 4.0_f64;
        let sigma = (mean * dispersion).sqrt();
        let top = 300;
        let summed = (1..=top)
            .map(|count| {
                native_ln_genome(count as f64, mean, dispersion, GenomeModel::Normal).exp()
            })
            .sum::<f64>();
        let analytic = native_log_genome_mass(mean, dispersion, GenomeModel::Normal, top).exp();
        assert!((summed - analytic).abs() < 1e-10);
        let zero = normal_cdf((0.5 - mean) / sigma);
        assert!((zero + summed - 1.0).abs() < 1e-6);
        for count in [1.0, 50.0, 120.0, 300.0, 1000.0] {
            assert!(native_ln_genome(count, mean, dispersion, GenomeModel::Normal).is_finite());
        }
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn normal_hole_cutoff_counts_the_unobserved_left_tail() {
        let fit = NativeSpectrumFit::for_test(
            ErrorModel::SingletonPareto,
            GenomeModel::Normal,
            [0.7, 0.29, 0.01],
            20.0,
            4.0,
            ErrorParams {
                singleton_probability: 0.85,
                tail_exponent: 2.0,
                weibull_shape: None,
            },
            1.0e6,
            120,
        );
        let allowed = 0.1;
        let cutoff = fit.hole_cutoff(allowed * fit.genome_kmers);
        let sigma = (fit.dispersion * fit.mean).sqrt();
        let below_cutoff = normal_cdf((f64::from(cutoff) - 0.5 - fit.mean) / sigma);
        let including_cutoff = normal_cdf((f64::from(cutoff) + 0.5 - fit.mean) / sigma);
        assert!(cutoff > 2);
        assert!(below_cutoff <= allowed);
        assert!(including_cutoff > allowed);
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn native_fit_recovers_the_pareto_tail_with_both_genomic_families() {
        for genome_model in [GenomeModel::NegativeBinomial, GenomeModel::Normal] {
            let error = ErrorParams {
                singleton_probability: 0.85,
                tail_exponent: 1.8,
                weibull_shape: None,
            };
            let result = fit_native_spectrum(&native_synthetic(error, genome_model), 12, 50, 4.0)
                .expect("synthetic fit should run");
            assert_eq!(result.attempts.len(), 4);
            assert_eq!(
                result
                    .attempts
                    .iter()
                    .filter(|attempt| attempt.genome_model == GenomeModel::Normal)
                    .count(),
                2
            );
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
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn native_fit_recovers_the_weibull_tail_with_both_genomic_families() {
        for genome_model in [GenomeModel::NegativeBinomial, GenomeModel::Normal] {
            let error = ErrorParams {
                singleton_probability: 0.78,
                tail_exponent: 1.5,
                weibull_shape: Some(0.8),
            };
            let result = fit_native_spectrum(&native_synthetic(error, genome_model), 12, 50, 4.0)
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
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn bic_ties_prefer_pareto_and_normal_fits_are_diagnostic_only() {
        let error = ErrorParams {
            singleton_probability: 0.85,
            tail_exponent: 2.0,
            weibull_shape: None,
        };
        let accepted = |fit: NativeSpectrumFit| FitAttempt {
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
            let mut fit = NativeSpectrumFit::for_test(
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
            make(ErrorModel::SingletonPareto, GenomeModel::Normal),
            make(
                ErrorModel::FreeSingletonWeibull,
                GenomeModel::NegativeBinomial,
            ),
            make(ErrorModel::FreeSingletonWeibull, GenomeModel::Normal),
        ];
        let selected = select_native_fit(&attempts).expect("a fit should be selected");
        assert_eq!(
            (selected.error_model, selected.genome_model),
            (ErrorModel::SingletonPareto, GenomeModel::NegativeBinomial)
        );
        attempts[1].candidate.as_mut().unwrap().bic -= 1.0e6;
        attempts[3].candidate.as_mut().unwrap().bic -= 1.0e6;
        let selected = select_native_fit(&attempts).expect("a fit should be selected");
        assert_eq!(
            (selected.error_model, selected.genome_model),
            (ErrorModel::SingletonPareto, GenomeModel::NegativeBinomial)
        );
        attempts[2].candidate.as_mut().unwrap().bic -= 2.0;
        let selected = select_native_fit(&attempts).expect("a fit should be selected");
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
        let fit = NativeSpectrumFit::for_test(
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
            validate_native_fit(&fit, 12, 413),
            Some(FitRejection::PrimaryModeOutsideBand)
        );
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn primary_mode_ratio_boundary_is_inclusive() {
        let at_boundary = NativeSpectrumFit::for_test(
            ErrorModel::SingletonPareto,
            GenomeModel::NegativeBinomial,
            [0.7, 0.29, 0.01],
            130.0,
            6.0,
            ErrorParams {
                singleton_probability: 0.85,
                tail_exponent: 2.0,
                weibull_shape: None,
            },
            10.0e6,
            600,
        );
        assert_eq!(at_boundary.primary_mode(), 125);
        assert_eq!(validate_native_fit(&at_boundary, 12, 100), None);

        let outside = NativeSpectrumFit::for_test(
            ErrorModel::SingletonPareto,
            GenomeModel::NegativeBinomial,
            [0.7, 0.29, 0.01],
            131.0,
            6.0,
            ErrorParams {
                singleton_probability: 0.85,
                tail_exponent: 2.0,
                weibull_shape: None,
            },
            10.0e6,
            600,
        );
        assert_eq!(outside.primary_mode(), 126);
        assert_eq!(
            validate_native_fit(&outside, 12, 100),
            Some(FitRejection::PrimaryModeOutsideBand)
        );
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn native_components_are_conditioned_on_the_fit_window() {
        for model in [
            ErrorModel::SingletonPareto,
            ErrorModel::FreeSingletonWeibull,
        ] {
            for genome_model in [GenomeModel::NegativeBinomial, GenomeModel::Normal] {
                let error = ErrorParams {
                    singleton_probability: 0.85,
                    tail_exponent: 2.0,
                    weibull_shape: (model == ErrorModel::FreeSingletonWeibull).then_some(0.5),
                };
                let fit = NativeSpectrumFit::for_test(
                    model,
                    genome_model,
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
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn shadow_floors_are_monotone_and_exclude_the_last_bad_count() {
        for model in [
            ErrorModel::SingletonPareto,
            ErrorModel::FreeSingletonWeibull,
        ] {
            let error = ErrorParams {
                singleton_probability: 0.85,
                tail_exponent: 2.0,
                weibull_shape: (model == ErrorModel::FreeSingletonWeibull).then_some(0.5),
            };
            let fit = NativeSpectrumFit::for_test(
                model,
                GenomeModel::NegativeBinomial,
                [0.8, 0.19, 0.01],
                100.0,
                6.0,
                error,
                10.0e6,
                570,
            );
            let floors = [0.25, 0.5, 0.75, 1.0].map(|x| fit.shadow_error_floor(x));
            assert!(floors.windows(2).all(|pair| pair[0] >= pair[1]));
            for (fraction, floor) in [0.25, 0.5, 0.75, 1.0].into_iter().zip(floors) {
                if floor > 2 {
                    let reference = fraction * fit.component_heights(fit.primary_mode())[1];
                    assert!(fit.component_heights(usize::from(floor - 1))[0] >= reference);
                    if usize::from(floor) <= fit.fit_window_end {
                        assert!(fit.component_heights(usize::from(floor))[0] < reference);
                    }
                }
            }
        }
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn no_accepted_candidate_produces_no_selection() {
        let rejected = |error_model, genome_model| FitAttempt {
            error_model,
            genome_model,
            candidate: None,
            rejection: Some(FitRejection::OptimisationFailed),
            total_iterations: 0,
            capped_starts: 0,
        };
        assert!(select_native_fit(&[
            rejected(ErrorModel::SingletonPareto, GenomeModel::NegativeBinomial),
            rejected(ErrorModel::SingletonPareto, GenomeModel::Normal),
            rejected(
                ErrorModel::FreeSingletonWeibull,
                GenomeModel::NegativeBinomial
            ),
            rejected(ErrorModel::FreeSingletonWeibull, GenomeModel::Normal),
        ])
        .is_none());
    }
}
