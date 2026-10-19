//! Fit coverage models to a k-mer spectrum, validate candidates, and select the best fit.

use super::{math_helpers::*, models::*};
use argmin::{
    core::{
        Error, Executor, State,
        TerminationReason::{MaxItersReached, SolverConverged},
    },
    solver::neldermead::NelderMead,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{fmt, sync::Arc};

/// Counts above this multiple of the peak are not fitted: conservative threshold.
const FIT_WINDOW_PEAK_MULT: usize = 6;

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

/// Where the ratio starts: just under a double, so the simplex has room on both sides.
const REPEAT_RATIO_START: f64 = 1.9;

/// Starting (retained fraction, concentration): a broad bump, and one ramping up to the lobe. Both sit
/// inside the bounds, so the simplex can move either way.
const SHELF_SHAPES: [(f64, f64); 2] = [(0.5, 3.0), (0.75, 4.0)];

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

/// Allow modest fit/sketch error, but not component totals larger than the whole spectrum.
const COMPONENT_MASS_MAX_RATIO: f64 = 1.10;

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

#[cfg(test)]
mod tests {
    use super::*;
    use argmin::core::CostFunction;

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

    /// The shelf model with a Pareto tail, over the NB-only synthetic spectrum's first 300 counts.
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

    /// A spectrum with a tenth of its single-copy k-mers in a uniform shelf selects the shelf model,
    /// and the hole guard then counts the shelf's low counts as genome.
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

    /// One candidate's starts run in turn, as the whole search once did.
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
