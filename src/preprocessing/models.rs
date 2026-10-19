//! Coverage models, optimiser parameters, and fitted mixture evaluation.

use super::math_helpers::*;
use argmin::core::{CostFunction, Error};
use std::{fmt, sync::Arc};

/// The fitted single-copy mode is held within peak × [MU_LO, MU_LO + MU_SPAN], 0.75 to 1.33.
pub(super) const MU_LO: f64 = 0.75;

/// Width of that band, as a multiple of the peak.
pub(super) const MU_SPAN: f64 = 0.58;

/// Ratio we allow the second lobe (repeats lobe) to float in
pub(super) const REPEAT_RATIO: (f64, f64) = (1.5, 2.0);

/// The shelf never holds more than this share of single-copy k-mers, so it cannot swallow the lobe.
pub(super) const SHELF_MAX_SHARE: f64 = 0.5;

/// The shelf's k-mers keep at least this fraction of their coverage on average: below it the shelf
/// sat on the error tail, at counts 0-1, in the 2026_09_28 sweep.
pub(super) const SHELF_MIN_RETAINED: f64 = 0.3;

/// Upper bound on the shelf's a + b: above it the shelf is a spike.
pub(super) const SHELF_MAX_CONCENTRATION: f64 = 20.0;

/// A log window mass is a probability and therefore cannot exceed one, apart from round-off.
pub(super) const WINDOW_LOG_MASS_TOLERANCE: f64 = 1e-12;

/// Require the NB mode encoded by mean and dispersion to retain half-count precision.
pub(super) const NB_MODE_ROUNDTRIP_TOLERANCE: f64 = 0.5;

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
    pub(super) const fn parameter_count(self) -> usize {
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
    pub(super) fn log_probability(self, count: f64) -> f64 {
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
    pub(super) fn log_window_mass(self, top: usize) -> f64 {
        let log_beyond =
            (1.0 - self.singleton_probability).ln() + self.log_survival_from_two(top as f64 + 1.0);
        (-log_beyond.exp_m1()).ln()
    }
}

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
    pub(super) const fn extra_parameters(self) -> usize {
        match self {
            Self::ShelfNegativeBinomial => 4,
            Self::NegativeBinomial => 1,
        }
    }
}

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
pub(super) fn shelf_concentration_bounds(retained: f64) -> (f64, f64) {
    (1.0 / retained, SHELF_MAX_CONCENTRATION)
}

/// A share inside (0, [`SHELF_MAX_SHARE`]), and finite shapes with a >= 1 within the retained-fraction
/// and concentration bounds.
pub(super) fn shelf_is_valid(shelf: Shelf) -> bool {
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
    pub(super) target_mode: f64,
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
    pub(super) component_log_totals: [f64; 3],
    /// Distinct k-mers inside the fit window, at library scale.
    pub(super) observed_total: f64,
    /// Distinct k-mers in the whole spectrum, at library scale.
    pub(super) full_distinct_total: f64,
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

/// One simplex point decoded into the model's parameters.
#[derive(Clone, Copy)]
pub(super) struct Params {
    /// Log mixture weights.
    pub(super) ln_error: f64,
    pub(super) ln_single: f64,
    pub(super) ln_repeat: f64,
    /// The single-copy lobe's mode, the coordinate the simplex moves.
    pub(super) target_mode: f64,
    /// The single-copy lobe's mean, from the mode and the dispersion.
    pub(super) mean: f64,
    pub(super) dispersion: f64,
    pub(super) error_params: ErrorParams,
    pub(super) shelf: Option<Shelf>,
    pub(super) repeat_ratio: f64,
}

/// The problem argmin minimises: one configuration over the fit window.
#[derive(Clone)]
pub(super) struct MixtureFit {
    /// (count, distinct k-mers) of every non-empty bin in the window, at sample scale.
    pub(super) observed: Arc<[(f64, f64)]>,
    /// Their total.
    pub(super) observed_total: f64,
    /// Distinct k-mers in the whole spectrum, at library scale.
    pub(super) full_distinct_total: f64,
    /// The empirical single-copy peak the mode is held around.
    pub(super) peak: f64,
    /// Last count fitted.
    pub(super) fit_window_end: usize,
    pub(super) error_model: ErrorModel,
    pub(super) genome_model: GenomeModel,
    /// Library k-mers each observed one stands for: 1/fraction for a sketch, 1 for an exact table.
    pub(super) sample_scale: f64,
}

impl MixtureFit {
    /// Decode θ: softmax weights with the error's logit fixed at 0, and sigmoid or exponential maps
    /// into each parameter's range.
    pub(super) fn params(&self, theta: &[f64]) -> Params {
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
    pub(super) fn unpack(&self, theta: &[f64], cost: f64, iterations: u64) -> SpectrumFit {
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

#[cfg(test)]
/// Genome lobes modelled: single-copy and two-copy
pub(super) const REPEAT_LOBE_COPIES: f64 = 2.0;

#[cfg(test)]
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
