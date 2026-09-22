//! Fits a coverage model to the k-mer spectrum, to choose a count cutoff.
//!
//! Three components: a Poisson error lobe, and negative-binomial genome lobes at single-copy and
//! two-copy coverage sharing one mean and one dispersion. Fitted by maximum likelihood.
//!
//! ====================================================== Got from ska-rust!! =====
//!
//! [`fit_spectrum`] is the main interface.

use argmin::{
    core::{CostFunction, Error, Executor, State, TerminationReason::SolverConverged},
    solver::neldermead::NelderMead,
};
use libm::lgamma;

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
/// Offset applied to one coordinate at a time to build the initial simplex.
const SIMPLEX_STEP: f64 = 0.5;

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
    for seed in DISPERSION_SEEDS.iter().copied().chain(std::iter::once(hint)) {
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
}
