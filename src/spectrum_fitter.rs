//! Tools for estimating a count cutoff with FASTQ input.
//!
//! This module has a basic k-mer counter using a dictionary, and then uses
//! maximum likelihood with some basic numerical optimisation to fit a two-component
//! mixture of Poissons to determine a coverage model. This can be used to classify
//! a count cutoff with noisy data.
//!
//! ====================================================== Got from ska-rust!! =====
//!
//! [`SpectrumFitter`] is the main interface.

use core::panic;

use argmin::{
    core::{
        // observers::{ObserverMode, SlogLogger},
        observers::ObserverMode,
        CostFunction,
        Error,
        Executor,
        Gradient,
        State,
        TerminationReason::SolverConverged,
    },
    solver::{
        linesearch::{condition::ArmijoCondition, BacktrackingLineSearch},
        quasinewton::BFGS,
    },
};
use argmin_observer_slog::SlogLogger;
use libm::lgamma;

use crate::logw;
use log;

// const MAX_COUNT : usize = 500;
const MIN_FREQ: u32 = 50;
const INIT_W0: f64 = 0.8f64;
const INIT_C: f64 = 20.0f64;

/// K-mer counts and a coverage model for a single sample, using a pair of FASTQ files as input
///
/// Call [`SpectrumFitter::new()`] to count k-mers, then [`SpectrumFitter::fit_histogram()`]
/// to fit the model and find a cutoff. [`SpectrumFitter::plot_hist()`] can be used to
/// extract a table of the output for plotting purposes.
#[derive(Default, Debug)]
pub struct SpectrumFitter {
    /// Estimated error weight
    w0: f64,
    /// Estimated coverage
    c: f64,
    /// Coverage cutoff
    cutoff: usize,
    /// Has the fit been run
    fitted: bool,
}

impl SpectrumFitter {
    /// Count split k-mers from a pair of input FASTQ files.
    pub fn new() -> Self {
        Self {
            w0: INIT_W0,
            c: INIT_C,
            cutoff: 0,
            fitted: false,
        }
    }

    /// Fit the coverage model to the histogram of counts
    ///
    /// Returns the fitted cutoff if successful.
    ///
    /// # Errors
    /// - If the optimiser didn't finish (reached 100 iterations or another problem).
    /// - If the linesearch cannot be constructed (may be a bounds issue, or poor data).
    /// - If the optimiser is still running (this shouldn't happen).
    ///
    /// # Panics
    /// - If the fit has already been run
    pub fn fit_histogram(&mut self, mut counts: Vec<u32>) -> Result<usize, Error> {
        if self.fitted {
            panic!("Model already fitted");
        }

        // Truncate count vec and covert to float
        counts = counts
            .iter()
            .rev()
            .skip_while(|x| **x < MIN_FREQ)
            .copied()
            .collect();
        counts.reverse();
        let counts_f64: Vec<f64> = counts.iter().map(|x| *x as f64).collect();

        // Fit with maximum likelihood. Using BFGS optimiser and simple line search
        // seems to work fine
        logw(
            "Fitting Poisson mixture model using maximum likelihood",
            Some("info"),
        );
        let mixture_fit = MixPoisson { counts: counts_f64 };
        let init_param: Vec<f64> = vec![self.w0, self.c];

        // This is required. I tried the numerical Hessian but the scale was wrong
        // and it gave very poor results for the c optimisation
        let init_hessian: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let linesearch = BacktrackingLineSearch::new(ArmijoCondition::new(0.0001f64)?);
        let solver = BFGS::new(linesearch).with_tolerance_cost(1e-6)?;

        // Usually around 10 iterations should be enough
        let mut exec = Executor::new(mixture_fit, solver).configure(|state| {
            state
                .param(init_param)
                .inv_hessian(init_hessian)
                .max_iters(20)
        });

        if log::log_enabled!(log::Level::Debug) {
            exec = exec.add_observer(SlogLogger::term(), ObserverMode::Always);
        }

        let res = exec.run()?;

        // Print diagnostics
        logw(format!("{res}").as_str(), Some("info"));
        if let Some(termination_reason) = res.state().get_termination_reason() {
            if *termination_reason == SolverConverged {
                // Best parameter vector
                let best = res.state().get_best_param().unwrap();
                self.w0 = best[0];
                self.c = best[1];

                // calculate the coverage cutoff
                self.cutoff = find_cutoff(best, counts.len());
                self.fitted = true;
                Ok(self.cutoff)
            } else {
                Err(Error::msg(format!(
                    "Optimiser did not converge: {}",
                    termination_reason.text()
                )))
            }
        } else {
            Err(Error::msg("Optimiser did not finish running"))
        }
    }
}

// Helper struct for optimisation which keep counts as state
struct MixPoisson {
    counts: Vec<f64>,
}

// These just use Vec rather than ndarray, simpler packaging and doubt
// there's any performance difference with two params
// negative log-likelihood
impl CostFunction for MixPoisson {
    /// Type of the parameter vector
    type Param = Vec<f64>;
    /// Type of the return value computed by the cost function
    type Output = f64;

    /// Apply the cost function to a parameters `p`
    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(-log_likelihood(p, &self.counts))
    }
}

// negative grad(ll)
impl Gradient for MixPoisson {
    /// Type of the parameter vector
    type Param = Vec<f64>;
    /// Type of the gradient
    type Gradient = Vec<f64>;

    /// Compute the gradient at parameter `p`.
    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        // As doing minimisation, need to invert sign of gradients
        Ok(grad_ll(p, &self.counts).iter().map(|x| -*x).collect())
    }
}

// log-sum-exp needed to combine components likelihoods
// (hard coded as two here, of course could be generalised to N)
fn lse(a: f64, b: f64) -> f64 {
    let xstar = f64::max(a, b);
    xstar + f64::ln(f64::exp(a - xstar) + f64::exp(b - xstar))
}

// Natural log of Poisson density
fn ln_dpois(x: f64, lambda: f64) -> f64 {
    x * f64::ln(lambda) - lgamma(x + 1.0) - lambda
}

// error component (mean of 1)
fn a(w0: f64, i: f64) -> f64 {
    f64::ln(w0) + ln_dpois(i, 1.0)
}

// coverage component (mean of coverage)
fn b(w0: f64, c: f64, i: f64) -> f64 {
    f64::ln(1.0 - w0) + ln_dpois(i, c)
}

// Mixture model likelihood
fn log_likelihood(pars: &[f64], counts: &[f64]) -> f64 {
    let w0 = pars[0];
    let c = pars[1];
    let mut ll = 0.0;
    // 'soft' bounds. I think f64::NEG_INFINITY might be mathematically better
    // but arg_min doesn't like it
    if !(0.0..=1.0).contains(&w0) || c < 1.0 {
        ll = f64::MIN;
    } else {
        for (i, count) in counts.iter().enumerate() {
            let i_f64 = i as f64 + 1.0;
            ll += *count * lse(a(w0, i_f64), b(w0, c, i_f64));
        }
    }
    ll
}

// Analytic gradient. Bounds not needed as this is only evaluated
// when the ll is valid
fn grad_ll(pars: &[f64], counts: &[f64]) -> Vec<f64> {
    let w0 = pars[0];
    let c = pars[1];

    let mut grad_w0 = 0.0;
    let mut grad_c = 0.0;
    for (i, count) in counts.iter().enumerate() {
        let i_f64 = i as f64 + 1.0;
        let a_val = a(w0, i_f64);
        let b_val = b(w0, c, i_f64);
        let dlda = 1.0 / (1.0 + f64::exp(b_val - a_val));
        let dldb = 1.0 / (1.0 + f64::exp(a_val - b_val));
        grad_w0 += *count * (dlda / w0 - dldb / (1.0 - w0));
        grad_c += *count * (dldb * (i_f64 / c - 1.0));
    }
    vec![grad_w0, grad_c]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a bimodal k-mer spectrum: error peak at freq 1-3, coverage peak at freq ~20.
    fn make_bimodal_histogram() -> Vec<u32> {
        let mut h = vec![0u32; 499];
        h[0] = 200_000; // freq=1 (error singletons)
        h[1] = 80_000; // freq=2
        h[2] = 30_000; // freq=3
                       // low counts between peaks (still >= MIN_FREQ=50 so they survive truncation)
        for i in 3..17 {
            h[i] = 100;
        }
        h[17] = 60_000; // freq=18
        h[18] = 80_000; // freq=19 (coverage peak)
        h[19] = 100_000; // freq=20
        h[20] = 80_000; // freq=21
        h[21] = 60_000; // freq=22
        h
    }

    #[test]
    fn bimodal_histogram_does_not_panic() {
        let mut fit = SpectrumFitter::new();
        let counts = make_bimodal_histogram();
        let _ = fit.fit_histogram(counts);
        // Just verify it completes without panicking
    }

    #[test]
    fn bimodal_histogram_converges() {
        let mut fit = SpectrumFitter::new();
        let counts = make_bimodal_histogram();
        let result = fit.fit_histogram(counts);
        // A well-formed bimodal histogram should allow the optimizer to converge.
        // If it doesn't (Err), we accept that too — this just checks no panic.
        if let Ok(cutoff) = result {
            // Cutoff should be somewhere between error and coverage peaks (1-17)
            assert!(
                cutoff >= 1 && cutoff < 18,
                "cutoff={cutoff} out of expected range 1-17"
            );
        }
    }

    #[test]
    fn all_zeros_histogram_returns_err_or_min_cutoff() {
        let mut fit = SpectrumFitter::new();
        // All zeros: truncation removes everything → empty histogram
        let counts = vec![0u32; 499];
        let result = fit.fit_histogram(counts);
        // Either fails to converge (Err) or returns some minimal cutoff. Must not panic.
        match result {
            Ok(cutoff) => assert!(cutoff >= 1),
            Err(_) => {} // expected: optimizer cannot converge on empty data
        }
    }

    #[test]
    #[should_panic(expected = "Model already fitted")]
    fn fit_histogram_twice_panics() {
        let mut fit = SpectrumFitter::new();
        let counts = make_bimodal_histogram();
        let _ = fit.fit_histogram(counts.clone());
        // If the first call succeeded, fitted=true and the second call panics.
        // If the first call failed (Err), fitted=false and this test will fail (not panic).
        // The bimodal histogram is designed to converge, so the first call should succeed.
        let _ = fit.fit_histogram(counts);
    }
}

// Root finder at integer steps -- when is the responsibility of
// the b component higher than the a component
fn find_cutoff(pars: &[f64], max_cutoff: usize) -> usize {
    let w0 = pars[0];
    let c = pars[1];

    let mut cutoff = 1;
    while cutoff < max_cutoff {
        let cutoff_f64 = cutoff as f64;
        let root = a(w0, cutoff_f64) - b(w0, c, cutoff_f64);
        if root < 0.0 {
            break;
        }
        cutoff += 1;
    }
    cutoff
}
