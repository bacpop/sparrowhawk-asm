//! Stable log-space arithmetic and probability density, mass, and recurrence helpers.

use super::models::Shelf;
use libm::lgamma;

/// Log-sum-exp of two terms, shifted by the larger so the exponentials stay in range.
#[inline]
pub(super) fn lse2(a: f64, b: f64) -> f64 {
    let maximum = a.max(b);
    if !maximum.is_finite() {
        return maximum;
    }
    maximum + ((a - maximum).exp() + (b - maximum).exp()).ln()
}

/// Log-sum-exp of three terms, shifted by the largest so the exponentials stay in range.
#[inline]
pub(super) fn lse3(a: f64, b: f64, c: f64) -> f64 {
    let m = a.max(b).max(c);
    if !m.is_finite() {
        return m;
    }
    m + ((a - m).exp() + (b - m).exp() + (c - m).exp()).ln()
}

/// Log of the three softmax weights, computed in log space so no exponential can overflow.
#[inline]
pub(super) fn ln_softmax3(a: f64, b: f64, c: f64) -> (f64, f64, f64) {
    let norm = lse3(a, b, c);
    (a - norm, b - norm, c - norm)
}

/// ln(e^high − e^low) for `high >= low`, computed as high + ln(−expm1(low − high)), so nothing
/// underflows. Minus infinity when `high` is minus infinity, and NaN when `low > high`.
#[inline]
pub(super) fn ln_sub_exp(high: f64, low: f64) -> f64 {
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
pub(super) fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let scaled = value.exp();
        scaled / (1.0 + scaled)
    }
}

/// ln Σ p(count) over counts 1 to `top`, from ln p(1) and the log ratio of consecutive terms.
pub(super) fn log_recurrence_mass<F>(mut log_probability: f64, top: usize, log_ratio: F) -> f64
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

/// Natural log of the Poisson density.
#[inline]
pub(super) fn ln_dpois(x: f64, lambda: f64) -> f64 {
    x * lambda.ln() - lgamma(x + 1.0) - lambda
}

/// Natural log of the negative-binomial density, by mean and dispersion (`variance = dispersion *
/// mean`), so `r = mean / (dispersion - 1)` and `p = 1 / dispersion`. Tends to the Poisson as the
/// dispersion approaches 1, which is where the genome lobe of a clean simulated library sits.
#[inline]
pub(super) fn ln_dnbinom(x: f64, mean: f64, dispersion: f64) -> f64 {
    if dispersion <= 1.0 + 1e-9 {
        return ln_dpois(x, mean);
    }
    let r = mean / (dispersion - 1.0);
    lgamma(x + r) - lgamma(x + 1.0) - lgamma(r)
        + r * (-dispersion.ln())
        + x * (1.0 - 1.0 / dispersion).ln()
}

/// The negative binomial's mode, as a count: floor(mean − (dispersion − 1)), or 0.
pub(super) fn discrete_nb_mode(mean: f64, dispersion: f64) -> usize {
    let mode = mean - (dispersion - 1.0);
    if mode.is_finite() && mode > 0.0 {
        mode.floor().min(usize::MAX as f64) as usize
    } else {
        0
    }
}

/// Log of the negative binomial's mass on [1, top].
pub(super) fn log_nb_mass(mean: f64, dispersion: f64, top: usize) -> f64 {
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

/// The shelf's trials: the single-copy mean, rounded, as the beta-binomial needs a whole number.
pub(super) fn shelf_trials(mean: f64) -> f64 {
    mean.round().max(1.0)
}

/// Natural log of the beta-binomial density over `trials` trials; minus infinity above them.
pub(super) fn ln_dbetabinom(x: f64, trials: f64, alpha: f64, beta: f64) -> f64 {
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
pub(super) fn shelf_log_table(mean: f64, shelf: Shelf, top: usize) -> Vec<f64> {
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
pub(super) fn ln_single_copy(count: f64, mean: f64, dispersion: f64, shelf: Option<Shelf>) -> f64 {
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
pub(super) fn mix_single_copy(lobe: f64, count: f64, shelf: Option<(Shelf, &[f64])>) -> f64 {
    shelf.map_or(lobe, |(s, shelf_table)| {
        let at = shelf_table
            .get(count as usize)
            .copied()
            .unwrap_or(f64::NEG_INFINITY);
        lse2((1.0 - s.share).ln() + lobe, s.share.ln() + at)
    })
}

/// Log of the single-copy component's mass inside `[1, top]`, the shelf read from its table.
pub(super) fn single_copy_log_mass(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessing::models::*;

    /// The negative binomial is a distribution with the mean it is given and `dispersion` times that
    /// as its variance, both just past the Poisson switch and well above it. Just past the switch it
    /// must also stay close to the Poisson, or the limit is discontinuous.
    #[test]
    fn the_negative_binomial_has_its_moments_and_tends_to_the_poisson() {
        let mean = 20.0;
        for dispersion in [1.001, 3.0] {
            let (mut total, mut first, mut second) = (0.0, 0.0, 0.0);
            for x in 0..=1_000 {
                let x = f64::from(x);
                let p = ln_dnbinom(x, mean, dispersion).exp();
                total += p;
                first += x * p;
                second += x * x * p;
            }
            let variance = second - first * first;
            assert!(
                (total - 1.0).abs() < 1e-9,
                "dispersion {dispersion}: mass {total}"
            );
            assert!(
                (first - mean).abs() < 1e-7,
                "dispersion {dispersion}: mean {first}"
            );
            assert!(
                (variance / (dispersion * mean) - 1.0).abs() < 1e-7,
                "dispersion {dispersion}: variance {variance}"
            );
        }
        for x in [0.0, 1.0, 5.0, 20.0, 40.0] {
            let (nb, po) = (ln_dnbinom(x, 20.0, 1.001), ln_dpois(x, 20.0));
            assert!((nb - po).abs() < 0.05, "x={x}: {nb} vs {po}");
        }
    }

    /// A(1, 1) beta-binomial is uniform over its trials, and any beta-binomial sums to one.
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
}
