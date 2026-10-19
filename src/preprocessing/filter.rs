//! Select quality floors and count cutoffs, then filter the counted k-mers.

use super::models::SpectrumFit;
use super::parsercounter::{carry_contig_kmers, countmaps_into_indexed_kmers, CountMap};
use super::plotter::{
    plot_kmer_histogram, suffixed_path, write_kmer_spectrum_tsv, SpectrumPlotDiagnostics,
};
use super::read_store::PackedSeqs;
use super::sketch::{check_sketch_against_table, rescaled_strict_spectrum, SpectrumSketch};
use super::spectrum_fitter::{fit_spectrum, FitAttempt};
use super::{
    add_to_histogram, distinct_above, occurrence_weighted_median, PeakSearch, PeakSource,
    PreprocessingError, SpectrumEstimate, MAXSIZEHISTO, MAX_GENOME_HOLES, UNRESOLVED_MINCOUNT,
};
use crate::graph_works::indexed_kmers::IndexedKmers;
use crate::utils::{bit_encoding::UInt, clock::Instant, qual_profile::MAX_GROUPS};
use crate::QualOpts;
use std::path::PathBuf;

/// A fit whose error component explains more than this share of observed counts above the applied
/// cutoff is considered too error-heavy when comparing multiple usable quality floors. Provisional;
/// cluster results should be used to recalibrate it.
const MAX_FIT_ERROR_TAIL_FRACTION: f64 = 0.05;

/// A very shallow empirical peak is a reason to inspect the lower floors, not a reason to refuse a fit.
const LOW_COVERAGE_FLOOR_REVIEW: usize = 10;

/// Peak gain a lower floor needs over a starved one (peak below [`LOW_COVERAGE_FLOOR_REVIEW`]) to be
/// stepped down to. Well-covered floors never step down: that cost Neisseria 6-8% auN locally.
const STEP_DOWN_GAIN: f64 = 1.10;

/// Fit the spectrum and report it. Seeded from the valley estimator's own peak and dispersion: the
/// estimator locates the lobe, the fit refines it and separates the error lobe from it explicitly.
/// `None` when no start converges, which leaves the estimator's answer standing.
/// `sample_scale`: library k-mers per observed one, 1/fraction for a sketch and 1 for an exact table.
pub(super) fn fit_and_log(
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

/// As [`choose_min_count`], and also the single-copy coverage the same estimate found.
///
/// The valley separates the lobes; the fit then says how much genome cutting there would cost. Those
/// answer different questions, and on deep libraries the valley is far the more aggressive.
/// `sample_scale` as in [`fit_and_log`].
pub(super) fn choose_min_count_and_peak(
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

pub(super) struct FloorDiagnostic {
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
pub(super) fn choose_min_count_and_floor_with_debug<F>(
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
pub(super) fn choose_cutoff_and_filter<IntT>(
    mut shards: Vec<CountMap<IntT>>,
    k: usize,
    qual: &QualOpts,
    floors: Option<&[u8]>,
    floor_has_kmers: &[bool; MAX_GROUPS],
    sketch: Option<SpectrumSketch>,
    do_fit: bool,
    do_bloom: bool,
    carried: Option<&PackedSeqs>,
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
    if kmers.is_empty() {
        return Err(PreprocessingError(format!(
            "no k-mers survived filtering at k={k}, base-quality floor {chosen_min_qual}, and minimum count {minc}"
        )));
    }
    Ok((kmers, histovec, minc, chosen_min_qual, genomic_peak))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessing::sketch::test_support::sketch_with;
    use crate::preprocessing::{
        models::{ErrorModel, ErrorParams, GenomeModel},
        test_support::*,
    };
    use crate::preprocessing::{parsercounter::KmerInfo, Verdict};
    use crate::utils::qual_profile::MAX_GROUPS;
    use crate::utils::reads::OwnedRecord;
    use std::{collections::HashMap, hash::BuildHasherDefault};

    /// When the hole guard would cut below the fitted 100 % error floor, the floor wins: a resolved
    /// cutoff is lowered to the hole cutoff and then raised back to the floor, never left under it.
    #[test]
    fn active_error_floor_remains_a_hard_minimum() {
        let fit = SpectrumFit::for_test(
            ErrorModel::SingletonPareto,
            GenomeModel::NegativeBinomial,
            [0.99, 0.0099, 0.0001],
            50.0,
            6.0,
            ErrorParams {
                singleton_probability: 0.85,
                tail_exponent: 2.0,
                weibull_shape: None,
            },
            10.0e6,
            300,
        );
        let floor = fit.error_floor_cutoff();
        let holes = fit
            .hole_cutoff(MAX_GENOME_HOLES)
            .expect("the lobe reaches low counts");
        assert!(floor > 2, "the fixture needs an active floor, got {floor}");
        assert!(
            holes < floor,
            "hole cutoff {holes} must conflict with floor {floor}"
        );
        let mut estimate = SpectrumEstimate {
            min_count: 20,
            verdict: Verdict::Ok,
            ..SpectrumEstimate::default()
        };
        assert_eq!(estimate.hole_guarded_min_count(&fit), holes);
        estimate.apply_hole_guard(&fit);
        assert_eq!(
            estimate.min_count, floor,
            "the full error floor is a hard minimum"
        );
    }

    #[test]
    fn empty_counts_at_every_candidate_floor_are_an_error() {
        let qual = QualOpts {
            min_count: 2,
            min_qual: 25,
        };
        let floors = [0u8, 11, 25];
        let mut iters = [Vec::<OwnedRecord>::new().into_iter()];
        let (shards, sketch, floor_has_kmers) =
            count_text::<u64, _>(&mut iters, 31, &qual, Some(&floors), false, true);
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
            count_text::<u64, _>(&mut iters, 31, &qual, Some(&floors), false, true);
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

    /// An empty sketch is not proof that a floor has no k-mers. With fitting on, a spectrum that
    /// resolves nowhere drops to the loosest floor exact counting saw k-mers at and asks for a recount
    /// there, rather than failing, though the sketch sampled nothing at any floor.
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
        let (kmers, _, minc, chosen_min_qual, _) = choose_cutoff_and_filter::<u64>(
            vec![shard],
            31,
            &qual,
            Some(&floors),
            &[true; MAX_GROUPS],
            Some(SpectrumSketch::new()),
            true,
            false,
            None,
            &mut None,
            false,
            None,
        )
        .expect("the sketch sample being empty is not proof that the floor has no k-mers");
        assert_eq!(chosen_min_qual, 0, "the loosest floor with k-mers");
        assert_eq!(minc, UNRESOLVED_MINCOUNT);
        assert_eq!(kmers.len(), 0, "a recount pass must not build the table");
    }

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

    /// Lower floors exist only in the sketch, so they must be fitted at the whole library's scale, as
    /// the strict floor is: the hole guard budgets in absolute k-mers. The fit is told that scale, so
    /// its likelihood counts the sample rather than the rescaled bins.
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

    /// The point of reading the spectrum from the sketch: the Bloom path now reports a usable peak,
    /// even though its own histogram has no count-1 bin to read one from.
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

    /// Floors only step down below [`LOW_COVERAGE_FLOOR_REVIEW`]: a resolved strict floor at 14 or 40 stays,
    /// however deep a lower floor is, and costs no extra fit.
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

    /// The tail a floor is judged by is measured from the cutoff the floor would actually apply, so
    /// error k-mers that cutoff removes anyway do not count against it.
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

        let candidate = evaluate_floor(
            25,
            &histogram,
            1.0,
            &mut |_: &[u32], _: &SpectrumEstimate, _| Some(fit),
        );
        assert!(candidate.fit_attempted);
        assert_eq!(candidate.error_tail_fraction, Some(applied));
    }

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

    /// Every carried k-mer passes the cutoff. A count the reads already clear is never touched, and
    /// carrying a contig twice changes nothing.
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
        let (mut shards, _, _) = count_text::<u64, _>(
            &mut [reads.clone().into_iter()],
            k,
            &qual,
            Some(&ladder[..]),
            false,
            true,
        );
        let before = as_counts(&shards);
        let absent = carry_contig_kmers(&mut shards, &packed_contigs(&contigs), k, minc);
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

    /// The point of multi-k: two tilings overlapping by 30 bases read every 21-mer but miss ten
    /// 41-mers, so k=41 alone breaks there, and carrying the k=21 contig joins it again. `u128`
    /// holds both k, as the pipeline would pick it for k=41.
    #[test]
    fn carrying_the_small_k_contig_bridges_a_seam_the_large_k_reads_miss() {
        use crate::graph_works::corrector::{CorrectionOpts, CoverageRef};
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
        let assemble = |k: usize, carried: Option<&PackedSeqs>| -> PackedSeqs {
            let (shards, sketch, present) = count_text::<u128, _>(
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
            let contigs = BasicAsm::assemble::<u128>(k, &mut kmers, correction, None).unwrap();
            crate::graph_works::spelling::spell_contigs::<u128>(
                &contigs,
                |hash| kmers.get_packed(hash).copied(),
                k,
            )
            .unwrap()
        };

        let small = assemble(21, None);
        assert_eq!(small.len(), 1, "every 21-mer is read");
        assert_eq!(assemble(41, None).len(), 2, "ten 41-mers are never read");
        let large = assemble(41, Some(&small));
        assert_eq!(large.len(), 1, "the carried contig bridges the seam");
        assert_eq!(large.iter().next().unwrap().len(), genome.len());
    }
}
