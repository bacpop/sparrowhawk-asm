//! Native spectrum plots and diagnostic tables, with a browser no-op plot adapter.

use super::models::SpectrumFit;
use super::{SpectrumEstimate, MAX_GENOME_HOLES};
#[cfg(not(target_family = "wasm"))]
use plotters::{
    coord::Shift,
    prelude::*,
    style::text_anchor::{HPos, Pos, VPos},
};
use std::path::PathBuf;

/// Smallest useful horizontal range for a diagnostic plot with no empirical genomic peak.
#[cfg(not(target_family = "wasm"))]
const MIN_PLOT_RANGE: usize = 10;

/// The linear plot's Y axis stops at this multiple of the empirical peak's bar, so the error lobe does
/// not flatten the genome lobe.
#[cfg(not(target_family = "wasm"))]
const PEAK_Y_CAP_MULT: f64 = 3.0;

/// Values needed to explain the spectrum decision in the native PNG. Kept independent of plotting
/// types so the fitting path can carry it without pulling native-only code into WASM, where nothing
/// reads it.
#[derive(Clone, Copy, Default)]
#[cfg_attr(target_family = "wasm", allow(dead_code))]
pub(super) struct SpectrumPlotDiagnostics {
    pub(super) valley_seed: Option<usize>,
    pub(super) valley: Option<usize>,
    pub(super) empirical_peak: Option<usize>,
    pub(super) raw_valley: Option<usize>,
    pub(super) genomic_peak_seed: Option<usize>,
    pub(super) fit: Option<SpectrumFit>,
    pub(super) fit_attempted: bool,
    pub(super) genomic_guard: Option<u16>,
    pub(super) error_floor: Option<u16>,
}

impl SpectrumPlotDiagnostics {
    pub(super) fn new(
        estimate: &SpectrumEstimate,
        fit: Option<SpectrumFit>,
        fit_attempted: bool,
    ) -> Self {
        Self {
            valley_seed: (estimate.valley_seed > 0).then_some(estimate.valley_seed),
            valley: (estimate.valley > 0).then_some(estimate.valley),
            empirical_peak: (estimate.genomic_peak > 0).then_some(estimate.genomic_peak),
            raw_valley: estimate.raw_valley,
            genomic_peak_seed: estimate.raw_genomic_peak,
            fit,
            fit_attempted,
            genomic_guard: estimate
                .verdict
                .is_ok()
                .then(|| fit.and_then(|fit| fit.hole_cutoff(MAX_GENOME_HOLES)))
                .flatten(),
            error_floor: fit.map(|fit| fit.error_floor_cutoff()),
        }
    }
}

/// Writes the full spectrum beside its plots as `count<TAB>distinct`, so a run's cutoff decision can
/// be replayed offline. The plots show the fitting window and cannot be read back.
pub(super) fn write_kmer_spectrum_tsv(histovec: &[u32], out_path: &std::path::Path) {
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

/// `path` with `suffix` appended to its file stem and `extension` set; dots already in the stem stay.
pub(super) fn suffixed_path(path: &std::path::Path, suffix: &str, extension: &str) -> PathBuf {
    let mut name = path.file_stem().unwrap_or_default().to_os_string();
    name.push(format!("{suffix}.{extension}"));
    path.with_file_name(name)
}

#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MarkerLineStyle {
    Solid,
    Dashed,
}

#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlotYAxis {
    /// Linear, topped at [`PEAK_Y_CAP_MULT`] times the empirical peak's bar, or at the tallest bar or
    /// curve without a peak.
    PeakCapped,
    /// Logarithmic, up to the tallest bar or curve.
    Log,
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
    if let Some(seed) = diagnostics.valley_seed {
        markers.push(PlotMarker {
            x: seed as f64,
            colour: RGBColor(125, 125, 125),
            label: format!("Valley seed = {seed}"),
            line_style: MarkerLineStyle::Dashed,
        });
    }
    if let Some(valley) = diagnostics.raw_valley {
        markers.push(PlotMarker {
            x: valley as f64,
            colour: RGBColor(125, 125, 125),
            label: format!("Raw valley candidate = {valley}"),
            line_style: MarkerLineStyle::Dashed,
        });
    }
    if let Some(peak) = diagnostics.genomic_peak_seed {
        markers.push(PlotMarker {
            x: peak as f64,
            colour: RGBColor(125, 125, 125),
            label: format!("Genomic peak seed = {peak}"),
            line_style: MarkerLineStyle::Dashed,
        });
    }
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
    if let Some(guard) = diagnostics.genomic_guard {
        markers.push(PlotMarker {
            x: f64::from(guard),
            colour: RGBColor(20, 135, 65),
            label: format!("Genomic hole guard = {guard}"),
            line_style: MarkerLineStyle::Dashed,
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
        if let Some(crossover) = fit.crossover() {
            markers.push(PlotMarker {
                x: f64::from(crossover),
                colour: RGBColor(120, 70, 200),
                label: format!("Error/genome crossover = {crossover}"),
                line_style: MarkerLineStyle::Solid,
            });
        }
    }
    if let Some(error_floor) = diagnostics.error_floor {
        markers.push(PlotMarker {
            x: f64::from(error_floor),
            colour: RGBColor(105, 105, 105),
            label: format!("Enforced 100% error floor = {error_floor}"),
            line_style: MarkerLineStyle::Dashed,
        });
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
fn plot_x_end(
    spectrum_len: usize,
    diagnostics: &SpectrumPlotDiagnostics,
    used_min_count: u16,
) -> usize {
    let histogram_end = spectrum_len.saturating_sub(1).max(1);
    let fit_end = diagnostics
        .fit
        .map_or_else(
            || {
                diagnostics.empirical_peak.map(|peak| {
                    crate::preprocessing::spectrum_fitter::fit_window_end(spectrum_len, peak)
                })
            },
            |fit| Some(fit.fit_window_end),
        )
        .unwrap_or(MIN_PLOT_RANGE);
    let marker_end = plot_markers(diagnostics, used_min_count)
        .iter()
        .map(|marker| marker.x.ceil() as usize)
        .max()
        .unwrap_or(0);
    fit_end
        .max(marker_end)
        .max(MIN_PLOT_RANGE)
        .min(histogram_end)
}

#[cfg(not(target_family = "wasm"))]
fn plot_y_max(
    decision_spectrum: &[u32],
    raw_bloom_spectrum: Option<&[u32]>,
    diagnostics: &SpectrumPlotDiagnostics,
    plot_end: usize,
) -> f64 {
    let decision_max = decision_spectrum
        .iter()
        .take(plot_end)
        .copied()
        .max()
        .unwrap_or(0);
    let raw_max = raw_bloom_spectrum
        .into_iter()
        .flat_map(|spectrum| spectrum.iter().take(plot_end))
        .copied()
        .max()
        .unwrap_or(0);
    let fit_max = diagnostics.fit.map_or(0.0, |fit| {
        (1..=plot_end.min(fit.fit_window_end))
            .map(|count| fit.component_heights(count).iter().sum::<f64>())
            .fold(0.0, f64::max)
    });

    f64::from(decision_max).max(f64::from(raw_max)).max(fit_max)
}

#[cfg(not(target_family = "wasm"))]
fn transform_plot_y(value: f64, y_axis: PlotYAxis) -> f64 {
    match y_axis {
        PlotYAxis::PeakCapped => value,
        // A half-count baseline keeps count-1 bins visible. Zero-height bars remain invisible at
        // that baseline, and zero-valued line points are omitted from logarithmic series.
        PlotYAxis::Log => value.max(0.5).log10(),
    }
}

#[cfg(not(target_family = "wasm"))]
fn plot_axis_y_min(y_axis: PlotYAxis) -> f64 {
    match y_axis {
        PlotYAxis::PeakCapped => 0.0,
        PlotYAxis::Log => 0.5f64.log10(),
    }
}

#[cfg(not(target_family = "wasm"))]
fn plot_axis_y_max(
    y_axis: PlotYAxis,
    decision_spectrum: &[u32],
    raw_bloom_spectrum: Option<&[u32]>,
    diagnostics: &SpectrumPlotDiagnostics,
    plot_end: usize,
) -> f64 {
    let tallest = plot_y_max(decision_spectrum, raw_bloom_spectrum, diagnostics, plot_end) * 1.05;
    match y_axis {
        PlotYAxis::PeakCapped => diagnostics
            .empirical_peak
            .and_then(|peak| decision_spectrum.get(peak - 1))
            .map_or(tallest, |&height| {
                tallest.min(PEAK_Y_CAP_MULT * f64::from(height))
            })
            .max(1.0),
        PlotYAxis::Log => transform_plot_y(tallest.max(10.0), y_axis),
    }
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
    y_axis: PlotYAxis,
) where
    DB::ErrorType: 'static,
{
    let markers = plot_markers(diagnostics, used_min_count);
    let plot_end = plot_x_end(decision_spectrum.len(), diagnostics, used_min_count);
    let y_max = plot_axis_y_max(
        y_axis,
        decision_spectrum,
        raw_bloom_spectrum,
        diagnostics,
        plot_end,
    );
    let mut notes: Vec<String> = markers
        .iter()
        .filter(|marker| marker.x > plot_end as f64)
        .map(|marker| format!("{} (off-scale)", marker.label))
        .collect();
    if diagnostics.fit_attempted && diagnostics.fit.is_none() {
        notes.push("mixture fit unavailable".to_string());
    }
    let plot_kind = match y_axis {
        PlotYAxis::PeakCapped => "k-mer spectrum",
        PlotYAxis::Log => "k-mer spectrum (logarithmic Y)",
    };
    let caption = if notes.is_empty() {
        plot_kind.to_string()
    } else {
        format!("{plot_kind} — {}", notes.join(", "))
    };

    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(35)
        .y_label_area_size(65)
        .margin(5)
        .caption(caption, 24.0)
        .build_cartesian_2d(0.0f64..plot_end as f64, plot_axis_y_min(y_axis)..y_max)
        .unwrap();
    chart
        .configure_mesh()
        .disable_x_mesh()
        .x_labels(15)
        .y_labels(11)
        .max_light_lines(2)
        .light_line_style(RGBColor(220, 220, 220).mix(0.35))
        .bold_line_style(RGBColor(180, 180, 180).mix(0.4))
        .y_label_formatter(&|y| match y_axis {
            PlotYAxis::PeakCapped => format!("{y:.0}"),
            PlotYAxis::Log => {
                let count = 10.0f64.powf(*y);
                if count < 1.0 {
                    "0.5".to_string()
                } else {
                    format!("{count:.0}")
                }
            }
        })
        .draw()
        .unwrap();

    let shown = &decision_spectrum[..plot_end.min(decision_spectrum.len())];
    chart
        .draw_series(shown.iter().enumerate().map(|(i, &height)| {
            let count = (i + 1) as f64;
            let height = transform_plot_y(f64::from(height), y_axis);
            Rectangle::new(
                [
                    (count - 0.5, plot_axis_y_min(y_axis)),
                    (count + 0.5, height),
                ],
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
                    .filter(|&(_, &height)| height > 0)
                    .map(|(i, &height)| {
                        ((i + 1) as f64, transform_plot_y(f64::from(height), y_axis))
                    }),
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
                    .filter(|(_, values)| values.iter().sum::<f64>() > 0.0)
                    .map(|(count, values)| (*count, transform_plot_y(values.iter().sum(), y_axis))),
                BLACK.stroke_width(2),
            ))
            .unwrap()
            .label("Fitted mixture")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 16, y)], BLACK.stroke_width(2)));
        chart
            .draw_series(DashedLineSeries::new(
                components
                    .iter()
                    .filter(|(_, values)| values[0] > 0.0)
                    .map(|(count, values)| (*count, transform_plot_y(values[0], y_axis))),
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
                components
                    .iter()
                    .filter(|(_, values)| values[1] > 0.0)
                    .map(|(count, values)| (*count, transform_plot_y(values[1], y_axis))),
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
        if fit.shelf.is_some() {
            chart
                .draw_series(DashedLineSeries::new(
                    (1..=plot_end.min(fit.fit_window_end))
                        .filter_map(|count| Some((count as f64, fit.shelf_height(count)?)))
                        .filter(|&(_, height)| height > 0.0)
                        .map(|(count, height)| (count, transform_plot_y(height, y_axis))),
                    3,
                    3,
                    RGBColor(30, 90, 220).stroke_width(1),
                ))
                .unwrap()
                .label("Shelf (beta-binomial)")
                .legend(|(x, y)| {
                    PathElement::new(
                        vec![(x, y), (x + 16, y)],
                        RGBColor(30, 90, 220).stroke_width(1),
                    )
                });
        }
        chart
            .draw_series(DashedLineSeries::new(
                components
                    .iter()
                    .filter(|(_, values)| values[2] > 0.0)
                    .map(|(count, values)| (*count, transform_plot_y(values[2], y_axis))),
                2,
                5,
                RGBColor(100, 100, 100).stroke_width(1),
            ))
            .unwrap()
            .label(format!(
                "Two-copy ({})",
                crate::preprocessing::models::GenomeModel::NegativeBinomial
            ))
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
                [(marker.x, plot_axis_y_min(y_axis)), (marker.x, y_max)],
                marker.colour.stroke_width(1),
            )),
            MarkerLineStyle::Dashed => chart.draw_series(DashedLineSeries::new(
                [(marker.x, plot_axis_y_min(y_axis)), (marker.x, y_max)],
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
    let axis_style = 15.0.into_text_style(&root);
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

/// The browser plots the spectrum on the page; `out_path` is always `None` there.
#[cfg(target_family = "wasm")]
pub(super) fn plot_kmer_histogram(
    _decision_spectrum: &[u32],
    _raw_bloom_spectrum: Option<&[u32]>,
    _diagnostics: &SpectrumPlotDiagnostics,
    _used_min_count: u16,
    _out_path: &std::path::Path,
) {
}

#[cfg(not(target_family = "wasm"))]
pub(super) fn plot_kmer_histogram(
    decision_spectrum: &[u32],
    raw_bloom_spectrum: Option<&[u32]>,
    diagnostics: &SpectrumPlotDiagnostics,
    used_min_count: u16,
    out_path: &std::path::Path,
) {
    // Register before constructing any text style; with Plotters' `ttf` feature disabled,
    // `ab_glyph` reads this bundled font instead of querying system fonts.
    static REGISTER_FONT: std::sync::Once = std::sync::Once::new();
    REGISTER_FONT.call_once(|| {
        assert!(
            plotters::style::register_font(
                "sans-serif",
                FontStyle::Normal,
                include_bytes!("../../assets/fonts/IBMPlexSans-Regular.ttf"),
            )
            .is_ok(),
            "Bundled IBM Plex Sans font is invalid"
        );
    });

    // `out_path` is the linear plot; the logarithmic one takes a `_log` suffix. Each is written as PNG
    // and as SVG.
    for (y_axis, png_path) in [
        (PlotYAxis::PeakCapped, out_path.to_path_buf()),
        (PlotYAxis::Log, suffixed_path(out_path, "_log", "png")),
    ] {
        draw_kmer_histogram(
            BitMapBackend::new(&png_path, (1280, 960)).into_drawing_area(),
            decision_spectrum,
            raw_bloom_spectrum,
            diagnostics,
            used_min_count,
            y_axis,
        );
        let svg_path = png_path.with_extension("svg");
        draw_kmer_histogram(
            SVGBackend::new(&svg_path, (1280, 960)).into_drawing_area(),
            decision_spectrum,
            raw_bloom_spectrum,
            diagnostics,
            used_min_count,
            y_axis,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessing::models::{ErrorModel, ErrorParams, GenomeModel};
    use crate::preprocessing::MAXSIZEHISTO;

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn plot_paths_keep_dots_in_the_output_prefix() {
        let base = std::path::Path::new("out/s.v2_kmerspectrum_k21.png");
        assert_eq!(
            suffixed_path(base, "_q20", "png"),
            std::path::Path::new("out/s.v2_kmerspectrum_k21_q20.png")
        );
        assert_eq!(
            suffixed_path(base, "_log", "png"),
            std::path::Path::new("out/s.v2_kmerspectrum_k21_log.png")
        );
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
        let fit = SpectrumFit::for_test(
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
        let diagnostics = SpectrumPlotDiagnostics::new(&estimate, Some(fit), true);

        assert_eq!(
            plot_axis_y_max(PlotYAxis::PeakCapped, &decision, None, &diagnostics, 570),
            PEAK_Y_CAP_MULT * f64::from(decision[94]),
            "the linear plot is capped at three times the empirical peak's bar"
        );
        let no_peak = SpectrumPlotDiagnostics::default();
        assert_eq!(
            plot_axis_y_max(PlotYAxis::PeakCapped, &decision, None, &no_peak, 570),
            plot_y_max(&decision, None, &no_peak, 570) * 1.05,
            "without an empirical peak the linear plot is unbounded"
        );

        for (suffix, raw) in [("exact", None), ("bloom", Some(raw_bloom.as_slice()))] {
            let path = std::env::temp_dir().join(format!(
                "sparrowhawk-spectrum-{suffix}-{}.png",
                std::process::id()
            ));
            let svg_path = path.with_extension("svg");
            let log_path = suffixed_path(&path, "_log", "png");
            let log_svg_path = log_path.with_extension("svg");
            plot_kmer_histogram(&decision, raw, &diagnostics, 12, &path);
            for written in [&path, &svg_path, &log_path, &log_svg_path] {
                assert!(std::fs::metadata(written).is_ok_and(|metadata| metadata.len() > 0));
            }
            let svg = std::fs::read_to_string(&svg_path).expect("read spectrum SVG");
            assert!(svg.contains("<svg"));
            assert!(svg.contains("Counts"));
            assert!(svg.contains("k-mer frequency"));
            assert!(svg.contains("k-mer spectrum"));
            assert!(!svg.contains("logarithmic"));
            assert!(svg.contains("Minimum count used = 12"));
            let log_svg = std::fs::read_to_string(&log_svg_path).expect("read logarithmic SVG");
            assert!(log_svg.contains("k-mer spectrum (logarithmic Y)"));
            for written in [path, svg_path, log_path, log_svg_path] {
                std::fs::remove_file(written).expect("remove temporary spectrum plot");
            }
        }
    }
}
