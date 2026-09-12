//! Some docs should be here

#[cfg(not(target_family = "wasm"))]
use std::{path::PathBuf, time::Instant};

use libm::lgamma;
use nohash_hasher::NoHashHasher;
use std::{cmp::Ordering, collections::HashMap, hash::BuildHasherDefault};

use rayon::prelude::*;
// use std::process::exit;

#[cfg(not(target_family = "wasm"))]
use plotters::prelude::*;

use super::HashInfoSimple;
use super::QualOpts;

// #[cfg(not(target_family = "wasm"))]
// use super::bit_encoding::{encode_base, rc_base};

use crate::bit_encoding::UInt;
use crate::bloom_filter::KmerFilter;
use crate::kmer::Kmer;
use crate::logw;
#[cfg(target_family = "wasm")]
use crate::spectrum_fitter::SpectrumFitter;

/// Tuple for name and list of input files
pub type InputFastx = (String, Vec<String>);

/// Everything the preprocessing of one k value produces. Named fields rather than a tuple, since the
/// backends and the public entry point order these differently.
#[cfg(not(target_family = "wasm"))]
pub struct PreprocessedK<IntT> {
    /// The k this was built with. Hashes from different k live in disjoint spaces, so carrying it
    /// alongside the maps is what stops us mixing them up.
    pub k: usize,
    /// canonical hash -> k-mer record (counts, hnc, bases, and later the pre/post neighbours)
    pub themap: HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
    /// canonical hash -> packed canonical k-mer bits, used to spell sequence back out
    pub thedict: HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    /// non-canonical hash -> canonical hash
    pub maxmindict: HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
    /// `MAXSIZEHISTO`-bin k-mer spectrum; index `c-1` holds the number of distinct k-mers seen `c` times
    pub histovec: Vec<u32>,
    /// the min-count actually applied (fitted, or taken from the CLI)
    pub used_min_count: u16,
}

// #[cfg(target_family = "wasm")]
// use wasm_bindgen::prelude::*;
#[cfg(target_family = "wasm")]
use crate::fastx_wasm::open_fastq;
#[cfg(target_family = "wasm")]
use crate::post_state;
#[cfg(target_family = "wasm")]
use seq_io::fastq::Record;
#[cfg(target_family = "wasm")]
use wasm_bindgen_file_reader::WebSysFile;

// For the fitting, we'll use actually MAXSIZEHISTO - 1
const MAXSIZEHISTO: usize = 8000;

/// Bins the original fit, peak finder and plot see. Pinned at the historical `MAXSIZEHISTO` so that
/// widening the histogram cannot move them: `add_to_histogram` writes count `c` to index `c-1`, so
/// indices 0..498 are identical either way and every legacy reader stays inside that window.
pub(crate) const LEGACY_HISTO_RANGE: usize = 500;

/// The pinned window has to fit inside the histogram, or the legacy readers would index out of bounds.
const _: () = assert!(LEGACY_HISTO_RANGE <= MAXSIZEHISTO);

#[inline]
fn add_to_histogram(histovec: &mut [u32], count: u32) {
    let idx = if (count as usize) >= MAXSIZEHISTO {
        MAXSIZEHISTO - 1
    } else {
        count as usize - 1
    };
    histovec[idx] = histovec[idx].saturating_add(1);
}

/// Single-copy coverage estimate straight from the spectrum: the tallest bin above the error peak.
///
/// Scans counts 3..=499 and returns a **count, not an index**; the range is pinned to
/// [`LEGACY_HISTO_RANGE`], so above ~500x the true peak is invisible here. It is also a *global* argmax,
/// so on a deep library the error lobe outvotes the genome lobe — see [`peak_above`].
///
/// Natively this now has no callers outside the tests, which assert exactly that failure; the browser
/// still reaches it through the mixture fit.
#[cfg_attr(all(not(target_family = "wasm"), not(test)), allow(dead_code))]
fn coverage_peak(histovec: &[u32]) -> usize {
    let mut best_count = 2usize; // nothing above the error peak; the caller's floor of 2 then applies
    let mut best_n = 0u32;
    for (i, &n) in histovec[2..(LEGACY_HISTO_RANGE - 1)].iter().enumerate() {
        if n > best_n {
            // Strict, so ties keep the lowest count and the result is deterministic.
            best_n = n;
            best_count = i + 3; // slice index 0 is count 3
        }
    }
    best_count
}

// =====================================================================================================
// An alternative `min_count` estimator, running beside the fit and for now only logged. It assumes no
// distribution at all: it walks up from the error lobe to the first trough and takes the peak above it.
// =====================================================================================================

/// Bins either side of a count in the smoothed spectrum, and the run of rising bins that confirms we
/// have left the error lobe rather than hit noise.
const SMOOTH: usize = 2;
const RISE_RUN: usize = 3;
/// Genome k-mers the cutoff may delete, and how far the genome lobe must stand above the trough.
const MAX_GENOME_LOSS: f64 = 0.01;
const MIN_LOBE_RATIO: u64 = 3;
/// Used when the lobes cannot be separated. Not 1: at a peak of 3-4 every singleton error survives and
/// we exhaust memory, which is worse than the ~20 % genome loss cutting at 2 costs there.
const UNRESOLVED_MINCOUNT: u16 = 2;

/// What the trough estimator concluded. This is what actually filters; the fit is logged beside it.
struct TroughEstimate {
    trough: usize,
    peak: usize,
    min_count: u16,
    /// Variance-to-mean ratio of the genome lobe, or NaN when there is no usable lobe.
    dispersion: f64,
    verdict: &'static str,
}

/// Smoothed spectrum height at `count` (a count, not an index).
fn smoothed(histovec: &[u32], count: usize) -> u64 {
    let lo = count.saturating_sub(SMOOTH).max(1);
    let hi = (count + SMOOTH).min(histovec.len() - 1);
    (lo..=hi).map(|c| histovec[c - 1] as u64).sum::<u64>() / (hi - lo + 1) as u64
}

/// First local minimum followed by [`RISE_RUN`] rising bins, as a **count**. `None` when the spectrum
/// never turns back up. Only a *seed* for [`error_valley`]: on a deep library whose error lobe decays
/// without a local minimum this walks into the genome lobe, which is harmless once bounded by the peak.
fn error_trough(histovec: &[u32]) -> Option<usize> {
    let hi = histovec.len() - 1; // the saturating bin is not part of the shape
    let mut best_count = 2usize;
    let mut best_n = smoothed(histovec, 2);
    let mut rising = 0usize;
    for count in 3..hi {
        let n = smoothed(histovec, count);
        if n < best_n {
            best_n = n;
            best_count = count;
            rising = 0;
        } else {
            rising += 1;
            if rising >= RISE_RUN {
                return Some(best_count);
            }
        }
    }
    None
}

/// Lowest smoothed bin in `2..=peak`, as a **count**: the valley between the error lobe and the genome
/// lobe. Bounded above by the peak, so unlike [`error_trough`] it cannot walk off into the lobe itself.
fn error_valley(histovec: &[u32], peak: usize) -> usize {
    let mut best_count = 2usize;
    let mut best_n = smoothed(histovec, 2);
    for count in 3..=peak.min(histovec.len() - 1) {
        let n = smoothed(histovec, count);
        if n < best_n {
            best_n = n;
            best_count = count;
        }
    }
    best_count
}

/// The cutoff ceiling: the tighter of a measured bound and a Poisson one. Neither is trustworthy alone
/// — each is conservative exactly where the other fails — so the guard is the smaller of the two.
fn loss_guard(histovec: &[u32], trough: usize, peak: usize) -> u16 {
    measured_guard(histovec, trough, peak).min(poisson_guard(peak))
}

/// Largest cutoff whose *measured* cost is at most [`MAX_GENOME_LOSS`] of the k-mers above the trough.
/// This is the binding one at depth, where the genome lobe measures 8-12x overdispersed and a Poisson
/// tail would permit a cutoff about 35 % too high.
fn measured_guard(histovec: &[u32], trough: usize, peak: usize) -> u16 {
    let total: u128 = (trough..histovec.len())
        .map(|c| histovec[c - 1] as u128)
        .sum();
    if total == 0 {
        return UNRESOLVED_MINCOUNT;
    }
    let budget = (total as f64 * MAX_GENOME_LOSS) as u128;
    // `spent` is what cutting at `m` already costs, so we step up only while the next bin still fits;
    // the value returned is therefore the last cutoff inside the budget, never the first one outside.
    let mut spent = 0u128;
    let mut m = trough;
    while m < peak {
        let next = spent + histovec[m - 1] as u128;
        if next > budget {
            break;
        }
        spent = next;
        m += 1;
    }
    (m as u16).max(2)
}

/// Largest cutoff deleting at most [`MAX_GENOME_LOSS`] of a Poisson(`peak`) genome. This is the binding
/// one at low coverage, where the lobe really is near-Poisson and the measured bound is loosened by the
/// error k-mers that still sit above the trough.
fn poisson_guard(peak: usize) -> u16 {
    let lam = peak as f64;
    let mut cdf = (-lam).exp();
    let mut m = 1usize;
    while m < peak {
        let next = cdf + (-lam + m as f64 * lam.ln() - lgamma(m as f64 + 1.0)).exp();
        if next > MAX_GENOME_LOSS {
            break;
        }
        cdf = next;
        m += 1;
    }
    (m as u16).max(2)
}

/// Variance-to-mean ratio of the single-copy lobe; 1.0 is Poisson. Logged because it is what says
/// whether a negative binomial in the mixture fit would be worth the work.
///
/// Stops at twice the peak: beyond that lie the repeat copies and the saturating bin, and including
/// them measures the spread of the whole spectrum rather than of the lobe the fit tries to model.
fn dispersion_above(histovec: &[u32], trough: usize, peak: usize) -> f64 {
    let top = (2 * peak).min(histovec.len() - 1);
    let (mut n, mut sx, mut sxx) = (0f64, 0f64, 0f64);
    for count in trough..top {
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

/// Tallest bin at or above the trough, as a **count**. Unlike [`coverage_peak`] this cannot lock onto
/// the error lobe, which is the whole difference between the two estimators.
fn peak_above(histovec: &[u32], trough: usize) -> usize {
    let hi = histovec.len() - 1;
    let mut best_count = trough;
    let mut best_n = 0u32;
    for count in trough..hi {
        if histovec[count - 1] > best_n {
            // Strict, so ties keep the lowest count.
            best_n = histovec[count - 1];
            best_count = count;
        }
    }
    best_count
}

/// The trough estimate for this spectrum. Falls back to [`UNRESOLVED_MINCOUNT`] with a stated verdict
/// wherever the lobes are not separated — at 3-5x, trusting it blindly deleted 98-99 % of the genome.
fn estimate_by_trough(histovec: &[u32]) -> TroughEstimate {
    let bail = |trough, peak, verdict| TroughEstimate {
        trough,
        peak,
        min_count: UNRESOLVED_MINCOUNT,
        dispersion: f64::NAN,
        verdict,
    };
    // The seed only has to land past the error head, not on the valley: the search below runs from 2,
    // so an overshoot into the genome lobe still yields the right answer. That is what stopped 142 of
    // the 195 bail-outs in the 2026-09-11 sweep, where the walk returned a count inside the lobe.
    let Some(seed) = error_trough(histovec) else {
        return bail(0, 0, "the spectrum never turns back up");
    };
    let peak = peak_above(histovec, seed);
    let trough = error_valley(histovec, peak);
    if peak <= trough {
        return bail(trough, peak, "no peak above the trough");
    }
    // k-mer INSTANCES, not distinct k-mers: at 500x the genome is under 1 % of distinct k-mers but most
    // of the sequence, so a distinct-count test would reject a perfectly healthy deep library.
    let instances_from = |from: usize| -> u128 {
        (from..=histovec.len())
            .map(|c| c as u128 * histovec[c - 1] as u128)
            .sum()
    };
    let total = instances_from(1);
    let above = instances_from(trough);
    if total == 0 || above * 100 < total {
        return bail(
            trough,
            peak,
            "the lobe above the trough holds under 1 % of the sequence",
        );
    }
    if (histovec[peak - 1] as u64) < MIN_LOBE_RATIO * (histovec[trough - 1].max(1) as u64) {
        return bail(
            trough,
            peak,
            "the lobe above the trough is not raised clear of it",
        );
    }
    TroughEstimate {
        trough,
        peak,
        // The trough removes the errors; the guard bounds what that costs in genome. The guard is the
        // binding one at low coverage, where the two lobes crowd together; at depth it has slack spare.
        min_count: (trough as u16).clamp(2, loss_guard(histovec, trough, peak)),
        dispersion: dispersion_above(histovec, trough, peak),
        verdict: "ok",
    }
}

/// What the estimator concluded, in one line.
fn log_spectrum(histovec: &[u32], estimate: &TroughEstimate) {
    logw(
        &format!(
            "K-mer spectrum: trough at count {}, peak at count {}, lobe dispersion {:.1}x Poisson. \
             Using min_count {} ({}).{}",
            estimate.trough,
            estimate.peak,
            estimate.dispersion,
            estimate.min_count,
            estimate.verdict,
            if histovec[histovec.len() - 1] > 0 {
                " NOTE: the spectrum saturates the histogram."
            } else {
                ""
            }
        ),
        Some("info"),
    );
}

/// What the Poisson mixture would have chosen. Kept for the browser, where there is no benchmark sweep
/// to calibrate against; native runs do not fit at all, since the estimator beat it wherever they
/// disagreed across 3780 sweep runs.
#[cfg(target_family = "wasm")]
fn log_fit_comparison(histovec: &[u32]) {
    let peak = coverage_peak(histovec);
    let floor = ((peak as f64 / 8.0).round() as u16).max(2);

    let mut fit = SpectrumFitter::new();
    let would_be = match fit.fit_histogram(histovec[..(LEGACY_HISTO_RANGE - 1)].to_vec()) {
        Ok(minc) if minc > TRUST_FIT_ABOVE => format!("{minc}"),
        Ok(minc) => format!("{floor} (fit returned {minc}, too small to be trusted; peak {peak})"),
        Err(e) => format!("{floor} (fit did not converge: {e}; peak {peak})"),
    };
    logw(
        &format!("The Poisson mixture would have used {would_be}."),
        Some("info"),
    );
}

// =====================================================================================================

/// A fitted cutoff at or below this is treated as unreliable and replaced by the histogram floor.
/// Measured cutoffs split cleanly into a trustworthy group (14-52) and an untrustworthy one (2-8).
#[cfg(target_family = "wasm")]
const TRUST_FIT_ABOVE: usize = 10;


/// Choose the minimum k-mer count from the spectrum. The returned value is an **inclusive** minimum:
/// both filter sites keep k-mers with `count >= min_count`.
fn choose_min_count(histovec: &[u32]) -> u16 {
    let estimate = estimate_by_trough(histovec);
    log_spectrum(histovec, &estimate);
    #[cfg(target_family = "wasm")]
    log_fit_comparison(histovec);
    if estimate.verdict != "ok" {
        logw(
            &format!(
                "The k-mer spectrum's error and genome lobes are not separated ({}), so no reliable \
                 minimum count exists and {} will be used — k-mers seen once are discarded and nothing \
                 else is. Expect a fragmented assembly if this is a deep library. This usually means \
                 the spectrum is thin: low coverage, or a large k. Check the k-mer spectrum histogram.",
                estimate.verdict, estimate.min_count
            ),
            Some("warn"),
        );
    }
    estimate.min_count
}

fn build_histogram_from_countmap(
    countmap: &HashMap<u64, (u32, u64, u8), BuildHasherDefault<NoHashHasher<u64>>>,
    histovec: &mut [u32],
) {
    for (_, tup) in countmap.iter() {
        add_to_histogram(histovec, tup.0);
    }
}

fn drain_countmap_into_themap<IntT>(
    countmap: &mut HashMap<u64, (u32, u64, u8), BuildHasherDefault<NoHashHasher<u64>>>,
    themap: &mut HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
    outdict: &mut HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    minmaxdict: &mut HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
    minc: u16,
    mut histovec_opt: Option<&mut [u32]>,
) where
    IntT: for<'a> UInt<'a>,
{
    countmap.retain(|h, tup| {
        if tup.0 >= minc as u32 {
            themap.entry(*h).or_insert_with(|| HashInfoSimple {
                hnc: tup.1,
                b: tup.2,
                pre: Vec::new(),
                post: Vec::new(),
                counts: tup.0,
            });
        } else {
            outdict.remove(h);
            minmaxdict.remove(&tup.1);
        }
        if let Some(ref mut hv) = histovec_opt {
            add_to_histogram(hv, tup.0);
        }
        false
    });
}

#[cfg(not(target_family = "wasm"))]
fn extract_kmers_from_files<F, I>(
    input_iters: &mut [I],
    mut on_record: F
)
where
    F: FnMut(std::borrow::Cow<'_, [u8]>, usize, Option<&[u8]>),
    I: Iterator<Item=(Vec<u8>, Option<Vec<u8>>)>,
{
    for (idx, records) in input_iters.iter_mut().enumerate() {
        log::info!("Getting kmers from file number {idx}.");
        for record in records {
            let seq: Vec<u8> = record.0;
            let qual: Option<Vec<u8>> = record.1;
            let num_bases = seq.len();
            on_record(seq.into(), num_bases, qual.as_deref());
        }
        log::info!("Finished getting kmers from file number {idx}.");
    }
    log::info!("Finished getting kmers from {} file(s)", input_iters.len());
}

/// One FASTQ record, owned: needletail's `Cow` borrows a reader buffer invalidated on the next
/// `next()`, so a batch processed in parallel must own its bytes.
#[cfg(not(target_family = "wasm"))]
type OwnedRecord = (Vec<u8>, Option<Vec<u8>>);

/// How many records go to the workers at a time: ~1M k-mers at 150 bp and k=31, small enough to stay
/// cache-friendly and large enough to amortise the rayon fork/join.
#[cfg(not(target_family = "wasm"))]
const BATCH_RECORDS: usize = 8192;

/// Parse `files` into owned batches of records, handing each batch to `on_batch`. The parse stays
/// serial but overlaps with the workers processing the previous batch.
#[cfg(not(target_family = "wasm"))]
fn extract_kmers_from_files_batched<F, I>(
    input_iters: &mut [I],
    batch_records: usize,
    mut on_batch: F
)
where
    F: FnMut(&[OwnedRecord]),
    I: Iterator<Item=(Vec<u8>, Option<Vec<u8>>)>,
{
    let mut batch: Vec<OwnedRecord> = Vec::with_capacity(batch_records);
    for (idx, records) in input_iters.iter_mut().enumerate() {
        log::info!("Getting kmers from file number {idx}.");
        for record in records {
            let seq: Vec<u8> = record.0;
            let qual: Option<Vec<u8>> = record.1;
            batch.push((seq, qual));
            if batch.len() == batch_records {
                on_batch(&batch);
                batch.clear();
            }
        }
        log::info!("Finished getting kmers from file number {idx}.");
    }
    if !batch.is_empty() {
        on_batch(&batch);
    }
    log::info!("Finished getting kmers from {} file(s)", input_iters.len());
}

/// Draws the first [`LEGACY_HISTO_RANGE`] bins, so the PNG stays comparable with every earlier run. The
/// final bin no longer piles up everything above it — that pile is now out at [`MAXSIZEHISTO`] — so the
/// old spike at 500 is gone; the trough-estimator log line carries what lies beyond.
#[cfg(not(target_family = "wasm"))]
fn plot_kmer_histogram(histovec: &[u32], out_path: &std::path::Path) {
    let shown = &histovec[..LEGACY_HISTO_RANGE.min(histovec.len())];
    let backend = BitMapBackend::new(out_path, (1280, 960));
    let root = backend.into_drawing_area();
    let _ = root.fill(&WHITE);
    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(35)
        .y_label_area_size(40)
        .margin(5)
        .caption("k-mer spectrum", ("ibm-plex-sans", 30.0))
        .build_cartesian_2d(
            (0u32..(LEGACY_HISTO_RANGE as u32)).into_segmented(),
            0u32..200000u32,
        )
        .unwrap();
    chart
        .configure_mesh()
        .disable_x_mesh()
        .bold_line_style(WHITE.mix(0.3))
        .y_desc("Counts")
        .x_desc("k-mer frequency")
        .axis_desc_style(("ibm-plex-sans", 15))
        .draw()
        .unwrap();
    chart
        .draw_series(
            Histogram::vertical(&chart)
                .style(RED.filled())
                .data(shown.iter().enumerate().map(|(i, x)| (i as u32, *x))),
        )
        .unwrap();
    root.present()
        .expect("Unable to write result to file. Does the output folder exist?");
}

// =====================================================================================================

// NOTE: these two functions were implemented to save the whole sequence in memory for GPGPU processing.
// This might be useful again in the future, but now it was just meaning that we were wasting memory!
// I'm disabling them for the moment
//
// #[cfg(not(target_family = "wasm"))]
// fn put_these_nts_into_an_efficient_vector(charseq : &[u8], compseq : &mut Vec<u64>, occ : u8) {
//     let mut tmpu64 : u64 = 0;
//     let mut tmpind : u8  = 0;
//
//     if occ != 0 {
//         tmpind = occ;
//         tmpu64 = compseq.pop().unwrap();
//     }
// //     log::debug!("{}", tmpind);
//
//     for nt in charseq {
// //         log::debug!("\n{:#010b}\n{}\n{:#066b}\n{:#066b}", *nt, tmpind, tmpu64, (encode_base(*nt) as u64));
//         tmpu64 <<= 2;
//         tmpu64 |= encode_base(*nt) as u64;
// //         log::debug!("{:#066b}", tmpu64);
//         if tmpind == 31 {
//             compseq.push(tmpu64);
//             tmpu64 = 0;
//             tmpind = 0;
//         } else {
//             tmpind += 1;
//         }
// //         log::debug!("{}", tmpind);
//     }
//
//     if tmpind != 0 {
//         compseq.push(tmpu64);
//     }
// }

// #[cfg(not(target_family = "wasm"))]
// fn put_these_nts_into_an_efficient_vector_rc(charseq : &[u8], compseq : &mut Vec<u64>, occ : u8) {
//     let mut tmpu64 : u64 = 0;
//     let mut tmpind : u8  = 0;
//
//     if occ != 0 {
//         tmpind = occ;
//         tmpu64 = compseq.pop().unwrap();
//     }
// //     log::debug!("{}", tmpind);
//     for nt in charseq.iter().rev() {
// //         log::debug!("\n{:#010b}\n{}\n{:#066b}\n{:#066b}", *nt, tmpind, tmpu64, (rc_base(encode_base(*nt)) as u64));
//         tmpu64 <<= 2;
//         tmpu64 |= rc_base(encode_base(*nt)) as u64;
// //         log::debug!("{:#066b}", tmpu64);
//         if tmpind == 31 {
//             compseq.push(tmpu64);
//             tmpu64 = 0;
//             tmpind = 0;
//         } else {
//             tmpind += 1;
//         }
//     }
//
//     if tmpind != 0 {
//         compseq.push(tmpu64);
//     }
// }

#[cfg(target_family = "wasm")]
fn chunked_processing_wasm<IntT>(
    file1: &mut WebSysFile,
    file2: Option<&mut WebSysFile>,
    k: usize,
    qual: &QualOpts,
    outvec: &mut Vec<(u64, u64, u8)>,
    csize: usize,
    do_fit: bool,
) -> (
    HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
    HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
    Vec<u32>,
    u16,
)
where
    IntT: for<'a> UInt<'a>,
{
    post_state("preprocess:chunked:start");
    logw(
        "Getting kmers from first file. Creating reader...",
        Some("info"),
    );

    let mut outdict = HashMap::with_hasher(BuildHasherDefault::default());
    let mut minmaxdict = HashMap::with_hasher(BuildHasherDefault::default());
    let mut themap = HashMap::with_hasher(BuildHasherDefault::default());
    let mut countmap: HashMap<u64, (u32, u64, u8), BuildHasherDefault<NoHashHasher<u64>>> =
        HashMap::with_hasher(BuildHasherDefault::default());

    let mut reader = open_fastq(file1);
    let mut histovec: Vec<u32> = vec![0; MAXSIZEHISTO];

    logw("Entering while loop...", Some("info"));

    // todo: use the same counter and is_multiple_of
    let mut i_record = 0;
    let mut count: usize = 0;
    post_state("preprocess:chunked:loop:start");

    while let Some(record) = reader.next() {
        let seqrec = record.expect("Invalid FASTQ record");
        let rl = seqrec.seq().len();
        let kmer_opt = Kmer::<IntT>::new(
            std::borrow::Cow::Borrowed(seqrec.seq()),
            rl,
            Some(seqrec.qual()),
            k,
            qual.min_qual,
            true,
        );
        if let Some(mut kmer_it) = kmer_opt {
            let (hc, hnc, b, km) = kmer_it.get_curr_kmerhash_and_bases_and_kmer();
            outvec.push((hc, hnc, b));
            outdict.entry(hc).or_insert(km);
            minmaxdict.entry(hnc).or_insert(hc);
            while let Some(tmptuple) = kmer_it.get_next_kmer_and_give_us_things() {
                let (hc, hnc, b, km) = tmptuple;
                outvec.push((hc, hnc, b));
                outdict.entry(hc).or_insert(km);
                minmaxdict.entry(hnc).or_insert(hc);
            }
        }

        i_record += 1;
        count += 1;
        if i_record >= csize {
            // Processssssss! And reset.
            if !outvec.is_empty() {
                logw("Processing chunk. Sorting k-mers...", Some("debug"));
                outvec.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
                logw("k-mers sorted. Counting k-mers...", Some("debug"));
                // Then, do a counting of everything and save the results in a dictionary and return it

                update_countmap(outvec, &mut countmap);
            }

            // Reset
            outvec.clear();
            post_state(&format!("preprocess:chunked:loop:{:?}", count));
            i_record = 0;
        }
    }
    logw(
        "Finished getting kmers from first file. Starting with the second...",
        Some("info"),
    );

    post_state(&format!("preprocess:chunked:loop:{:?}:50", count));
    let percentageblock = (count as f64 / 10_f64) as usize;

    if let Some(file2) = file2 {
        let mut reader = open_fastq(file2);

        // Filling the seq of the second file!
        while let Some(record) = reader.next() {
            let seqrec = record.expect("Invalid FASTQ record");
            // put_these_nts_into_an_efficient_vector_rc(&seqrec.seq(), &mut theseq, (itrecord % 32) as u8);
            let rl = seqrec.seq().len();
            let kmer_opt = Kmer::<IntT>::new(
                std::borrow::Cow::Borrowed(seqrec.seq()),
                rl,
                Some(seqrec.qual()),
                k,
                qual.min_qual,
                true,
            );
            if let Some(mut kmer_it) = kmer_opt {
                let (hc, hnc, b, km) = kmer_it.get_curr_kmerhash_and_bases_and_kmer();
                outvec.push((hc, hnc, b));
                outdict.entry(hc).or_insert(km);
                minmaxdict.entry(hnc).or_insert(hc);
                while let Some(tmptuple) = kmer_it.get_next_kmer_and_give_us_things() {
                    let (hc, hnc, b, km) = tmptuple;
                    outvec.push((hc, hnc, b));
                    outdict.entry(hc).or_insert(km);
                    minmaxdict.entry(hnc).or_insert(hc);
                }
            }

            i_record += 1;
            count += 1;
            if i_record >= csize {
                // Processssssss! And reset.
                if !outvec.is_empty() {
                    logw("Processing chunk. Sorting k-mers...", Some("info"));
                    outvec.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
                    logw("k-mers sorted. Counting k-mers...", Some("info"));
                    // Then, do a counting of everything and save the results in a dictionary and return it

                    update_countmap(outvec, &mut countmap);
                }

                // Reset
                outvec.clear();
                i_record = 0;
            }

            // This might be done slightly more efficiently??
            if percentageblock > 0 && count.is_multiple_of(percentageblock) {
                post_state(&format!(
                    "preprocess:chunked:loop:{:?}:{:?}",
                    count,
                    count / percentageblock * 5
                ));
            }
        }

    }

    logw("Finished getting kmers from the input file(s)", Some("info"));

    // The residual chunk. This MUST stay outside the `if let Some(file2)` block above, or single-file
    // input silently loses the k-mers of its trailing partial chunk.
    if !outvec.is_empty() {
        logw("Processing last chunk. Sorting k-mers...", Some("info"));
        outvec.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
        logw("k-mers sorted. Counting k-mers...", Some("info"));
        update_countmap(outvec, &mut countmap);
        outvec.clear();
    }

    post_state("preprocess:chunked:loop:end");
    logw("Filtering...", Some("info"));

    // Now, get themap, histovec, and filter outdict and minmaxdict
    countmap.shrink_to_fit();
    let mut minc: u16 = qual.min_count;

    // This can be optimised. also better written: I had to repeat the code for the retains, to try to improve slightly the running time in
    // case no autofitting is requested. In any case, it could be improved in the future.
    if do_fit {
        post_state("preprocess:chunked:fitting");
        build_histogram_from_countmap(&countmap, &mut histovec);

        // Remove the last bin, as it might affect the fit, but we want it in the vector to plot it in case the coverage is really
        // large (and so that we can detect it).
        logw("Counting finished. Choosing the minimum count...", Some("info"));
        minc = choose_min_count(&histovec);
        logw(
            format!(
                "Minimum count chosen: {}. Starting filtering...",
                minc
            )
            .as_str(),
            Some("info"),
        );

        post_state("preprocess:chunked:filtering");
        drain_countmap_into_themap(
            &mut countmap,
            &mut themap,
            &mut outdict,
            &mut minmaxdict,
            minc,
            None,
        );
    } else {
        post_state("preprocess:chunked:filtering");
        drain_countmap_into_themap(
            &mut countmap,
            &mut themap,
            &mut outdict,
            &mut minmaxdict,
            minc,
            Some(&mut histovec),
        );
    }

    outdict.shrink_to_fit();
    minmaxdict.shrink_to_fit();
    histovec.shrink_to_fit();
    drop(countmap);

    (outdict, minmaxdict, themap, histovec, minc)
}

#[cfg(target_family = "wasm")]
fn bloom_filter_preprocessing_wasm<IntT>(
    file1: &mut WebSysFile,
    file2: Option<&mut WebSysFile>,
    k: usize,
    qual: &QualOpts,
    do_fit: bool,
) -> (
    HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
    HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
    Vec<u32>,
    u16,
)
where
    IntT: for<'a> UInt<'a>,
{
    post_state("preprocess:bloom:start");
    logw("Getting kmers from first file with Bloom filter. Creating reader and initialising filter...", Some("info"));

    let mut outdict = HashMap::with_hasher(BuildHasherDefault::default());
    let mut minmaxdict = HashMap::with_hasher(BuildHasherDefault::default());
    let mut themap = HashMap::with_hasher(BuildHasherDefault::default());

    let mut histovec: Vec<u32> = vec![0; MAXSIZEHISTO];

    let mut kmer_filter = KmerFilter::new(if do_fit { 3 } else { qual.min_count });
    kmer_filter.init();

    logw("Entering while loop for the first file...", Some("info"));
    post_state("preprocess:bloom:loop:start");

    let mut count: usize = 0;

    let mut reader = open_fastq(file1);
    while let Some(record) = reader.next() {
        let seqrec = record.expect("Invalid FASTQ record");
        let rl = seqrec.seq().len();
        let kmer_opt = Kmer::<IntT>::new(
            std::borrow::Cow::Borrowed(seqrec.seq()),
            rl,
            Some(seqrec.qual()),
            k,
            qual.min_qual,
            true,
        );
        if let Some(mut kmer_it) = kmer_opt {
            let (hc, hnc, b) = kmer_it.get_curr_hash_and_bases();
            if Ordering::is_eq(kmer_filter.filter(hc, hnc, b)) {
                outdict.entry(hc).or_insert_with(|| kmer_it.get_kmer());
                minmaxdict.entry(hnc).or_insert(hc);
            }
            while let Some((hc, hnc, b)) = kmer_it.get_next_hash_and_bases() {
                if Ordering::is_eq(kmer_filter.filter(hc, hnc, b)) {
                    outdict.entry(hc).or_insert_with(|| kmer_it.get_kmer());
                    minmaxdict.entry(hnc).or_insert(hc);
                }
            }
        }
        count += 1;
        if count.is_multiple_of(150000) {
            post_state(&format!("preprocess:bloom:loop:{:?}", count));
        }
    }
    logw(
        "Finished getting kmers from first file. Starting with the second...",
        Some("info"),
    );
    post_state(&format!("preprocess:bloom:loop:{:?}:50", count));
    let percentageblock = (count as f64 / 10_f64) as usize;

    if let Some(file2) = file2 {
        let mut reader = open_fastq(file2);

        // Filling the seq of the second file!
        while let Some(record) = reader.next() {
            let seqrec = record.expect("Invalid FASTQ record");
            // put_these_nts_into_an_efficient_vector_rc(&seqrec.seq(), &mut theseq, (itrecord % 32) as u8);
            let rl = seqrec.seq().len();
            let kmer_opt = Kmer::<IntT>::new(
                std::borrow::Cow::Borrowed(seqrec.seq()),
                rl,
                Some(seqrec.qual()),
                k,
                qual.min_qual,
                true,
            );
            if let Some(mut kmer_it) = kmer_opt {
                let (hc, hnc, b) = kmer_it.get_curr_hash_and_bases();
                if Ordering::is_eq(kmer_filter.filter(hc, hnc, b)) {
                    outdict.entry(hc).or_insert_with(|| kmer_it.get_kmer());
                    minmaxdict.entry(hnc).or_insert(hc);
                }
                while let Some((hc, hnc, b)) = kmer_it.get_next_hash_and_bases() {
                    if Ordering::is_eq(kmer_filter.filter(hc, hnc, b)) {
                        outdict.entry(hc).or_insert_with(|| kmer_it.get_kmer());
                        minmaxdict.entry(hnc).or_insert(hc);
                    }
                }
            }
            count += 1;

            // This might be done slightly more efficiently??
            if percentageblock > 0 && count.is_multiple_of(percentageblock) {
                post_state(&format!(
                    "preprocess:bloom:loop:{:?}:{:?}",
                    count,
                    count / percentageblock * 5
                ));
            }
        }
    }

    post_state("preprocess:bloom:loop:end");
    logw("Finished getting kmers from the second file", Some("info"));
    logw("Second part of filtering...", Some("info"));

    // // Now, get themap, histovec, and filter outdict and minmaxdict
    let mut countmap = kmer_filter.get_counts_map();
    countmap.shrink_to_fit();

    // This can be optimised. also better written: I had to repeat the code for the retains, to try to improve slightly the running time in
    // case no autofitting is requested. In any case, it could be improved in the future.
    let mut minc = qual.min_count;
    if do_fit {
        post_state("preprocess:bloom:fitting");
        build_histogram_from_countmap(&countmap, &mut histovec);

        // Remove the last bin, as it might affect the fit, but we want it in the vector to plot it in case the coverage is really
        // large (and so that we can detect it).
        logw("Counting finished. Choosing the minimum count...", Some("info"));
        minc = choose_min_count(&histovec);
        logw(
            format!(
                "Minimum count chosen: {}. Starting filtering...",
                minc
            )
            .as_str(),
            Some("info"),
        );

        post_state("preprocess:bloom:filtering");
        drain_countmap_into_themap(
            &mut countmap,
            &mut themap,
            &mut outdict,
            &mut minmaxdict,
            minc,
            None,
        );
    } else {
        post_state("preprocess:bloom:filtering");
        // I think this part can be improved: now that the min_count error has been corrected, some conditionals could be removed here?
        drain_countmap_into_themap(
            &mut countmap,
            &mut themap,
            &mut outdict,
            &mut minmaxdict,
            minc,
            Some(&mut histovec),
        );
    }

    outdict.shrink_to_fit();
    minmaxdict.shrink_to_fit();

    (outdict, minmaxdict, themap, histovec, minc)
}

/// Bloom-filter preprocessing. Its spectrum is **not** comparable to the exact counter's: there is no
/// count-1 bin, and false positives inflate the rest.
#[cfg(not(target_family = "wasm"))]
fn bloom_filter_preprocessing_standalone<IntT, I>(
    input_iters: &mut [I],
    k: usize,
    qual: &QualOpts,
    do_fit: bool,
    out_path: &mut Option<PathBuf>,
) -> (
    HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
    HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
    Vec<u32>,
    u16,
)
where
    IntT: for<'a> UInt<'a>,
    I: Iterator<Item=(Vec<u8>, Option<Vec<u8>>)>,
{
    log::info!("Initialising variables and filter...");

    let mut outdict = HashMap::with_hasher(BuildHasherDefault::default());
    let mut minmaxdict = HashMap::with_hasher(BuildHasherDefault::default());
    let mut themap = HashMap::with_hasher(BuildHasherDefault::default());

    let mut histovec: Vec<u32> = vec![0; MAXSIZEHISTO];

    let mut kmer_filter = KmerFilter::new(qual.min_count);
    kmer_filter.init();

    // NOTE, potential TODO? : This could be slightly improved by filling outdict and minmaxdict only once, though it'd require saving also km, but it could be better
    extract_kmers_from_files(input_iters, |seq, num_bases, qual_bytes| {
        let kmer_opt = Kmer::<IntT>::new(seq, num_bases, qual_bytes, k, qual.min_qual, true);
        if let Some(mut kmer_it) = kmer_opt {
            let (hc, hnc, b) = kmer_it.get_curr_hash_and_bases();
            if Ordering::is_eq(kmer_filter.filter(hc, hnc, b)) {
                outdict.entry(hc).or_insert_with(|| kmer_it.get_kmer());
                minmaxdict.entry(hnc).or_insert(hc);
            }
            while let Some((hc, hnc, b)) = kmer_it.get_next_hash_and_bases() {
                if Ordering::is_eq(kmer_filter.filter(hc, hnc, b)) {
                    outdict.entry(hc).or_insert_with(|| kmer_it.get_kmer());
                    minmaxdict.entry(hnc).or_insert(hc);
                }
            }
        }
    });
    log::info!("Finishing filtering...");

    // Now, get themap, histovec, and filter outdict and minmaxdict
    let mut countmap = kmer_filter.get_counts_map();
    countmap.shrink_to_fit();
    let minc;

    // This can be optimised. also better written: I had to repeat the code for the retains, to try to improve slightly the running time in
    // case no autofitting is requested. In any case, it could be improved in the future.
    if do_fit {
        build_histogram_from_countmap(&countmap, &mut histovec);

        // Remove the last bin, as it might affect the fit, but we want it in the vector to plot it in case the coverage is really
        // large (and so that we can detect it).
        log::info!("Choosing the minimum count...");
        minc = choose_min_count(&histovec);
        log::info!(
            "Minimum count chosen: {}. Filtering k-mers...",
            minc
        );

        drain_countmap_into_themap(
            &mut countmap,
            &mut themap,
            &mut outdict,
            &mut minmaxdict,
            minc,
            None,
        );
    } else {
        log::info!("Filtering k-mers...");
        minc = qual.min_count;
        drain_countmap_into_themap(
            &mut countmap,
            &mut themap,
            &mut outdict,
            &mut minmaxdict,
            minc,
            Some(&mut histovec),
        );
    }

    outdict.shrink_to_fit();
    minmaxdict.shrink_to_fit();

    if let Some(p) = out_path {
        plot_kmer_histogram(&histovec, p.as_path());
    }

    histovec.shrink_to_fit();
    (outdict, minmaxdict, themap, histovec, minc)
}

fn update_countmap(
    invec: &[(u64, u64, u8)],
    countmap: &mut HashMap<u64, (u32, u64, u8), BuildHasherDefault<NoHashHasher<u64>>>,
) {
    let mut i = 0;
    let mut c: u32 = 0;
    let mut tmphash = invec[i].0;
    // let mut tmpcounter = 0;

    while i < invec.len() {
        if tmphash != invec[i].0 {
            let tmpref = countmap
                .entry(tmphash)
                .or_insert((0, invec[i - 1].1, invec[i - 1].2));
            tmpref.0 = tmpref.0.saturating_add(c);

            tmphash = invec[i].0;
            c = 1;
        } else {
            c = c.saturating_add(1);
        }
        i += 1;
    }

    let tmpref = countmap
        .entry(tmphash)
        .or_insert((0, invec[i - 1].1, invec[i - 1].2));
    tmpref.0 = tmpref.0.saturating_add(c);
}

/// Hash one batch of records at a single k, in parallel. The occurrence order fixes the dictionary
/// insertion order and so the node numbering in the GFA/DOT dumps; contigs do not depend on it.
#[cfg(not(target_family = "wasm"))]
fn hash_batch<IntT>(batch: &[OwnedRecord], k: usize, min_qual: u8) -> Vec<(u64, u64, u8, IntT)>
where
    IntT: for<'a> UInt<'a>,
{
    batch
        .par_iter()
        .flat_map_iter(|(seq, qual_bytes)| {
            let mut local = Vec::new();
            let kmer_opt = Kmer::<IntT>::new(
                std::borrow::Cow::Borrowed(seq),
                seq.len(),
                qual_bytes.as_deref(),
                k,
                min_qual,
                true,
            );
            if let Some(mut kmer_it) = kmer_opt {
                local.push(kmer_it.get_curr_kmerhash_and_bases_and_kmer());
                while let Some(tup) = kmer_it.get_next_kmer_and_give_us_things() {
                    local.push(tup);
                }
            }
            local
        })
        .collect()
}

/// Buffer k-mer occurrences, sort them so equal k-mers become adjacent, and run-length count them.
/// `csize` is the memory knob: records buffered before a flush, with `usize::MAX` meaning no chunking.
#[cfg(not(target_family = "wasm"))]
#[allow(clippy::too_many_arguments)]
fn chunked_preprocessing_standalone<IntT, I>(
    input_iters: &mut [I],
    k: usize,
    qual: &QualOpts,
    outvec: &mut Vec<(u64, u64, u8)>,
    csize: usize,
    do_fit: bool,
    out_path: &mut Option<PathBuf>,
) -> (
    HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
    HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
    Vec<u32>,
    u16,
)
where
    IntT: for<'a> UInt<'a>,
    I: Iterator<Item=(Vec<u8>, Option<Vec<u8>>)>,
{
    log::info!("Getting kmers from files. Creating reader...");

    let mut outdict = HashMap::with_hasher(BuildHasherDefault::default());
    let mut minmaxdict = HashMap::with_hasher(BuildHasherDefault::default());
    let mut countmap: HashMap<u64, (u32, u64, u8), BuildHasherDefault<NoHashHasher<u64>>> =
        HashMap::with_hasher(BuildHasherDefault::default());

    let histovec: Vec<u32> = vec![0; MAXSIZEHISTO];
    let mut i_record = 0;
    // let mut ncols : usize = 0;

    // The k-mer work runs in parallel per batch, with the dictionary probes kept out of the hot loop.
    // A chunk therefore closes at the first batch boundary at or past `csize`: a hint, not a contract.
    extract_kmers_from_files_batched(input_iters, BATCH_RECORDS, |batch| {
        let items: Vec<(u64, u64, u8, IntT)> = hash_batch::<IntT>(batch, k, qual.min_qual);

        outvec.extend(items.iter().map(|&(hc, hnc, b, _)| (hc, hnc, b)));

        // `or_insert` keeps the first writer, but every occurrence of a given hash is the same k-mer,
        // so which one wins is immaterial — the batch order does not affect the result.
        for (hc, hnc, _, km) in items {
            outdict.entry(hc).or_insert(km);
            minmaxdict.entry(hnc).or_insert(hc);
        }

        i_record += batch.len();
        if i_record >= csize {
            if !outvec.is_empty() {
                log::debug!("Processing chunk. Sorting k-mers...");
                outvec.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
                log::debug!("k-mers sorted. Counting k-mers...");
                update_countmap(outvec, &mut countmap);
            }
            outvec.clear();
            i_record = 0;
        }
    });

    // The residual chunk. Sits outside the extraction closure so it runs whatever the input was.
    if !outvec.is_empty() {
        log::info!("Processing last chunk. Sorting k-mers...");
        outvec.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
        log::info!("k-mers sorted. Counting k-mers...");
        update_countmap(outvec, &mut countmap);
        outvec.clear();
    }

    finish_sort_counter(
        countmap, outdict, minmaxdict, histovec, qual, do_fit, out_path,
    )
}

/// Fit, filter and plot a finished sort-counter, whatever drove it.
#[cfg(not(target_family = "wasm"))]
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
fn finish_sort_counter<IntT>(
    mut countmap: HashMap<u64, (u32, u64, u8), BuildHasherDefault<NoHashHasher<u64>>>,
    mut outdict: HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    mut minmaxdict: HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
    mut histovec: Vec<u32>,
    qual: &QualOpts,
    do_fit: bool,
    out_path: &mut Option<PathBuf>,
) -> (
    HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
    HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
    Vec<u32>,
    u16,
)
where
    IntT: for<'a> UInt<'a>,
{
    log::info!("Filtering...");

    let mut themap = HashMap::with_hasher(BuildHasherDefault::default());

    // Now, get themap, histovec, and filter outdict and minmaxdict
    countmap.shrink_to_fit();
    let minc;

    // The two branches differ only in when the histogram is built: fitting needs it up front, so it is
    // built first and the drain then skips it; without a fit the drain builds it as it goes.
    if do_fit {
        build_histogram_from_countmap(&countmap, &mut histovec);

        log::info!("Counting finished. Choosing the minimum count...");
        minc = choose_min_count(&histovec);
        log::info!(
            "Minimum count chosen: {}. Starting filtering...",
            minc
        );

        drain_countmap_into_themap(
            &mut countmap,
            &mut themap,
            &mut outdict,
            &mut minmaxdict,
            minc,
            None,
        );
    } else {
        minc = qual.min_count;
        drain_countmap_into_themap(
            &mut countmap,
            &mut themap,
            &mut outdict,
            &mut minmaxdict,
            minc,
            Some(&mut histovec),
        );
    }

    drop(countmap);
    outdict.shrink_to_fit();
    minmaxdict.shrink_to_fit();

    if let Some(p) = out_path {
        plot_kmer_histogram(&histovec, p.as_path());
    }

    histovec.shrink_to_fit();
    (outdict, minmaxdict, themap, histovec, minc)
}

/// Read fastq files, get the reads, get the k-mers, count them, filter them by count, and get some way of recovering the sequence later.
#[cfg(not(target_family = "wasm"))]
pub fn preprocessing_standalone<IntT, I>(
    input_iters: &mut [I],
    k: usize,
    qual: &QualOpts,
    timevec: &mut Option<&mut Vec<Instant>>,
    out_path: &mut Option<PathBuf>,
    csize: usize,
    do_bloom: bool,
    do_fit: bool,
    estimated_kmers: Option<usize>,
) -> PreprocessedK<IntT>
where
    IntT: for<'a> UInt<'a>,
    I: Iterator<Item=(Vec<u8>, Option<Vec<u8>>)>,
{
    log::info!("Starting preprocessing_standalone with k = {k}");

    let (thedict, maxmindict, themap, histovec, used_min_count) = if do_bloom {
        log::info!("Processing using a Bloom filter");
        bloom_filter_preprocessing_standalone::<IntT, _>(input_iters, k, qual, do_fit, out_path)
    } else {
        // "No chunking" is one unbounded chunk. The guard matters: `i_record >= 0` holds on every
        // record, so passing 0 through would sort and count after every single read.
        let csize = if csize == 0 { usize::MAX } else { csize };
        if csize == usize::MAX {
            log::info!("Counting k-mers by sorting, without chunking");
        } else {
            log::info!("Counting k-mers by sorting, in chunks of {csize} records");
        }

        let mut tmpvec: Vec<(u64, u64, u8)> = Vec::with_capacity(estimated_kmers.unwrap_or(200000_usize));
        let out = chunked_preprocessing_standalone::<IntT, _>(
            input_iters, k, qual, &mut tmpvec, csize, do_fit, out_path,
        );
        drop(tmpvec);
        out
    };

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

    PreprocessedK {
        k,
        themap,
        thedict,
        maxmindict,
        histovec,
        used_min_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nohash_hasher::NoHashHasher;
    use std::{collections::HashMap, hash::BuildHasherDefault};

    fn empty_countmap() -> HashMap<u64, (u32, u64, u8), BuildHasherDefault<NoHashHasher<u64>>> {
        HashMap::with_hasher(BuildHasherDefault::default())
    }

    fn empty_themap() -> HashMap<u64, crate::HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>
    {
        HashMap::with_hasher(BuildHasherDefault::default())
    }

    fn empty_dict() -> HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>> {
        HashMap::with_hasher(BuildHasherDefault::default())
    }

    #[test]
    fn update_countmap_single_entry() {
        let input = vec![(100u64, 200u64, 1u8)];
        let mut cmap = empty_countmap();
        update_countmap(&input, &mut cmap);
        assert_eq!(cmap[&100].0, 1);
        assert_eq!(cmap[&100].1, 200u64);
        assert_eq!(cmap[&100].2, 1u8);
    }

    #[test]
    fn update_countmap_two_same_hash() {
        let input = vec![(100u64, 200u64, 1u8), (100u64, 200u64, 1u8)];
        let mut cmap = empty_countmap();
        update_countmap(&input, &mut cmap);
        assert_eq!(cmap[&100].0, 2);
    }

    #[test]
    fn update_countmap_two_different_hashes() {
        let input = vec![(100u64, 200u64, 1u8), (200u64, 100u64, 2u8)];
        let mut cmap = empty_countmap();
        update_countmap(&input, &mut cmap);
        assert_eq!(cmap[&100].0, 1);
        assert_eq!(cmap[&200].0, 1);
    }

    #[test]
    fn update_countmap_accumulates_across_calls() {
        // Two calls: first adds 2, second adds 1 → total 3
        let input1 = vec![(42u64, 0u64, 0u8), (42u64, 0u64, 0u8)];
        let input2 = vec![(42u64, 0u64, 0u8)];
        let mut cmap = empty_countmap();
        update_countmap(&input1, &mut cmap);
        update_countmap(&input2, &mut cmap);
        assert_eq!(cmap[&42].0, 3);
    }

    #[test]
    fn update_countmap_run_of_five() {
        let input: Vec<_> = (0..5).map(|_| (7u64, 8u64, 0u8)).collect();
        let mut cmap = empty_countmap();
        update_countmap(&input, &mut cmap);
        assert_eq!(cmap[&7].0, 5);
    }

    #[test]
    fn drain_countmap_above_threshold_included() {
        let mut cmap = empty_countmap();
        cmap.insert(1u64, (5u32, 2u64, 0u8)); // count=5 >= minc=3 → in themap
        let mut themap = empty_themap();
        let mut outdict = empty_dict();
        let mut minmaxdict = empty_dict();
        drain_countmap_into_themap::<u64>(
            &mut cmap,
            &mut themap,
            &mut outdict,
            &mut minmaxdict,
            3,
            None,
        );
        assert!(themap.contains_key(&1));
    }

    #[test]
    fn drain_countmap_below_threshold_excluded() {
        let mut cmap = empty_countmap();
        cmap.insert(2u64, (2u32, 3u64, 0u8)); // count=2 < minc=3 → removed from outdict
        let mut themap = empty_themap();
        let mut outdict = empty_dict();
        outdict.insert(2u64, 99u64); // should be removed
        let mut minmaxdict = empty_dict();
        minmaxdict.insert(3u64, 2u64); // hnc → hc, should be removed
        drain_countmap_into_themap::<u64>(
            &mut cmap,
            &mut themap,
            &mut outdict,
            &mut minmaxdict,
            3,
            None,
        );
        assert!(!themap.contains_key(&2));
        assert!(!outdict.contains_key(&2));
        assert!(!minmaxdict.contains_key(&3));
    }

    #[test]
    fn drain_countmap_boundary_equal_minc() {
        // count == minc → included (>= check)
        let mut cmap = empty_countmap();
        cmap.insert(5u64, (3u32, 0u64, 0u8));
        let mut themap = empty_themap();
        let mut outdict = empty_dict();
        let mut minmaxdict = empty_dict();
        drain_countmap_into_themap::<u64>(
            &mut cmap,
            &mut themap,
            &mut outdict,
            &mut minmaxdict,
            3,
            None,
        );
        assert!(themap.contains_key(&5));
    }

    #[test]
    fn drain_countmap_with_histogram() {
        let mut cmap = empty_countmap();
        cmap.insert(1u64, (5u32, 0u64, 0u8)); // count=5 → histovec[4]
        cmap.insert(2u64, (10u32, 0u64, 0u8)); // count=10 → histovec[9]
        let mut themap = empty_themap();
        let mut outdict = empty_dict();
        let mut minmaxdict = empty_dict();
        let mut histovec = vec![0u32; MAXSIZEHISTO];
        drain_countmap_into_themap::<u64>(
            &mut cmap,
            &mut themap,
            &mut outdict,
            &mut minmaxdict,
            1,
            Some(&mut histovec),
        );
        assert!(histovec[4] > 0, "count=5 should be at histovec[4]");
        assert!(histovec[9] > 0, "count=10 should be at histovec[9]");
    }

    /// A bimodal spectrum: a tall error peak at count 1-2 and the real single-copy peak further out.
    /// `coverage_peak` must return the *count* of the second peak, ignoring the first.
    #[test]
    fn coverage_peak_finds_the_mode_above_the_error_peak() {
        for expected in [3usize, 10, 30, 113, 498] {
            let mut h = vec![0u32; MAXSIZEHISTO];
            h[0] = 1_000_000; // count 1: errors, must be ignored
            h[1] = 200_000; // count 2: still errors
            h[expected - 1] = 50_000; // the genuine single-copy peak
            assert_eq!(
                coverage_peak(&h),
                expected,
                "peak planted at count {expected}"
            );
        }
    }

    /// Ties keep the lowest count, and a spectrum with nothing above the error peak falls back to 2 —
    /// so the peak-derived floor stays well defined even for a degenerate histogram.
    #[test]
    fn coverage_peak_is_deterministic_and_has_a_floor() {
        let mut h = vec![0u32; MAXSIZEHISTO];
        h[9] = 42;
        h[19] = 42; // equal height: the lower count wins
        assert_eq!(coverage_peak(&h), 10);

        let mut only_errors = vec![0u32; MAXSIZEHISTO];
        only_errors[0] = 99;
        only_errors[1] = 7;
        assert_eq!(coverage_peak(&only_errors), 2);
    }


    /// `coverage_peak` stops one bin short of [`LEGACY_HISTO_RANGE`], and `fit_histogram` is handed the
    /// same window, so neither can be dragged by whatever sits at the edge of it.
    #[test]
    fn coverage_peak_ignores_the_edge_of_its_pinned_window() {
        let mut h = vec![0u32; MAXSIZEHISTO];
        h[LEGACY_HISTO_RANGE - 1] = 1_000_000;
        h[29] = 10;
        assert_eq!(coverage_peak(&h), 30);
    }

    /// Widening the histogram must not move the legacy readers: they see counts 1..=499, and
    /// `add_to_histogram` puts count `c` at index `c-1` regardless of how long the vector is.
    #[test]
    fn widening_the_histogram_leaves_the_pinned_window_alone() {
        let mut narrow = vec![0u32; LEGACY_HISTO_RANGE];
        let mut wide = vec![0u32; MAXSIZEHISTO];
        for c in [1u32, 2, 3, 47, 163, 499] {
            add_to_histogram(&mut narrow[..], c);
            add_to_histogram(&mut wide[..], c);
        }
        // Counts below the old ceiling land identically; only what used to saturate now moves.
        assert_eq!(narrow[..LEGACY_HISTO_RANGE - 1], wide[..LEGACY_HISTO_RANGE - 1]);
        assert_eq!(coverage_peak(&narrow), coverage_peak(&wide));
    }

    // ---------------------------------------------------------------------------------------------
    // The trough estimator. Spectra are built from the same model the plan was simulated with: a
    // genome lobe at Poisson(lambda) over 4.6 Mb, errors at Poisson(lambda * p / 3) over 3k slots.
    // ---------------------------------------------------------------------------------------------

    fn poisson(k: usize, lam: f64) -> f64 {
        (-lam + k as f64 * lam.ln() - libm::lgamma(k as f64 + 1.0)).exp()
    }

    /// A synthetic spectrum at k-mer coverage `lam`, with a 1 % per-base error rate at k = 51.
    fn synthetic_spectrum(lam: f64) -> Vec<u32> {
        const GENOME: f64 = 4_600_000.0;
        const K: f64 = 51.0;
        let mut h = vec![0u32; MAXSIZEHISTO];
        let lam_err = lam * 0.01 / 3.0;
        let err_slots = GENOME * 3.0 * K;
        for c in 1..MAXSIZEHISTO {
            let n = GENOME * poisson(c, lam) + if c < 60 { err_slots * poisson(c, lam_err) } else { 0.0 };
            h[c - 1] = n as u32;
        }
        h
    }

    /// Fraction of a Poisson(`lam`) genome deleted by cutting below `min_count`.
    fn genome_loss(lam: f64, min_count: u16) -> f64 {
        (0..min_count as usize).map(|c| poisson(c, lam)).sum()
    }

    /// The case that breaks today: at 500x the error lobe outvotes the genome lobe, so the global
    /// argmax returns ~3 and `peak/8` yields 2, while the trough finds the real peak out at ~500.
    #[test]
    fn trough_estimator_survives_a_deep_library() {
        let h = synthetic_spectrum(500.0);
        assert!(
            coverage_peak(&h) < 10,
            "the old argmax should be fooled here; that is the bug being fixed"
        );
        let e = estimate_by_trough(&h);
        assert_eq!(e.verdict, "ok");
        assert!((450..550).contains(&e.peak), "peak was {}", e.peak);
        assert!((10..40).contains(&e.min_count), "min_count was {}", e.min_count);
        assert!(genome_loss(500.0, e.min_count) < 1e-6);
    }

    /// On a smooth spectrum the walk stops at the valley and widening the search cannot move it. This
    /// does *not* hold on real data: where the error tail is noisy the walk stops at the first bump
    /// and the spectrum keeps dipping afterwards, which is the case this change exists to correct.
    #[test]
    fn the_valley_equals_the_walk_on_a_clean_spectrum() {
        for lam in [20.0, 60.0, 150.0, 500.0] {
            let h = synthetic_spectrum(lam);
            let seed = error_trough(&h).expect("a clean spectrum turns back up");
            let peak = peak_above(&h, seed);
            assert_eq!(
                error_valley(&h, peak),
                seed,
                "lam {lam}: valley moved (seed {seed}, peak {peak})"
            );
        }
    }

    #[test]
    fn a_seed_past_the_genome_lobe_still_finds_the_valley() {
        let h = synthetic_spectrum(500.0);
        let truth = error_trough(&h).unwrap();
        for overshoot in [700, 1500, 4000] {
            assert_eq!(
                error_valley(&h, overshoot),
                truth,
                "a peak of {overshoot} moved the valley away from {truth}"
            );
        }
    }

    /// A library with no separable lobe must still bail: widening the search must not manufacture a
    /// valley where the spectrum is one monotone slide.
    #[test]
    fn a_flat_library_still_bails() {
        let h = synthetic_spectrum(4.0);
        let e = estimate_by_trough(&h);
        assert_ne!(e.verdict, "ok", "min_count was {}", e.min_count);
        assert_eq!(e.min_count, UNRESOLVED_MINCOUNT);
    }

    /// A deep library is mostly error k-mers by *count* and mostly genome by *sequence*. Weighing
    /// distinct k-mers instead of instances would reject this healthy spectrum.
    #[test]
    fn a_deep_library_is_not_rejected_for_being_mostly_errors() {
        let h = synthetic_spectrum(500.0);
        let distinct_total: u64 = h.iter().map(|&n| n as u64).sum();
        let distinct_genome: u64 = h[100..].iter().map(|&n| n as u64).sum();
        assert!(
            distinct_genome * 50 < distinct_total,
            "the genome should be a small minority of distinct k-mers here"
        );
        assert_eq!(estimate_by_trough(&h).verdict, "ok");
    }

    /// Where the lobes merge there is no honest answer, so it must say so rather than invent a trough
    /// in the empty tail — which at 3-5x deleted almost the whole genome.
    #[test]
    fn trough_estimator_bails_out_when_the_lobes_merge() {
        for lam in [3.0, 5.0, 8.0] {
            let e = estimate_by_trough(&synthetic_spectrum(lam));
            assert_ne!(e.verdict, "ok", "lambda {lam} should not be resolved");
            assert_eq!(e.min_count, UNRESOLVED_MINCOUNT, "lambda {lam}");
        }
    }

    /// Through the working range the cutoff must clear the errors without eating the genome.
    #[test]
    fn trough_estimator_is_safe_through_the_working_range() {
        for lam in [10.0, 15.0, 20.0, 50.0, 100.0, 250.0] {
            let e = estimate_by_trough(&synthetic_spectrum(lam));
            assert_eq!(e.verdict, "ok", "lambda {lam}");
            assert!(e.min_count >= 2, "lambda {lam}");
            assert!(
                genome_loss(lam, e.min_count) < 0.01,
                "lambda {lam} lost {:.2} % of the genome at min_count {}",
                genome_loss(lam, e.min_count) * 100.0,
                e.min_count
            );
        }
    }

    /// The reason for widening the histogram: a peak past the old 500-bin ceiling must still be found.
    #[test]
    fn trough_estimator_finds_a_peak_beyond_the_old_ceiling() {
        let mut h = vec![0u32; MAXSIZEHISTO];
        for c in 1..40 {
            h[c - 1] = 1_000_000 / (c as u32 * c as u32); // a decaying error lobe
        }
        for c in 5500..6500 {
            h[c - 1] = 20_000; // a genome lobe far outside LEGACY_HISTO_RANGE
        }
        let e = estimate_by_trough(&h);
        assert_eq!(e.verdict, "ok");
        assert!(e.peak >= LEGACY_HISTO_RANGE, "peak was {}", e.peak);
        assert!(coverage_peak(&h) < LEGACY_HISTO_RANGE);
    }

    /// What cutting at `m` costs, as a fraction of the k-mers above `trough` — the quantity the guard
    /// budgets, measured the same way from the same histogram.
    fn measured_loss(histovec: &[u32], trough: usize, m: usize) -> f64 {
        let total: f64 = (trough..histovec.len())
            .map(|c| histovec[c - 1] as f64)
            .sum();
        let cut: f64 = (trough..m).map(|c| histovec[c - 1] as f64).sum();
        if total > 0.0 {
            cut / total
        } else {
            0.0
        }
    }

    /// The measured half of the guard must spend its whole budget and no more: one bin higher has to
    /// break it.
    #[test]
    fn measured_guard_spends_its_budget_and_stops() {
        for lam in [20.0, 50.0, 100.0] {
            let h = synthetic_spectrum(lam);
            let trough = error_trough(&h).unwrap();
            let peak = peak_above(&h, trough);
            let m = measured_guard(&h, trough, peak) as usize;
            assert!(m >= 2, "lambda {lam}");
            assert!(
                measured_loss(&h, trough, m) <= MAX_GENOME_LOSS,
                "lambda {lam} guard {m} cost {:.4}",
                measured_loss(&h, trough, m)
            );
            assert!(
                m + 1 >= peak || measured_loss(&h, trough, m + 1) > MAX_GENOME_LOSS,
                "lambda {lam} guard {m} could have gone higher"
            );
        }
    }

    /// Neither half is safe alone, so the guard takes the tighter: the Poisson bound binds at low
    /// coverage, where errors above the trough loosen the measured one, and the measured bound binds at
    /// depth, where the lobe is far too wide for a Poisson tail.
    #[test]
    fn loss_guard_takes_the_tighter_of_its_two_bounds() {
        // Low coverage: the Poisson bound is the strict one.
        let h = synthetic_spectrum(10.0);
        let trough = error_trough(&h).unwrap();
        let peak = peak_above(&h, trough);
        assert!(poisson_guard(peak) < measured_guard(&h, trough, peak));
        assert_eq!(loss_guard(&h, trough, peak), poisson_guard(peak));

        // Depth, with a lobe as wide as the real libraries: the measured bound is the strict one.
        let mut wide = vec![0u32; MAXSIZEHISTO];
        for c in 1..MAXSIZEHISTO {
            let z = (c as f64 - 200.0) / 45.0;
            wide[c - 1] = (1_000_000.0 * (-0.5 * z * z).exp()) as u32;
        }
        assert!(measured_guard(&wide, 60, 200) < poisson_guard(200));
        assert_eq!(loss_guard(&wide, 60, 200), measured_guard(&wide, 60, 200));
    }

    /// The reason the Poisson tail was dropped: on a lobe as wide as the real ones it permits a cutoff
    /// far above what the data can afford. Mean 200, sd ~45, against a Poisson sd of 14.
    #[test]
    fn loss_guard_is_tighter_than_a_poisson_tail_on_an_overdispersed_lobe() {
        let mut h = vec![0u32; MAXSIZEHISTO];
        let (mu, sd) = (200.0f64, 45.0f64);
        for c in 1..MAXSIZEHISTO {
            let z = (c as f64 - mu) / sd;
            h[c - 1] = (1_000_000.0 * (-0.5 * z * z).exp()) as u32;
        }
        let trough = 60;
        let m = loss_guard(&h, trough, 200) as usize;
        // A Poisson(200) tail would allow ~168; the measured 1 % quantile of this lobe is far lower.
        assert!(m < 150, "guard returned {m}, no tighter than a Poisson tail");
        assert!(
            measured_loss(&h, trough, m) <= MAX_GENOME_LOSS,
            "guard {m} cost {:.4}",
            measured_loss(&h, trough, m)
        );
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

    /// The swap: on a deep library the trough and the mixture fit disagree, and it is the trough that
    /// must come out of `choose_min_count`.
    #[test]
    fn choose_min_count_returns_the_trough_not_the_fit() {
        let h = synthetic_spectrum(500.0);
        let e = estimate_by_trough(&h);
        assert_eq!(e.verdict, "ok");
        assert_eq!(choose_min_count(&h), e.min_count);
        // ...and that is emphatically not what the old path would have produced.
        let floor = ((coverage_peak(&h) as f64 / 8.0).round() as u16).max(2);
        assert!(
            e.min_count > floor,
            "the old floor was {floor} and the trough {}; the swap changes nothing here",
            e.min_count
        );
    }
}

#[cfg(target_family = "wasm")]
/// Main preprocessing function for wasm
pub fn preprocessing_wasm<IntT>(
    file1: &mut WebSysFile,
    file2: Option<&mut WebSysFile>,
    k: usize,
    qual: &QualOpts,
    csize: usize,
    do_bloom: bool,
    do_fit: bool,
) -> (
    HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
    Option<HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>>,
    HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
    Vec<u32>,
    u16,
)
where
    IntT: for<'a> UInt<'a>,
{
    if do_bloom {
        // Build indexes
        logw("Processing using a Bloom filter", Some("info"));

        let (thedict, maxmindict, themap, histovec, used_min_count) =
            bloom_filter_preprocessing_wasm::<IntT>(file1, file2, k, qual, do_fit);
        (themap, Some(thedict), maxmindict, histovec, used_min_count)
    } else {
        // "No chunking" is one unbounded chunk. The guard matters: `i_record >= 0` holds on every
        // record, so passing 0 through would sort and count after every single read.
        let csize = if csize == 0 { usize::MAX } else { csize };
        if csize == usize::MAX {
            logw("Counting k-mers by sorting, without chunking", Some("info"));
        } else {
            logw(
                format!("Counting k-mers by sorting, in chunks of {csize} records").as_str(),
                Some("info"),
            );
        }

        let mut tmpvec: Vec<(u64, u64, u8)> = Vec::new();
        let (thedict, maxmindict, themap, mut histovec, used_min_count) =
            chunked_processing_wasm::<IntT>(
                file1,
                file2,
                k,
                qual,
                &mut tmpvec,
                csize,
                do_fit,
            );
        drop(tmpvec);
        histovec.shrink_to_fit();
        (themap, Some(thedict), maxmindict, histovec, used_min_count)
    }
}
