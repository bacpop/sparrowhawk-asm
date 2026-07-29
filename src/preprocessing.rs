//! Some docs should be here

#[cfg(not(target_family = "wasm"))]
use std::{path::PathBuf, time::Instant};

use nohash_hasher::NoHashHasher;
use std::{cmp::Ordering, collections::HashMap, hash::BuildHasherDefault};

use rayon::prelude::*;

#[cfg(not(target_family = "wasm"))]
use needletail::parse_fastx_file;

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
    /// 500-bin k-mer spectrum; index `c-1` holds the number of distinct k-mers seen `c` times
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
const MAXSIZEHISTO: usize = 500;

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
/// Scans counts 3..=499 and returns a **count, not an index**; the saturating final bin is excluded, so
/// above ~500x the true peak is invisible here.
fn coverage_peak(histovec: &[u32]) -> usize {
    let mut best_count = 2usize; // nothing above the error peak; the caller's floor of 2 then applies
    let mut best_n = 0u32;
    for (i, &n) in histovec[2..(MAXSIZEHISTO - 1)].iter().enumerate() {
        if n > best_n {
            // Strict, so ties keep the lowest count and the result is deterministic.
            best_n = n;
            best_count = i + 3; // slice index 0 is count 3
        }
    }
    best_count
}

/// A fitted cutoff at or below this is treated as unreliable and replaced by the histogram floor.
/// Measured cutoffs split cleanly into a trustworthy group (14-52) and an untrustworthy one (2-8).
const TRUST_FIT_ABOVE: usize = 10;


/// Choose the minimum k-mer count from the spectrum. The returned value is an **inclusive** minimum:
/// both filter sites keep k-mers with `count >= min_count`.
fn apply_spectrum_fit(histovec: &[u32]) -> u16 {
    // Used whenever the fit is not trusted. Scaling off the peak keeps a deep library from collapsing
    // to a near-useless 2 or 3; the divisor errs low because a surviving error k-mer fragments a contig.
    let peak = coverage_peak(histovec);
    let floor = ((peak as f64 / 8.0).round() as u16).max(2);

    let mut fit = SpectrumFitter::new();
    match fit.fit_histogram(histovec[..(MAXSIZEHISTO - 1)].to_vec()) {
        // A large cutoff means the fit separated the true-k-mer component cleanly, so trust it.
        Ok(minc) if minc > TRUST_FIT_ABOVE => minc as u16,
        // Below that, a fitted 5/6/7/8 costs up to 400 kb of assembly and halves N50, so use the floor.
        outcome => {
            let why = match &outcome {
                Ok(minc) => format!("returned {minc}, too small to be reliable"),
                Err(e) => format!("did not converge ({e})"),
            };
            logw(
                &format!(
                    "The k-mer spectrum fit {why}. Falling back to the histogram: its peak is at count \
                     {peak}, so a minimum count of {floor} will be used This usually \
                     means the spectrum is thin — low coverage, or a large k. Check the k-mer spectrum \
                     histogram to confirm the value is appropriate."
                ),
                Some("warn"),
            );
            floor
        }
    }
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
fn extract_kmers_from_files<F>(files: &[String], mut on_record: F)
where
    F: FnMut(std::borrow::Cow<'_, [u8]>, usize, Option<&[u8]>),
{
    for file in files {
        log::info!("Getting kmers from file {file}. Creating reader...");
        let mut reader =
            parse_fastx_file(file).unwrap_or_else(|_| panic!("Invalid path/file: {file}"));
        log::info!("Parsing...");
        while let Some(record) = reader.next() {
            let seqrec = record.expect("Invalid FASTQ record");
            on_record(seqrec.seq(), seqrec.num_bases(), seqrec.qual());
        }
        log::info!("Finished getting kmers from file {file}.");
    }
    log::info!("Finished getting kmers from {} file(s)", files.len());
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
fn extract_kmers_from_files_batched<F>(files: &[String], batch_records: usize, mut on_batch: F)
where
    F: FnMut(&[OwnedRecord]),
{
    let mut batch: Vec<OwnedRecord> = Vec::with_capacity(batch_records);
    for file in files {
        log::info!("Getting kmers from file {file}. Creating reader...");
        let mut reader =
            parse_fastx_file(file).unwrap_or_else(|_| panic!("Invalid path/file: {file}"));
        log::info!("Parsing...");
        while let Some(record) = reader.next() {
            let seqrec = record.expect("Invalid FASTQ record");
            batch.push((seqrec.seq().into_owned(), seqrec.qual().map(|q| q.to_vec())));
            if batch.len() == batch_records {
                on_batch(&batch);
                batch.clear();
            }
        }
        log::info!("Finished getting kmers from file {file}.");
    }
    if !batch.is_empty() {
        on_batch(&batch);
    }
    log::info!("Finished getting kmers from {} file(s)", files.len());
}

#[cfg(not(target_family = "wasm"))]
fn plot_kmer_histogram(histovec: &[u32], out_path: &std::path::Path) {
    let backend = BitMapBackend::new(out_path, (1280, 960));
    let root = backend.into_drawing_area();
    let _ = root.fill(&WHITE);
    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(35)
        .y_label_area_size(40)
        .margin(5)
        .caption("k-mer spectrum", ("ibm-plex-sans", 30.0))
        .build_cartesian_2d(
            (0u32..(MAXSIZEHISTO as u32)).into_segmented(),
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
                .data(histovec.iter().enumerate().map(|(i, x)| (i as u32, *x))),
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
        logw("Counting finished. Starting fit...", Some("info"));
        minc = apply_spectrum_fit(&histovec);
        logw(
            format!(
                "Fit done! Fitted min_count value: {}. Starting filtering...",
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
        logw("Counting finished. Starting fit...", Some("info"));
        minc = apply_spectrum_fit(&histovec);
        logw(
            format!(
                "Fit done! Fitted min_count value: {}. Starting filtering...",
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
fn bloom_filter_preprocessing_standalone<IntT>(
    files: &[String],
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
{
    log::info!("Initialising variables and filter...");

    let mut outdict = HashMap::with_hasher(BuildHasherDefault::default());
    let mut minmaxdict = HashMap::with_hasher(BuildHasherDefault::default());
    let mut themap = HashMap::with_hasher(BuildHasherDefault::default());

    let mut histovec: Vec<u32> = vec![0; MAXSIZEHISTO];

    let mut kmer_filter = KmerFilter::new(qual.min_count);
    kmer_filter.init();

    // NOTE, potential TODO? : This could be slightly improved by filling outdict and minmaxdict only once, though it'd require saving also km, but it could be better
    extract_kmers_from_files(files, |seq, num_bases, qual_bytes| {
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
        log::info!("Starting fit...");
        minc = apply_spectrum_fit(&histovec);
        log::info!(
            "Fit done! Minimum count value to be used: {}. Filtering k-mers...",
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
fn chunked_preprocessing_standalone<IntT>(
    files: &[String],
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
    extract_kmers_from_files_batched(files, BATCH_RECORDS, |batch| {
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

        log::info!("Counting finished. Starting fit...");
        minc = apply_spectrum_fit(&histovec);
        log::info!(
            "Fit done! Fitted min_count value: {}. Starting filtering...",
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
pub fn preprocessing_standalone<IntT>(
    input_files: &[InputFastx],
    k: usize,
    qual: &QualOpts,
    timevec: &mut Vec<Instant>,
    out_path: &mut Option<PathBuf>,
    csize: usize,
    do_bloom: bool,
    do_fit: bool,
) -> PreprocessedK<IntT>
where
    IntT: for<'a> UInt<'a>,
{
    log::info!("Starting preprocessing_standalone with k = {k}");

    // This is temporal, to be changed in the future.
    let mut all_files: Vec<String> = input_files[0].1.clone();
    if input_files.len() > 1 {
        for ifile in input_files.iter().skip(1) {
            all_files.extend(ifile.1.clone());
        }
    }

    let (thedict, maxmindict, themap, histovec, used_min_count) = if do_bloom {
        log::info!("Processing using a Bloom filter");
        bloom_filter_preprocessing_standalone::<IntT>(&all_files, k, qual, do_fit, out_path)
    } else {
        // "No chunking" is one unbounded chunk. The guard matters: `i_record >= 0` holds on every
        // record, so passing 0 through would sort and count after every single read.
        let csize = if csize == 0 { usize::MAX } else { csize };
        if csize == usize::MAX {
            log::info!("Counting k-mers by sorting, without chunking");
        } else {
            log::info!("Counting k-mers by sorting, in chunks of {csize} records");
        }

        let estimated_kmers = all_files
            .iter()
            .map(|f| std::fs::metadata(f).map_or(0, |m| m.len()))
            .sum::<u64>() as usize
            / 5;
        let mut tmpvec: Vec<(u64, u64, u8)> = Vec::with_capacity(estimated_kmers);
        let out = chunked_preprocessing_standalone::<IntT>(
            &all_files, k, qual, &mut tmpvec, csize, do_fit, out_path,
        );
        drop(tmpvec);
        out
    };

    timevec.push(Instant::now());
    log::info!(
        "k-mers extracted, counted and filtered in {} s",
        timevec
            .last()
            .unwrap()
            .duration_since(*timevec.get(timevec.len().wrapping_sub(2)).unwrap())
            .as_secs()
    );
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
    /// so `apply_spectrum_fit`'s floor is well defined even for a degenerate histogram.
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


    /// The last bin saturates (it absorbs every count >= MAXSIZEHISTO), so it must not be mistaken for
    /// a peak — `fit_histogram` excludes it for the same reason.
    #[test]
    fn coverage_peak_ignores_the_saturating_bin() {
        let mut h = vec![0u32; MAXSIZEHISTO];
        h[MAXSIZEHISTO - 1] = 1_000_000; // everything >= 500 piled up here
        h[29] = 10;
        assert_eq!(coverage_peak(&h), 30);
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
