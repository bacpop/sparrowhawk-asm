//! Some docs should be here

use core::panic;

#[cfg(not(target_arch = "wasm32"))]
use std::{path::PathBuf, time::Instant};

use nohash_hasher::NoHashHasher;
use std::{cmp::Ordering, collections::HashMap, hash::BuildHasherDefault};

use rayon::prelude::*;

#[cfg(not(target_arch = "wasm32"))]
use needletail::parse_fastx_file;

// use std::process::exit;

#[cfg(not(target_arch = "wasm32"))]
use plotters::prelude::*;

use super::HashInfoSimple;
use super::QualOpts;

// #[cfg(not(target_arch = "wasm32"))]
// use super::bit_encoding::{encode_base, rc_base};

use crate::bit_encoding::UInt;
use crate::bloom_filter::KmerFilter;
use crate::kmer::Kmer;
use crate::logw;
use crate::spectrum_fitter::SpectrumFitter;

/// Tuple for name and list of input files
pub type InputFastx = (String, Vec<String>);

// #[cfg(target_arch = "wasm32")]
// use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use crate::fastx_wasm::open_fastq;
#[cfg(target_arch = "wasm32")]
use crate::post_state;
#[cfg(target_arch = "wasm32")]
use seq_io::fastq::Record;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_file_reader::WebSysFile;

// For the fitting, we'll use actually MAXSIZEHISTO - 1
const MAXSIZEHISTO: usize = 500;

#[cfg(not(target_arch = "wasm32"))]
use crate::gpu_filter;

#[inline]
fn add_to_histogram(histovec: &mut [u32], count: u16) {
    let idx = if (count as usize) >= MAXSIZEHISTO {
        MAXSIZEHISTO - 1
    } else {
        count as usize - 1
    };
    histovec[idx] = histovec[idx].saturating_add(1);
}

fn apply_spectrum_fit(histovec: &[u32]) -> u16 {
    let mut fit = SpectrumFitter::new();
    match fit.fit_histogram(histovec[..(MAXSIZEHISTO - 1)].to_vec()) {
        Ok(minc) => {
            let minc = minc as u16;
            if minc == 0 {
                panic!("Fitted min_count is zero or negative!");
            } else if minc <= 10 {
                logw(
                    "Fit has converged to a value smaller than 10. A value of 3 will be used as minimum, as usually this happens when the remaining k-mers go to low values, where the fit might give bad results. You should check whether this value is appropiated or not by looking at the k-mer spectrum histogram.",
                    Some("warn"),
                );
                3
            } else {
                minc
            }
        }
        Err(_) => {
            logw(
                "Fit has not converged. A value of 3 will be used as minimum, as usually this happens when the remaining k-mers go to low values. You should check whether this value is appropiated or not by looking at the k-mer spectrum histogram.",
                Some("warn"),
            );
            3
        }
    }
}

fn build_histogram_from_countmap(
    countmap: &HashMap<u64, (u16, u64, u8), BuildHasherDefault<NoHashHasher<u64>>>,
    histovec: &mut [u32],
) {
    for (_, tup) in countmap.iter() {
        add_to_histogram(histovec, tup.0);
    }
}

fn drain_countmap_into_themap<IntT>(
    countmap: &mut HashMap<u64, (u16, u64, u8), BuildHasherDefault<NoHashHasher<u64>>>,
    themap: &mut HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
    outdict: &mut HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    minmaxdict: &mut HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
    minc: u16,
    mut histovec_opt: Option<&mut [u32]>,
) where
    IntT: for<'a> UInt<'a>,
{
    countmap.retain(|h, tup| {
        if tup.0 >= minc {
            themap.entry(*h).or_insert_with(|| {
                HashInfoSimple {
                    hnc: tup.1,
                    b: tup.2,
                    pre: Vec::new(),
                    post: Vec::new(),
                    counts: tup.0,
                }
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

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
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
// #[cfg(not(target_arch = "wasm32"))]
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

// #[cfg(not(target_arch = "wasm32"))]
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

/// Read all FASTQ records from `files` into contiguous packed ASCII buffers suitable
/// for the GPU extraction pipeline.  Returns (seq_data, qual_data, read_offsets, read_lengths).
#[cfg(not(target_arch = "wasm32"))]
fn collect_raw_reads_for_gpu(
    files: &[String],
) -> (Vec<u8>, Vec<u8>, Vec<u32>, Vec<u32>) {
    let mut seq_data:     Vec<u8> = Vec::new();
    let mut qual_data:    Vec<u8> = Vec::new();
    let mut read_offsets: Vec<u32> = Vec::new();
    let mut read_lengths: Vec<u32> = Vec::new();

    for file in files {
        let mut reader = parse_fastx_file(file)
            .unwrap_or_else(|_| panic!("Invalid path/file: {file}"));
        while let Some(record) = reader.next() {
            let seqrec = record.expect("Invalid FASTQ record");
            let seq = seqrec.seq();
            let len = seq.len() as u32;
            if len == 0 { continue; }
            read_offsets.push(seq_data.len() as u32);
            read_lengths.push(len);
            seq_data.extend_from_slice(&seq);
            if let Some(q) = seqrec.qual() {
                qual_data.extend_from_slice(q);
            } else {
                qual_data.extend(std::iter::repeat_n(b'~', len as usize));
            }
        }
    }
    (seq_data, qual_data, read_offsets, read_lengths)
}

/// Read all FASTQ records from `file1` and `file2` into contiguous packed ASCII buffers
/// suitable for the GPU extraction pipeline (WASM version using seq_io readers).
/// Returns (seq_data, qual_data, read_offsets, read_lengths).
#[cfg(target_arch = "wasm32")]
fn collect_raw_reads_for_gpu_wasm(
    file1: &mut WebSysFile,
    file2: &mut WebSysFile,
) -> (Vec<u8>, Vec<u8>, Vec<u32>, Vec<u32>) {
    let mut seq_data:     Vec<u8>  = Vec::new();
    let mut qual_data:    Vec<u8>  = Vec::new();
    let mut read_offsets: Vec<u32> = Vec::new();
    let mut read_lengths: Vec<u32> = Vec::new();

    let mut reader = open_fastq(file1);
    while let Some(record) = reader.next() {
        let seqrec = record.expect("Invalid FASTQ record");
        let seq = seqrec.seq();
        let len = seq.len() as u32;
        if len == 0 { continue; }
        read_offsets.push(seq_data.len() as u32);
        read_lengths.push(len);
        seq_data.extend_from_slice(seq);
        qual_data.extend_from_slice(seqrec.qual());
    }
    drop(reader);

    let mut reader = open_fastq(file2);
    while let Some(record) = reader.next() {
        let seqrec = record.expect("Invalid FASTQ record");
        let seq = seqrec.seq();
        let len = seq.len() as u32;
        if len == 0 { continue; }
        read_offsets.push(seq_data.len() as u32);
        read_lengths.push(len);
        seq_data.extend_from_slice(seq);
        qual_data.extend_from_slice(seqrec.qual());
    }

    (seq_data, qual_data, read_offsets, read_lengths)
}

/// CPU bulk extraction: pushes all kmer occurrences into a flat `outvec` for CPU sort+count.
#[cfg(not(target_arch = "wasm32"))]
fn bulk_preprocessing_standalone_cpu<IntT>(
    files: &[String],
    k: usize,
    qual: &QualOpts,
    outvec: &mut Vec<(u64, u64, u8)>,
) -> (
    Vec<u64>,
    HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
)
where
    IntT: for<'a> UInt<'a>,
{
    let mut outdict = HashMap::with_hasher(BuildHasherDefault::default());
    let mut minmaxdict = HashMap::with_hasher(BuildHasherDefault::default());

    let theseq: Vec<u64> = Vec::new();

    extract_kmers_from_files(files, |seq, num_bases, qual_bytes| {
        let kmer_opt = Kmer::<IntT>::new(seq, num_bases, qual_bytes, k, qual.min_qual, true);
        if let Some(mut kmer_it) = kmer_opt {
            let (hc, hnc, b, km) = kmer_it.get_curr_kmerhash_and_bases_and_kmer();
            outvec.push((hc, hnc, b));
            outdict.entry(hc).or_insert(km);
            minmaxdict.entry(hnc).or_insert(hc);
            while let Some((hc, hnc, b, km)) = kmer_it.get_next_kmer_and_give_us_things() {
                outvec.push((hc, hnc, b));
                outdict.entry(hc).or_insert(km);
                minmaxdict.entry(hnc).or_insert(hc);
            }
        }
    });

    (theseq, outdict, minmaxdict)
}

#[cfg(target_arch = "wasm32")]
fn get_kmers_from_both_files_wasm<IntT>(
    file1: &mut WebSysFile,
    file2: &mut WebSysFile,
    k: usize,
    qual: &QualOpts,
    outvec: &mut Vec<(u64, u64, u8)>,
) -> (
    HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
)
where
    IntT: for<'a> UInt<'a>,
{
    let mut outdict = HashMap::with_hasher(BuildHasherDefault::default());
    let mut minmaxdict = HashMap::with_hasher(BuildHasherDefault::default());
    logw(
        "Getting kmers from first file. Creating reader...",
        Some("info"),
    );
    let mut reader = open_fastq(file1);

    logw("Entering while loop...", Some("info"));
    post_state("preprocess:bulk:loop:start");
    let mut count: usize = 0;

    //     let maxkmers = 200;
    //     let mut numkmers = 0;

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
        count += 1;
        if count.is_multiple_of(150000) {
            post_state(&format!("preprocess:bulk:loop:{:?}", count));
        }
    }
    logw(
        "Finished getting kmers from first file. Starting with the second...",
        Some("info"),
    );

    post_state(&format!("preprocess:bulk:loop:{:?}:50", count));
    let percentageblock = (count as f64 / 10_f64) as usize;

    let mut reader = open_fastq(file2);

    // Filling the seq of the second file!
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
            //             numkmers += 1;
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
        count += 1;
        // This might be done slightly more efficiently??
        if count.is_multiple_of(percentageblock) {
            post_state(&format!(
                "preprocess:bulk:loop:{:?}:{:?}",
                count,
                count / percentageblock * 5
            ));
        }
    }

    logw("Finished getting kmers from the second file", Some("info"));
    post_state("preprocess:bulk:loop:end");

    (outdict, minmaxdict)
}

#[cfg(target_arch = "wasm32")]
fn chunked_processing_wasm<IntT>(
    file1: &mut WebSysFile,
    file2: &mut WebSysFile,
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
    let mut countmap: HashMap<u64, (u16, u64, u8), BuildHasherDefault<NoHashHasher<u64>>> =
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
        if count.is_multiple_of(percentageblock) {
            post_state(&format!(
                "preprocess:chunked:loop:{:?}:{:?}",
                count,
                count / percentageblock * 5
            ));
        }
    }

    if i_record > 0 {
        // Processssssss! And reset.
        if !outvec.is_empty() {
            logw("Processing last chunk. Sorting k-mers...", Some("info"));
            outvec.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
            logw("k-mers sorted. Counting k-mers...", Some("info"));
            // Then, do a counting of everything and save the results in a dictionary and return it

            update_countmap(outvec, &mut countmap);
        }
        // Reset
        outvec.clear();
    }

    logw("Finished getting kmers from the second file", Some("info"));
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
            format!("Fit done! Fitted min_count value: {}. Starting filtering...", minc).as_str(),
            Some("info"),
        );

        post_state("preprocess:chunked:filtering");
        drain_countmap_into_themap(&mut countmap, &mut themap, &mut outdict, &mut minmaxdict, minc, None);
    } else {
        post_state("preprocess:chunked:filtering");
        drain_countmap_into_themap(&mut countmap, &mut themap, &mut outdict, &mut minmaxdict, minc, Some(&mut histovec));
    }

    outdict.shrink_to_fit();
    minmaxdict.shrink_to_fit();
    histovec.shrink_to_fit();
    drop(countmap);

    (outdict, minmaxdict, themap, histovec, minc)
}

#[cfg(target_arch = "wasm32")]
fn bloom_filter_preprocessing_wasm<IntT>(
    file1: &mut WebSysFile,
    file2: &mut WebSysFile,
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
        if count.is_multiple_of(percentageblock) {
            post_state(&format!(
                "preprocess:bloom:loop:{:?}:{:?}",
                count,
                count / percentageblock * 5
            ));
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
            format!("Fit done! Fitted min_count value: {}. Starting filtering...", minc).as_str(),
            Some("info"),
        );

        post_state("preprocess:bloom:filtering");
        drain_countmap_into_themap(&mut countmap, &mut themap, &mut outdict, &mut minmaxdict, minc, None);
    } else {
        post_state("preprocess:bloom:filtering");
        // I think this part can be improved: now that the min_count error has been corrected, some conditionals could be removed here?
        drain_countmap_into_themap(&mut countmap, &mut themap, &mut outdict, &mut minmaxdict, minc, Some(&mut histovec));
    }

    outdict.shrink_to_fit();
    minmaxdict.shrink_to_fit();

    (outdict, minmaxdict, themap, histovec, minc)
}

#[cfg(not(target_arch = "wasm32"))]
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
        log::info!("Fit done! Minimum count value to be used: {}. Filtering k-mers...", minc);

        drain_countmap_into_themap(&mut countmap, &mut themap, &mut outdict, &mut minmaxdict, minc, None);
    } else {
        log::info!("Filtering k-mers...");
        minc = qual.min_count;
        drain_countmap_into_themap(&mut countmap, &mut themap, &mut outdict, &mut minmaxdict, minc, Some(&mut histovec));
    }

    outdict.shrink_to_fit();
    minmaxdict.shrink_to_fit();

    if let Some(p) = out_path {
        plot_kmer_histogram(&histovec, p.as_path());
    }

    (outdict, minmaxdict, themap)
}

#[cfg(not(target_arch = "wasm32"))]
fn get_map_with_counts(
    invec: &[(u64, u64, u8)],
    min_count: u16,
    out_path: &mut Option<PathBuf>,
) -> HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>> {
    let mut outdict = HashMap::with_hasher(BuildHasherDefault::default());

    let mut i = 0;
    let mut c: u16 = 0;
    let mut tmphash = invec[i].0;
    // let mut tmpcounter = 0;

    // I'm not sure if there is another way of avoiding the extra conditional checks for plotting (or perhaps the
    // compiler is smart enough to write in assembly exactly what I am now writing here?) than to rewrite the function
    // with a big "if" as it is here, to gain a bit of efficiency (not sure how much, but well).
    if out_path.is_some() {
        let mut plotvec: Vec<u16> = Vec::new(); // For plotting

        while i < invec.len() {
            if tmphash != invec[i].0 {
                if c >= min_count {
                    // tmpcounter += 1;
                    outdict
                        .entry(tmphash)
                        .or_insert(HashInfoSimple {
                            hnc: invec[i - 1].1,
                            b: invec[i - 1].2,
                            pre: Vec::new(),
                            post: Vec::new(),
                            counts: c,
                        });
                } else {
                    plotvec.push(c);
                }
                tmphash = invec[i].0;
                c = 1;
            } else {
                c = c.saturating_add(1);
            }
            i += 1;
        }

        if c >= min_count {
            // tmpcounter += 1;
            outdict
                .entry(tmphash)
                .or_insert(HashInfoSimple {
                    hnc: invec[i - 1].1,
                    b: invec[i - 1].2,
                    pre: Vec::new(),
                    post: Vec::new(),
                    counts: c,
                });
        } else {
            plotvec.push(c);
        }
        plotvec.shrink_to_fit();

        // Plotting!
        // // TEST
        // for i in 0..plotvec.len() {
        //     logw(format!("#######{:?}-{:?}", i, plotvec[i]).as_str(), Some("info"));
        // }
        // // TEST END

        let root = BitMapBackend::new(out_path.as_ref().unwrap().as_path(), (1280, 960))
            .into_drawing_area();

        let _ = root.fill(&WHITE);

        let mut chart = ChartBuilder::on(&root)
            .x_label_area_size(35)
            .y_label_area_size(40)
            .margin(5)
            .caption("k-mer spectrum", ("sans-serif", 30.0))
            .build_cartesian_2d((0u32..200u32).into_segmented(), 0u32..200000u32)
            .unwrap();

        chart
            .configure_mesh()
            .disable_x_mesh()
            .bold_line_style(WHITE.mix(0.3))
            .y_desc("Counts")
            .x_desc("k-mer frequency")
            .axis_desc_style(("sans-serif", 15))
            .draw()
            .unwrap();

        chart
            .draw_series(
                Histogram::vertical(&chart)
                    .style(RED.filled())
                    // .data(plotvec.iter().map(|x: &u16| (*x as u32, 1)).chain(outdict.iter().map(|(_, x)| (x.borrow().counts as u32, 1)))),
                    .data(
                        plotvec
                            .iter()
                            .map(|x: &u16| (*x as u32, 1))
                            .chain(outdict.values().map(|x| (x.counts as u32, 1))),
                    ),
            )
            .unwrap();

        root.present()
            .expect("Unable to write result to file. Does the output folder exist?");

    //     exit(1);

    // log::debug!("Good kmers {}", tmpcounter);
    } else {
        // Here we don't need to check for plotting or anything
        while i < invec.len() {
            if tmphash != invec[i].0 {
                if c >= min_count {
                    // tmpcounter += 1;
                    outdict
                        .entry(tmphash)
                        .or_insert(HashInfoSimple {
                            hnc: invec[i - 1].1,
                            b: invec[i - 1].2,
                            pre: Vec::new(),
                            post: Vec::new(),
                            counts: c,
                        });
                }
                tmphash = invec[i].0;
                c = 1;
            } else {
                c = c.saturating_add(1);
            }
            i += 1;
        }

        if c >= min_count {
            // tmpcounter += 1;
            outdict
                .entry(tmphash)
                .or_insert(HashInfoSimple {
                    hnc: invec[i - 1].1,
                    b: invec[i - 1].2,
                    pre: Vec::new(),
                    post: Vec::new(),
                    counts: c,
                });
        }
    }
    outdict
}

#[cfg(not(target_arch = "wasm32"))]
fn get_map_with_counts_and_fit(
    invec: &mut Vec<(u64, u64, u8)>,
    out_path: &mut Option<PathBuf>,
) -> HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>> {
    let mut outdict = HashMap::with_hasher(BuildHasherDefault::default());

    let mut i = 0;
    let mut c: u16 = 0;
    let mut tmphash = invec[i].0;
    // let mut tmpcounter = 0;

    // Here we don't have the same issue as with the pre-defined min_count setting.
    let mut plotvec: Vec<u32> = vec![0_u32; MAXSIZEHISTO]; // For plotting

    // We need to construct the histogram as well

    while i < invec.len() {
        if tmphash != invec[i].0 {
            // tmpcounter += 1;
            outdict
                .entry(tmphash)
                .or_insert(HashInfoSimple {
                    hnc: invec[i - 1].1,
                    b: invec[i - 1].2,
                    pre: Vec::new(),
                    post: Vec::new(),
                    counts: c,
                });

            if c as usize > MAXSIZEHISTO {
                plotvec[MAXSIZEHISTO - 1] = plotvec[MAXSIZEHISTO - 1].saturating_add(1);
            } else {
                plotvec[c as usize - 1] = plotvec[c as usize - 1].saturating_add(1);
            }

            tmphash = invec[i].0;
            c = 1;
        } else {
            c = c.saturating_add(1);
        }
        i += 1;
    }

    // tmpcounter += 1;
    outdict
        .entry(tmphash)
        .or_insert(HashInfoSimple {
            hnc: invec[i - 1].1,
            b: invec[i - 1].2,
            pre: Vec::new(),
            post: Vec::new(),
            counts: c,
        });

    if c as usize > MAXSIZEHISTO {
        plotvec[MAXSIZEHISTO - 1] = plotvec[MAXSIZEHISTO - 1].saturating_add(1);
    } else {
        plotvec[c as usize - 1] = plotvec[c as usize - 1].saturating_add(1);
    }

    invec.clear();
    invec.shrink_to_fit(); // Quick optimisation

    // // TEST
    // for i in 0..plotvec.len() {
    //     logw(format!("#######{:?}-{:?}", i, plotvec[i]).as_str(), Some("info"));
    // }
    // // TEST END
    // We need to do the fit!
    log::info!("Counting finished. Starting fit...");
    let mut fit = SpectrumFitter::new();
    // Remove the last bin, as it might affect the fit, but we want it in the vector to plot it in case the coverage is really
    // large (and so that we can detect it).
    // let fitted_min_count = fit.fit_histogram(plotvec[..(MAXSIZEHISTO - 1)].to_vec()).expect("Fit to the k-mer spectrum failed!") as u16;
    let mut fitted_min_count: u16;

    let result = fit.fit_histogram(plotvec[..(MAXSIZEHISTO - 1)].to_vec());
    if let Ok(theres) = result {
        fitted_min_count = theres as u16;
    } else {
        logw("Fit has not converged. A value of 3 will be used as minimum, as usually this happens when the remaining k-mers go to low values. You should check whether this value is appropiated or not by looking at the k-mer spectrum histogram.", Some("warn"));
        fitted_min_count = 3;
    }

    if fitted_min_count == 0 {
        panic!("Fitted min_count value is zero or negative!");
    } else if fitted_min_count <= 10 {
        logw("Fit has converged to a value smaller than 10. A value of 3 will be used as minimum, as usually this happens when the remaining k-mers go to low values, where the fit might give bad results. You should check whether this value is appropiated or not by looking at the k-mer spectrum histogram.", Some("warn"));
        fitted_min_count = 3;
    }

    log::info!(
        "Fit done! Fitted min_count value: {}. Starting filtering...",
        fitted_min_count
    );

    outdict.retain(|_, hi| hi.counts >= fitted_min_count);
    outdict.shrink_to_fit();

    if let Some(p) = out_path {
        plot_kmer_histogram(&plotvec, p.as_path());
    }

    // log::debug!("Good kmers {}", tmpcounter);
    //     exit(1);

    outdict
}

#[cfg(target_arch = "wasm32")]
fn get_map_wasm(
    invec: &mut Vec<(u64, u64, u8)>,
    min_count: u16,
    do_fit: bool,
) -> (
    HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
    Vec<u32>,
    u16,
) {
    let mut outdict = HashMap::with_hasher(BuildHasherDefault::default());

    let mut i = 0;
    let mut c: u16 = 0;
    let mut minc: u16 = min_count;
    let mut tmphash = invec[i].0;
    // let mut tmpcounter = 0;

    let mut plotvec: Vec<u32> = vec![0_u32; MAXSIZEHISTO]; // For plotting

    if do_fit {
        post_state("preprocess:bulk:fitting");
        while i < invec.len() {
            if tmphash != invec[i].0 {
                // tmpcounter += 1;
                outdict
                    .entry(tmphash)
                    .or_insert(HashInfoSimple {
                        hnc: invec[i - 1].1,
                        b: invec[i - 1].2,
                        pre: Vec::new(),
                        post: Vec::new(),
                        counts: c,
                    });

                if c as usize > MAXSIZEHISTO {
                    plotvec[MAXSIZEHISTO - 1] = plotvec[MAXSIZEHISTO - 1].saturating_add(1);
                } else {
                    plotvec[c as usize - 1] = plotvec[c as usize - 1].saturating_add(1);
                }

                tmphash = invec[i].0;
                c = 1;
            } else {
                c = c.saturating_add(1);
            }
            i += 1;
        }

        // tmpcounter += 1;
        outdict
            .entry(tmphash)
            .or_insert(HashInfoSimple {
                hnc: invec[i - 1].1,
                b: invec[i - 1].2,
                pre: Vec::new(),
                post: Vec::new(),
                counts: c,
            });

        if c as usize > MAXSIZEHISTO {
            plotvec[MAXSIZEHISTO - 1] = plotvec[MAXSIZEHISTO - 1].saturating_add(1);
        } else {
            plotvec[c as usize - 1] = plotvec[c as usize - 1].saturating_add(1);
        }

        invec.clear();
        invec.shrink_to_fit(); // Quick optimisation

        // We need to do the fit!
        logw("Counting finished. Starting fit...", Some("info"));
        let mut fit = SpectrumFitter::new();
        // Remove the last bin, as it might affect the fit, but we want it in the vector to plot it in case the coverage is really
        // large (and so that we can detect it).
        // minc = fit.fit_histogram(plotvec[..(MAXSIZEHISTO - 1)].to_vec()).expect("Fit to the k-mer spectrum failed!") as u16;

        let result = fit.fit_histogram(plotvec[..(MAXSIZEHISTO - 1)].to_vec());
        if let Ok(theres) = result {
            minc = theres as u16;
            if minc == 0 {
                panic!("Fitted min_count value is zero or negative!");
            } else if minc <= 10 {
                logw("Fit has converged to a value smaller than 10. A value of 3 will be used as minimum, as usually this happens when the remaining k-mers go to low values, where the fit might give bad results. You should check whether this value is appropiated or not by looking at the k-mer spectrum histogram.", Some("warn"));
                minc = 3;
            }
        } else {
            logw("Fit has not converged. A value of 3 will be used as minimum, as usually this happens when the remaining k-mers go to low values. You should check whether this value is appropiated or not by looking at the k-mer spectrum histogram.", Some("warn"));
            minc = 3;
        }

        logw(
            format!(
                "Fit done! Fitted min_count value: {}. Starting filtering...",
                minc
            )
            .as_str(),
            Some("info"),
        );

        post_state("preprocess:bulk:filtering");
        outdict.retain(|_, hi| hi.counts >= minc);
        outdict.shrink_to_fit();
    } else {
        post_state("preprocess:bulk:filtering");
        while i < invec.len() {
            if tmphash != invec[i].0 {
                if c >= minc {
                    // tmpcounter += 1;
                    outdict
                        .entry(tmphash)
                        .or_insert(HashInfoSimple {
                            hnc: invec[i - 1].1,
                            b: invec[i - 1].2,
                            pre: Vec::new(),
                            post: Vec::new(),
                            counts: c,
                        });
                }

                if c as usize > MAXSIZEHISTO {
                    plotvec[MAXSIZEHISTO - 1] = plotvec[MAXSIZEHISTO - 1].saturating_add(1);
                } else {
                    plotvec[c as usize - 1] = plotvec[c as usize - 1].saturating_add(1);
                }

                tmphash = invec[i].0;
                c = 1;
            } else {
                c = c.saturating_add(1);
            }
            i += 1;
        }

        if c >= minc {
            // tmpcounter += 1;
            outdict
                .entry(tmphash)
                .or_insert(HashInfoSimple {
                    hnc: invec[i - 1].1,
                    b: invec[i - 1].2,
                    pre: Vec::new(),
                    post: Vec::new(),
                    counts: c,
                });
        }

        if c as usize > MAXSIZEHISTO {
            plotvec[MAXSIZEHISTO - 1] = plotvec[MAXSIZEHISTO - 1].saturating_add(1);
        } else {
            plotvec[c as usize - 1] = plotvec[c as usize - 1].saturating_add(1);
        }
        // logw(format!("Good kmers {}", tmpcounter).as_str(), Some("debug"));
    }
    (outdict, plotvec, minc)
}

fn update_countmap(
    invec: &[(u64, u64, u8)],
    countmap: &mut HashMap<u64, (u16, u64, u8), BuildHasherDefault<NoHashHasher<u64>>>,
) {
    let mut i = 0;
    let mut c: u16 = 0;
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

#[cfg(not(target_arch = "wasm32"))]
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
)
where
    IntT: for<'a> UInt<'a>,
{
    log::info!("Getting kmers from files. Creating reader...");

    let mut outdict = HashMap::with_hasher(BuildHasherDefault::default());
    let mut minmaxdict = HashMap::with_hasher(BuildHasherDefault::default());
    let mut themap = HashMap::with_hasher(BuildHasherDefault::default());
    let mut countmap: HashMap<u64, (u16, u64, u8), BuildHasherDefault<NoHashHasher<u64>>> =
        HashMap::with_hasher(BuildHasherDefault::default());

    let mut histovec: Vec<u32> = vec![0; MAXSIZEHISTO];
    let mut i_record = 0;
    // let mut ncols : usize = 0;

    extract_kmers_from_files(files, |seq, num_bases, qual_bytes| {
        let kmer_opt = Kmer::<IntT>::new(seq, num_bases, qual_bytes, k, qual.min_qual, true);
        if let Some(mut kmer_it) = kmer_opt {
            let (hc, hnc, b, km) = kmer_it.get_curr_kmerhash_and_bases_and_kmer();
            outvec.push((hc, hnc, b));
            outdict.entry(hc).or_insert(km);
            minmaxdict.entry(hnc).or_insert(hc);
            while let Some((hc, hnc, b, km)) = kmer_it.get_next_kmer_and_give_us_things() {
                outvec.push((hc, hnc, b));
                outdict.entry(hc).or_insert(km);
                minmaxdict.entry(hnc).or_insert(hc);
            }
        }

        i_record += 1;
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

    if i_record > 0 {
        // Processssssss! And reset.
        if !outvec.is_empty() {
            log::info!("Processing last chunk. Sorting k-mers...");
            outvec.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
            log::info!("k-mers sorted. Counting k-mers...");
            // Then, do a counting of everything and save the results in a dictionary and return it

            update_countmap(outvec, &mut countmap);
        }
        // Reset
        outvec.clear();
    }
    log::info!("Finished getting kmers from the second file");

    // println!("k | Number of collisions =+=+ {} {}", k, ncols);
    //exit(0);

    log::info!("Filtering...");

    // Now, get themap, histovec, and filter outdict and minmaxdict
    countmap.shrink_to_fit();
    let minc;

    // This can be optimised. also better written: I had to repeat the code for the retains, to try to improve slightly the running time in
    // case no autofitting is requested. In any case, it could be improved in the future.
    if do_fit {
        build_histogram_from_countmap(&countmap, &mut histovec);

        // // TEST
        // for i in 0..histovec.len() {
        //     logw(format!("#######{:?}-{:?}", i, histovec[i]).as_str(), Some("info"));
        // }
        // // TEST END

        // Remove the last bin, as it might affect the fit, but we want it in the vector to plot it in case the coverage is really
        // large (and so that we can detect it).
        log::info!("Counting finished. Starting fit...");
        minc = apply_spectrum_fit(&histovec);
        log::info!("Fit done! Fitted min_count value: {}. Starting filtering...", minc);

        drain_countmap_into_themap(&mut countmap, &mut themap, &mut outdict, &mut minmaxdict, minc, None);
    } else {
        minc = qual.min_count;
        drain_countmap_into_themap(&mut countmap, &mut themap, &mut outdict, &mut minmaxdict, minc, Some(&mut histovec));
    }

    drop(countmap);
    outdict.shrink_to_fit();
    minmaxdict.shrink_to_fit();

    if let Some(p) = out_path {
        plot_kmer_histogram(&histovec, p.as_path());
    }

    (outdict, minmaxdict, themap)
}

/// Read fastq files, get the reads, get the k-mers, count them, filter them by count, and get some way of recovering the sequence later.
#[cfg(not(target_arch = "wasm32"))]
pub async fn preprocessing_standalone<IntT>(
    input_files: &[InputFastx],
    k: usize,
    qual: &QualOpts,
    timevec: &mut Vec<Instant>,
    out_path: &mut Option<PathBuf>,
    csize: usize,
    do_bloom: bool,
    do_fit: bool,
    use_gpu: bool,
) -> (
    HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
    Vec<u64>,
    HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
)
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

    if do_bloom {
        // Build indexes
        log::info!("Processing using a Bloom filter");

        let (thedict, maxmindict, themap) =
            bloom_filter_preprocessing_standalone::<IntT>(&all_files, k, qual, do_fit, out_path);
        (themap, Vec::new(), thedict, maxmindict)
    } else if csize == 0 {
        log::info!("Processing in bulk");

        let themap;
        let theseq;
        let thedict;
        let maxmindict;

        if use_gpu {
            // GPU k-mer extraction path: upload raw sequences, extract+count+filter on GPU,
            // then do a single CPU pass to build outdict/maxmindict for surviving k-mers.
            log::info!("Using GPU k-mer extraction + count + filter");

            let (seq_data, qual_data, gpu_read_offsets, gpu_read_lengths) =
                collect_raw_reads_for_gpu(&all_files);

            timevec.push(Instant::now());
            log::info!(
                "Reads collected for GPU in {} s",
                timevec
                    .last()
                    .unwrap()
                    .duration_since(*timevec.get(timevec.len().wrapping_sub(2)).unwrap())
                    .as_secs()
            );

            let gpu_themap = gpu_filter::gpu_extract_count_filter(
                &seq_data,
                &qual_data,
                &gpu_read_offsets,
                &gpu_read_lengths,
                k as u32,
                // CPU accepts (raw - 33) > min_qual_phred, i.e. raw >= phred + 34.
                // GPU shader rejects raw < params.min_qual, so pass phred + 34.
                qual.min_qual as u32 + 34,
                qual.min_count,
                wgpu::PowerPreference::HighPerformance,
            ).await;
            drop(seq_data);
            drop(qual_data);

            timevec.push(Instant::now());
            log::info!(
                "GPU extraction+count+filter done in {} s",
                timevec
                    .last()
                    .unwrap()
                    .duration_since(*timevec.get(timevec.len().wrapping_sub(2)).unwrap())
                    .as_secs()
            );

            // Second CPU pass: build outdict + maxmindict only for k-mers that survived.
            let mut od: HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>> =
                HashMap::with_hasher(BuildHasherDefault::default());
            let mut mmd: HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>> =
                HashMap::with_hasher(BuildHasherDefault::default());
            extract_kmers_from_files(&all_files, |seq, num_bases, qual_bytes| {
                let kmer_opt =
                    Kmer::<IntT>::new(seq, num_bases, qual_bytes, k, qual.min_qual, true);
                if let Some(mut kmer_it) = kmer_opt {
                    let (hc, hnc, _b, km) =
                        kmer_it.get_curr_kmerhash_and_bases_and_kmer();
                    if gpu_themap.contains_key(&hc) {
                        od.entry(hc).or_insert(km);
                        mmd.entry(hnc).or_insert(hc);
                    }
                    while let Some((hc, hnc, _b, km)) =
                        kmer_it.get_next_kmer_and_give_us_things()
                    {
                        if gpu_themap.contains_key(&hc) {
                            od.entry(hc).or_insert(km);
                            mmd.entry(hnc).or_insert(hc);
                        }
                    }
                }
            });

            timevec.push(Instant::now());
            log::info!(
                "outdict+maxmindict built in {} s",
                timevec
                    .last()
                    .unwrap()
                    .duration_since(*timevec.get(timevec.len().wrapping_sub(2)).unwrap())
                    .as_secs()
            );

            themap = gpu_themap;
            theseq = Vec::new();
            thedict = od;
            maxmindict = mmd;
        } else {
            // CPU path: flat vec → par_sort → count+filter
            log::info!("Using CPU sort + count + filter");
            let estimated_kmers = all_files.iter()
                .map(|f| std::fs::metadata(f).map_or(0, |m| m.len()))
                .sum::<u64>() as usize / 5;
            let mut tmpvec: Vec<(u64, u64, u8)> = Vec::with_capacity(estimated_kmers);
            let (tseq, tdict, mmdict) =
                bulk_preprocessing_standalone_cpu::<IntT>(&all_files, k, qual, &mut tmpvec);
            theseq = tseq;
            thedict = tdict;
            maxmindict = mmdict;

            timevec.push(Instant::now());
            log::info!(
                "k-mers extracted in {} s",
                timevec
                    .last()
                    .unwrap()
                    .duration_since(*timevec.get(timevec.len().wrapping_sub(2)).unwrap())
                    .as_secs()
            );

            log::debug!("Number of kmers BEFORE cleaning: {:?}", tmpvec.len());
            log::info!("Sorting vector");
            tmpvec.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));

            timevec.push(Instant::now());
            log::info!(
                "k-mers sorted in {} s",
                timevec
                    .last()
                    .unwrap()
                    .duration_since(*timevec.get(timevec.len().wrapping_sub(2)).unwrap())
                    .as_secs()
            );

            log::info!("Counting and filtering k-mers");
            themap = if !do_fit {
                get_map_with_counts(&tmpvec, qual.min_count, out_path)
            } else {
                get_map_with_counts_and_fit(&mut tmpvec, out_path)
            };
            drop(tmpvec);

            timevec.push(Instant::now());
            log::info!(
                "k-mers counted and filtered in {} s",
                timevec
                    .last()
                    .unwrap()
                    .duration_since(*timevec.get(timevec.len().wrapping_sub(2)).unwrap())
                    .as_secs()
            );
        }

        (themap, theseq, thedict, maxmindict)
    } else {
        log::info!("Processing in chunks of size {}", csize);

        let estimated_kmers = all_files.iter()
            .map(|f| std::fs::metadata(f).map_or(0, |m| m.len()))
            .sum::<u64>() as usize / 5;
        let mut tmpvec: Vec<(u64, u64, u8)> = Vec::with_capacity(estimated_kmers);
        let (thedict, maxmindict, themap) = chunked_preprocessing_standalone::<IntT>(
            &all_files,
            k,
            qual,
            &mut tmpvec,
            csize,
            do_fit,
            out_path,
        );
        drop(tmpvec);

        timevec.push(Instant::now());
        log::info!(
            "Chunked preprocessing done in {} s",
            timevec
                .last()
                .unwrap()
                .duration_since(*timevec.get(timevec.len().wrapping_sub(2)).unwrap())
                .as_secs()
        );
        (themap, Vec::new(), thedict, maxmindict)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nohash_hasher::NoHashHasher;
    use std::{collections::HashMap, hash::BuildHasherDefault};

    fn empty_countmap() -> HashMap<u64, (u16, u64, u8), BuildHasherDefault<NoHashHasher<u64>>> {
        HashMap::with_hasher(BuildHasherDefault::default())
    }

    fn empty_themap(
    ) -> HashMap<u64, crate::HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>> {
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
        cmap.insert(1u64, (5u16, 2u64, 0u8)); // count=5 >= minc=3 → in themap
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
        cmap.insert(2u64, (2u16, 3u64, 0u8)); // count=2 < minc=3 → removed from outdict
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
        cmap.insert(5u64, (3u16, 0u64, 0u8));
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
        cmap.insert(1u64, (5u16, 0u64, 0u8)); // count=5 → histovec[4]
        cmap.insert(2u64, (10u16, 0u64, 0u8)); // count=10 → histovec[9]
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
}

#[cfg(target_arch = "wasm32")]
/// Main preprocessing function for wasm
pub async fn preprocessing_wasm<IntT>(
    file1: &mut WebSysFile,
    file2: &mut WebSysFile,
    k: usize,
    qual: &QualOpts,
    csize: usize,
    do_bloom: bool,
    do_fit: bool,
    use_gpu: bool,
    gpu_power_pref: u32,
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
    } else if csize == 0 {
        post_state("preprocess:bulk:start");
        logw("Starting preprocessing with k = {k}", Some("info"));

        if use_gpu {
            // GPU k-mer extraction path: upload raw sequences, extract+count+filter on GPU,
            // then do a second CPU pass to build outdict/maxmindict for surviving k-mers only.
            logw("Using GPU k-mer extraction + count + filter", Some("info"));
            post_state("preprocess:bulk:gpu:collect");

            let (seq_data, qual_data, gpu_read_offsets, gpu_read_lengths) =
                collect_raw_reads_for_gpu_wasm(file1, file2);

            let pref = match gpu_power_pref {
                1 => wgpu::PowerPreference::HighPerformance,
                2 => wgpu::PowerPreference::LowPower,
                _ => wgpu::PowerPreference::None,
            };

            post_state("preprocess:bulk:gpu:extract");
            let gpu_themap = crate::gpu_filter::gpu_extract_count_filter(
                &seq_data,
                &qual_data,
                &gpu_read_offsets,
                &gpu_read_lengths,
                k as u32,
                // CPU accepts (raw - 33) > min_qual_phred, i.e. raw >= phred + 34.
                // GPU shader rejects raw < params.min_qual, so pass phred + 34.
                qual.min_qual as u32 + 34,
                qual.min_count,
                pref,
            ).await;
            drop(seq_data);
            drop(qual_data);

            // Second CPU pass: build outdict + maxmindict only for surviving k-mers.
            post_state("preprocess:bulk:gpu:second_pass");
            let mut od: HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>> =
                HashMap::with_hasher(BuildHasherDefault::default());
            let mut mmd: HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>> =
                HashMap::with_hasher(BuildHasherDefault::default());

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
                    let (hc, hnc, _b, km) = kmer_it.get_curr_kmerhash_and_bases_and_kmer();
                    if gpu_themap.contains_key(&hc) {
                        od.entry(hc).or_insert(km);
                        mmd.entry(hnc).or_insert(hc);
                    }
                    while let Some((hc, hnc, _b, km)) = kmer_it.get_next_kmer_and_give_us_things() {
                        if gpu_themap.contains_key(&hc) {
                            od.entry(hc).or_insert(km);
                            mmd.entry(hnc).or_insert(hc);
                        }
                    }
                }
            }
            drop(reader);

            let mut reader = open_fastq(file2);
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
                    let (hc, hnc, _b, km) = kmer_it.get_curr_kmerhash_and_bases_and_kmer();
                    if gpu_themap.contains_key(&hc) {
                        od.entry(hc).or_insert(km);
                        mmd.entry(hnc).or_insert(hc);
                    }
                    while let Some((hc, hnc, _b, km)) = kmer_it.get_next_kmer_and_give_us_things() {
                        if gpu_themap.contains_key(&hc) {
                            od.entry(hc).or_insert(km);
                            mmd.entry(hnc).or_insert(hc);
                        }
                    }
                }
            }

            post_state("preprocess:bulk:gpu:done");
            return (gpu_themap, Some(od), mmd, Vec::new(), qual.min_count);
        }

        // CPU path: flat vec → par_sort → count+filter
        // First, we want to fill our mega-vector with all k-mers from both paired-end reads
        logw("Filling vector", Some("info"));

        let mut tmpvec: Vec<(u64, u64, u8)> = Vec::new();
        let (thedict, maxmindict) =
            get_kmers_from_both_files_wasm::<IntT>(file1, file2, k, qual, &mut tmpvec);

        logw("k-mers extracted", Some("info"));

        // Then, we want to sort it according to the hash
        // log::debug!("Number of kmers BEFORE cleaning: {:?}", tmpvec.len());
        logw("Sorting vector", Some("info"));
        post_state("preprocess:bulk:sorting");
        tmpvec.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));

        logw("k-mers sorted.", Some("info"));

        // Then, do a counting of everything and save the results in a dictionary and return it
        logw("Counting k-mers", Some("info"));
        let (themap, mut histovec, used_min_count) =
            get_map_wasm(&mut tmpvec, qual.min_count, do_fit);
        histovec.shrink_to_fit();
        drop(tmpvec);

        logw("k-mers counted.", Some("info"));

        (themap, Some(thedict), maxmindict, histovec, used_min_count)
    } else {
        // Build indexes
        logw(
            format!("Processing in chunks of size {} the input files", csize).as_str(),
            Some("info"),
        );

        let mut tmpvec: Vec<(u64, u64, u8)> = Vec::new();
        let (thedict, maxmindict, themap, mut histovec, used_min_count) =
            chunked_processing_wasm::<IntT>(file1, file2, k, qual, &mut tmpvec, csize, do_fit);
        drop(tmpvec);
        histovec.shrink_to_fit();
        (themap, Some(thedict), maxmindict, histovec, used_min_count)
    }
}

