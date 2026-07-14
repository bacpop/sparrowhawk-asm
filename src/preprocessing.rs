//! Some docs should be here

use core::panic;

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
#[cfg(not(target_family = "wasm"))]
use crate::cli::{Counter, MultiKExtraction};
use crate::kmer::Kmer;
use crate::logw;
use crate::spectrum_fitter::SpectrumFitter;

/// Tuple for name and list of input files
pub type InputFastx = (String, Vec<String>);

/// Everything the preprocessing of one k value produces.
///
/// This replaces a 5-tuple that was returned in *two different field orders* — the inner backends
/// hand back `(thedict, maxmindict, themap, histovec, minc)` while the public entry point returned
/// `(themap, thedict, maxmindict, ...)` and silently re-ordered on destructuring. With more than one
/// k in flight that is a trap waiting to be sprung, so name the fields instead.
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

/// One FASTQ record, owned. needletail hands out a `Cow` borrowing its internal reader buffer, which
/// is invalidated on the next `next()`, so a batch that is to be processed in parallel must own its
/// bytes.
#[cfg(not(target_family = "wasm"))]
type OwnedRecord = (Vec<u8>, Option<Vec<u8>>);

/// How many records are handed to the workers at a time. At ~150 bp and k=31 a batch of 8192 records
/// yields ~1M k-mers, i.e. a few tens of MB of intermediate — small enough to stay cache-friendly,
/// large enough to amortise the rayon fork/join.
#[cfg(not(target_family = "wasm"))]
const BATCH_RECORDS: usize = 8192;

/// Parse `files` into owned batches of records, handing each batch to `on_batch`.
///
/// This is the batched twin of [`extract_kmers_from_files`]. Batching is what makes the per-record
/// k-mer work parallelisable: the parse itself stays serial (needletail is a serial reader), but it
/// overlaps with the workers chewing on the previous batch.
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

/// CPU bulk extraction: counts every k-mer straight into a [`CountMap`].
///
/// This replaces the old "push every occurrence into a flat vector, sort it, then run-length count
/// it" scheme. Bringing equal k-mers together is the only thing that vector ever did, and a hash map
/// does it in one probe per occurrence — whereas the old path paid a push, a 130M-element sort, *and*
/// two map probes (`outdict` + `minmaxdict`) per occurrence. Counting here is therefore strictly less
/// work, and it drops the multi-GB occurrence buffer entirely.
#[cfg(not(target_family = "wasm"))]
fn bulk_preprocessing_standalone_cpu<IntT>(
    files: &[String],
    k: usize,
    qual: &QualOpts,
) -> Vec<CountMap<IntT>>
where
    IntT: for<'a> UInt<'a> + Send,
{
    let mut shards: Vec<CountMap<IntT>> = (0..COUNTMAP_SHARDS)
        .map(|_| HashMap::with_hasher(BuildHasherDefault::default()))
        .collect();
    // Reused across batches so the per-batch split does not churn allocations.
    let mut buckets: Vec<Vec<(u64, u64, u8, IntT)>> =
        (0..COUNTMAP_SHARDS).map(|_| Vec::new()).collect();

    extract_kmers_from_files_batched(files, BATCH_RECORDS, |batch| {
        // Parallel: hash the batch. `km` is materialised here because the counting stage below cannot
        // reach back into the k-mer iterator.
        let items: Vec<(u64, u64, u8, IntT)> = hash_batch::<IntT>(batch, k, qual.min_qual);
        absorb_into_shards(items, &mut shards, &mut buckets);
    });

    shards
}

/// Count one batch's occurrences into the sharded count-map.
///
/// Split by shard, then count each shard on its own thread. Shards partition the key space, so no two
/// threads can ever reach the same entry — no locking, and each map is a fraction of the working set,
/// which probes far better than one big one.
///
/// `hnc`, `b` and `km` are identical for every occurrence of a given `hc` (they describe the k-mer, not
/// the sighting), so recording them on first sight leaves the result independent of the order in which
/// batches, records or threads arrive.
#[cfg(not(target_family = "wasm"))]
fn absorb_into_shards<IntT>(
    items: Vec<(u64, u64, u8, IntT)>,
    shards: &mut [CountMap<IntT>],
    buckets: &mut [Vec<(u64, u64, u8, IntT)>],
) where
    IntT: for<'a> UInt<'a>,
{
    for bucket in buckets.iter_mut() {
        bucket.clear();
    }
    for item in items {
        buckets[shard_of(item.0)].push(item);
    }

    shards
        .par_iter_mut()
        .zip(buckets.par_iter_mut())
        .for_each(|(map, bucket)| {
            for (hc, hnc, b, km) in bucket.drain(..) {
                map.entry(hc)
                    .and_modify(|e| e.count = e.count.saturating_add(1))
                    .or_insert(KmerInfo {
                        count: 1,
                        hnc,
                        b,
                        km,
                    });
            }
        });
}

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

    // The residual chunk, i.e. the records left over after the last full one.
    //
    // This MUST sit outside the `if let Some(file2)` block above. It used to be nested inside it, so a
    // single-file input never counted its trailing partial chunk and silently lost the k-mers of up to
    // `csize - 1` records (149,999 at the browser's default).
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

/// Bloom-filter preprocessing.
///
/// Note its spectrum is **not** comparable to the exact counters': `KmerFilter` only records a k-mer
/// once the bloom filter has already seen it, so the histogram has no count-1 bin, and bloom false
/// positives inflate the rest. It is approximate by design.
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

/// One distinct k-mer, accumulated while counting in bulk.
///
/// `hnc`, `b` and `km` are properties of the *k-mer*, not of the *occurrence*, so they are recorded
/// once, on first sight, rather than once per occurrence as the old sorting vector did.
#[cfg(not(target_family = "wasm"))]
struct KmerInfo<IntT> {
    count: u16,
    hnc: u64,
    b: u8,
    km: IntT,
}

#[cfg(not(target_family = "wasm"))]
type CountMap<IntT> = HashMap<u64, KmerInfo<IntT>, BuildHasherDefault<NoHashHasher<u64>>>;

/// The count-map is split into shards so that counting can run on all cores: shards partition the key
/// space, so two threads never touch the same entry and no locking is needed. Smaller maps also probe
/// better — each shard is a fraction of the working set.
#[cfg(not(target_family = "wasm"))]
const COUNTMAP_SHARDS: usize = 16;

/// Pick a shard for a canonical hash, mixing first.
///
/// Neither end of `hc` can be used raw:
///
/// - **Not the high bits.** ntHash itself is uniform, but `hc = min(fwd, rc)` is not — the minimum of
///   two uniform values has a triangular density. Measured on real reads (k=31), the top 4 bits of `hc`
///   run from 12.2% in bin 0 down to 0.37% in bin 15, a 33x spread. Sharding on them put ~2x the ideal
///   load on shard 0 and left shard 15 all but empty, so 16 shards behaved like 8.
/// - **Not the low bits.** `NoHashHasher` passes the hash straight through and hashbrown indexes its
///   buckets with the low bits, so every key in a shard would land in the same buckets.
///
/// One multiply gives bits that are uniform and independent of both. Note an xor-fold does *not* work
/// here: it shuffles bits without removing the magnitude bias, and leaves the 33x spread intact.
#[cfg(not(target_family = "wasm"))]
#[inline(always)]
fn shard_of(hc: u64) -> usize {
    const MIX: u64 = 0x9E37_79B9_7F4A_7C15; // odd, golden-ratio derived
    ((hc.wrapping_mul(MIX) >> (64 - COUNTMAP_SHARDS.trailing_zeros())) as usize)
        & (COUNTMAP_SHARDS - 1)
}

/// Split the count-map into the two artefacts the assembler needs, keeping only k-mers seen at least
/// `minc` times.
///
/// Unlike [`drain_countmap_into_themap`], which *prunes* dictionaries that were already fully
/// populated, this one *builds* them, so k-mers below the threshold never enter a map at all.
#[cfg(not(target_family = "wasm"))]
fn drain_countmap_bulk<IntT>(
    shards: Vec<CountMap<IntT>>,
    minc: u16,
) -> (
    HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
    HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
)
where
    IntT: for<'a> UInt<'a>,
{
    let mut themap = HashMap::with_hasher(BuildHasherDefault::default());
    let mut thedict = HashMap::with_hasher(BuildHasherDefault::default());

    // Shards are drained in order, so the result is independent of how many threads did the counting.
    for shard in shards {
        for (hc, info) in shard {
            if info.count >= minc {
                themap.insert(
                    hc,
                    HashInfoSimple {
                        hnc: info.hnc,
                        b: info.b,
                        pre: Vec::new(),
                        post: Vec::new(),
                        counts: info.count,
                    },
                );
                thedict.insert(hc, info.km);
            }
        }
    }

    (themap, thedict)
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

/// Hash one batch of records at a single k, in parallel.
///
/// Both counters used to carry this block verbatim. It is the *single-k* path, and it is kept exactly
/// as it was: the order of the occurrences it returns fixes the insertion order of `outdict`/`themap`,
/// which fixes the hash-map iteration order, which fixes the petgraph node numbering in the GFA/DOT
/// dumps. Contigs do not depend on it, but the graph dumps do, so do not "tidy" this into the multi-k
/// version below.
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

/// Hash one batch at *every* k in `ks`.
///
/// This is what "extract both k in one go" actually buys, and it is worth being precise: the file read,
/// the needletail record decode, and the owned-`Vec` copy of every record happen **once** instead of
/// once per k. That last one is the real prize — the batched reader must allocate a `Vec` per record,
/// because needletail's `Cow` borrows a buffer it invalidates on the very next record.
///
/// It does **not** save the hashing. ntHash's rolling state is k-dependent (k indexes the multi-shift
/// tables), so each k genuinely needs its own roller and the roll cost is inherently k-fold.
///
/// Note we call `hash_batch` once per k over the whole batch, rather than rolling every k inside a
/// single pass over the records. The single-pass version is the obvious thing to write and it is
/// *measurably slower*: gathering per-k vectors needs a `fold`/`reduce`, and rayon's reduce combines
/// accumulators pairwise up a tree, memcpying the whole occurrence stream at every level. It cost
/// ~35% on this workload. Re-walking the batch is nearly free by comparison — a batch is ~1 MB, so it
/// stays in cache — and it keeps `flat_map_iter().collect()`, which writes straight into the final
/// buffer. It also gives each k byte-for-byte the same occurrence stream as the sequential path, which
/// is what makes the two extraction modes agree exactly.
#[cfg(not(target_family = "wasm"))]
fn hash_batch_multik<IntT>(
    batch: &[OwnedRecord],
    ks: &[usize],
    min_qual: u8,
) -> Vec<Vec<(u64, u64, u8, IntT)>>
where
    IntT: for<'a> UInt<'a>,
{
    ks.iter()
        .map(|&k| hash_batch::<IntT>(batch, k, min_qual))
        .collect()
}

/// The **sort** counter: buffer k-mer occurrences, sort them so equal k-mers become adjacent, and
/// run-length count them.
///
/// `csize` bounds how many *records* worth of occurrences are buffered before a sort+count flush, and
/// is therefore the memory knob. `usize::MAX` means "one unbounded chunk", which is what the old bulk
/// path was — the two are the same algorithm, so there is only this one implementation.
#[cfg(not(target_family = "wasm"))]
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
    let mut countmap: HashMap<u64, (u16, u64, u8), BuildHasherDefault<NoHashHasher<u64>>> =
        HashMap::with_hasher(BuildHasherDefault::default());

    let histovec: Vec<u32> = vec![0; MAXSIZEHISTO];
    let mut i_record = 0;
    // let mut ncols : usize = 0;

    // The k-mer/hash work runs in parallel over a batch of records; the dictionaries are filled
    // afterwards in a tight serial loop. Keeping the dictionary probes out of the hot loop matters on
    // its own: interleaving `outvec.push` with two random hash-map probes evicts the rolling-hash
    // working set on every k-mer, and costs about as much again as the hashing itself.
    //
    // A chunk therefore closes at the first batch boundary at or past `csize` records, rather than at
    // exactly `csize`. `csize` is a memory hint, not a contract, so that is fine — and it is what lets
    // the extraction be batched at all.
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

    // The residual chunk. Note this sits outside the extraction closure, so it runs whatever the input
    // was — the wasm twin of this function had the equivalent flush nested inside its `if let
    // Some(file2)`, which silently dropped the tail of any single-file input.
    if !outvec.is_empty() {
        log::info!("Processing last chunk. Sorting k-mers...");
        outvec.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
        log::info!("k-mers sorted. Counting k-mers...");
        update_countmap(outvec, &mut countmap);
        outvec.clear();
    }

    finish_sort_counter(countmap, outdict, minmaxdict, histovec, qual, do_fit, out_path)
}

/// Fit, filter and plot a finished sort-counter, whatever drove it.
///
/// Split out of `chunked_preprocessing_standalone` so the joint multi-k path counts into exactly the
/// same structures and then finishes through exactly the same code — the two must agree to the byte,
/// and the cheapest way to guarantee that is to have one implementation.
#[cfg(not(target_family = "wasm"))]
#[allow(clippy::type_complexity)]
fn finish_sort_counter<IntT>(
    mut countmap: HashMap<u64, (u16, u64, u8), BuildHasherDefault<NoHashHasher<u64>>>,
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

/// Fit, filter and plot a finished sharded count-map, whatever drove it.
///
/// Split out of `preprocessing_standalone` for the same reason as [`finish_sort_counter`]: the joint
/// multi-k path must finish through identical code, or the two extraction modes could silently diverge.
#[cfg(not(target_family = "wasm"))]
#[allow(clippy::type_complexity)]
fn finish_map_counter<IntT>(
    shards: Vec<CountMap<IntT>>,
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
    // The slowest shard sets the pace of the parallel counting, so report the balance rather than
    // assume it: an imbalanced `shard_of` silently costs parallelism, which is exactly what a naive
    // high-bit shard selector did here (see `shard_of`).
    let sizes: Vec<usize> = shards.iter().map(|s| s.len()).collect();
    let total: usize = sizes.iter().sum();
    log::debug!("Number of distinct kmers BEFORE cleaning: {total:?}");
    if total > 0 {
        let ideal = total as f64 / sizes.len() as f64;
        let worst = *sizes.iter().max().unwrap() as f64 / ideal;
        log::info!("Count-map shard balance: busiest shard {worst:.2}x ideal");
        log::debug!("Count-map shard sizes: {sizes:?}");
        if worst > 1.5 {
            log::warn!(
                "Count-map shards are badly imbalanced (busiest {worst:.2}x ideal). Parallel \
                 counting is limited by the busiest shard, so this costs speed."
            );
        }
    }

    // The count-map holds every distinct k-mer with its exact count, singletons included, so the
    // spectrum is the same one the sort counter produces: count everything, then histogram, then fit,
    // then filter.
    let mut histovec = vec![0_u32; MAXSIZEHISTO];
    for shard in &shards {
        for info in shard.values() {
            add_to_histogram(&mut histovec, info.count);
        }
    }

    let minc = if do_fit {
        log::info!("Counting finished. Starting fit...");
        let minc = apply_spectrum_fit(&histovec);
        log::info!("Fit done! Fitted min_count value: {minc}. Starting filtering...");
        minc
    } else {
        qual.min_count
    };

    let (themap, thedict) = drain_countmap_bulk::<IntT>(shards, minc);

    // Falls out of `themap` for free: `HashInfoSimple` already carries `hnc`, and every consumer
    // (`check_bkg`/`check_fwd`) re-validates the hash it gets back against `themap` anyway, so entries
    // for non-surviving k-mers were only ever dead weight.
    let maxmindict: HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>> =
        themap.iter().map(|(hc, hi)| (hi.hnc, *hc)).collect();

    if let Some(p) = out_path {
        plot_kmer_histogram(&histovec, p.as_path());
    }

    (thedict, maxmindict, themap, histovec, minc)
}

/// Preprocess every k in `ks`, returning one [`PreprocessedK`] per k, in the same order.
///
/// `Sequential` simply runs the single-k path once per k: two passes over the reads, but the caller can
/// drop each k's structures before starting the next, which is the whole point of offering it.
///
/// `Joint` reads the files once and rolls every k on each record. It saves the I/O, the record decode
/// and the owned-copy — not the hashing, which is inherently per-k. Both must produce identical counts.
#[cfg(not(target_family = "wasm"))]
#[allow(clippy::too_many_arguments)]
pub fn preprocessing_standalone_multik<IntT>(
    input_files: &[InputFastx],
    ks: &[usize],
    qual: &QualOpts,
    timevec: &mut Vec<Instant>,
    out_paths: &mut [Option<PathBuf>],
    csize: usize,
    counter: Counter,
    do_bloom: bool,
    do_fit: bool,
    extraction: MultiKExtraction,
) -> Vec<PreprocessedK<IntT>>
where
    IntT: for<'a> UInt<'a>,
{
    assert_eq!(
        ks.len(),
        out_paths.len(),
        "one histogram path is needed per k"
    );

    // A single k is a single k however you slice it. The bloom filter owns its own per-k structure and
    // has no joint form, so it lands here too rather than growing a second implementation.
    if ks.len() == 1 || extraction == MultiKExtraction::Sequential || do_bloom {
        if ks.len() > 1 {
            log::info!(
                "Extracting k-mers sequentially: one pass over the reads per k ({} passes)",
                ks.len()
            );
        }
        return ks
            .iter()
            .zip(out_paths.iter_mut())
            .map(|(&k, out_path)| {
                preprocessing_standalone::<IntT>(
                    input_files,
                    k,
                    qual,
                    timevec,
                    out_path,
                    csize,
                    counter,
                    do_bloom,
                    do_fit,
                )
            })
            .collect();
    }

    log::info!("Extracting k-mers jointly for k = {ks:?}: one pass over the reads");

    let mut all_files: Vec<String> = input_files[0].1.clone();
    if input_files.len() > 1 {
        for ifile in input_files.iter().skip(1) {
            all_files.extend(ifile.1.clone());
        }
    }

    let nk = ks.len();
    let results = match counter {
        Counter::Sort => {
            // Same guard as the single-k path: `i_record >= 0` holds on every record, so a raw 0 would
            // sort and count after every single read rather than never.
            let csize = if csize == 0 { usize::MAX } else { csize };

            let mut outvecs: Vec<Vec<(u64, u64, u8)>> = (0..nk).map(|_| Vec::new()).collect();
            let mut countmaps: Vec<HashMap<u64, (u16, u64, u8), BuildHasherDefault<NoHashHasher<u64>>>> =
                (0..nk)
                    .map(|_| HashMap::with_hasher(BuildHasherDefault::default()))
                    .collect();
            let mut outdicts: Vec<HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>> = (0..nk)
                .map(|_| HashMap::with_hasher(BuildHasherDefault::default()))
                .collect();
            let mut minmaxdicts: Vec<HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>> = (0
                ..nk)
                .map(|_| HashMap::with_hasher(BuildHasherDefault::default()))
                .collect();

            // One record counter for all k: they see the same batches, so their chunks close together.
            let mut i_record = 0usize;

            extract_kmers_from_files_batched(&all_files, BATCH_RECORDS, |batch| {
                let per_k = hash_batch_multik::<IntT>(batch, ks, qual.min_qual);

                for (ik, items) in per_k.into_iter().enumerate() {
                    outvecs[ik].extend(items.iter().map(|&(hc, hnc, b, _)| (hc, hnc, b)));
                    for (hc, hnc, _, km) in items {
                        outdicts[ik].entry(hc).or_insert(km);
                        minmaxdicts[ik].entry(hnc).or_insert(hc);
                    }
                }

                i_record += batch.len();
                if i_record >= csize {
                    for ik in 0..nk {
                        if !outvecs[ik].is_empty() {
                            outvecs[ik].par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
                            update_countmap(&outvecs[ik], &mut countmaps[ik]);
                        }
                        outvecs[ik].clear();
                    }
                    i_record = 0;
                }
            });

            // The residual chunk, outside the closure so it runs whatever the input was.
            for ik in 0..nk {
                if !outvecs[ik].is_empty() {
                    outvecs[ik].par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
                    update_countmap(&outvecs[ik], &mut countmaps[ik]);
                    outvecs[ik].clear();
                }
            }
            drop(outvecs);

            countmaps
                .into_iter()
                .zip(outdicts)
                .zip(minmaxdicts)
                .zip(out_paths.iter_mut())
                .map(|(((countmap, outdict), minmaxdict), out_path)| {
                    finish_sort_counter::<IntT>(
                        countmap,
                        outdict,
                        minmaxdict,
                        vec![0; MAXSIZEHISTO],
                        qual,
                        do_fit,
                        out_path,
                    )
                })
                .collect::<Vec<_>>()
        }
        Counter::Map => {
            if csize != 0 {
                log::warn!(
                    "--chunk-size is ignored by --counter map: it buffers no occurrences, so its \
                     memory is bounded by the number of distinct k-mers instead."
                );
            }

            let mut all_shards: Vec<Vec<CountMap<IntT>>> = (0..nk)
                .map(|_| {
                    (0..COUNTMAP_SHARDS)
                        .map(|_| HashMap::with_hasher(BuildHasherDefault::default()))
                        .collect()
                })
                .collect();
            let mut buckets: Vec<Vec<(u64, u64, u8, IntT)>> =
                (0..COUNTMAP_SHARDS).map(|_| Vec::new()).collect();

            extract_kmers_from_files_batched(&all_files, BATCH_RECORDS, |batch| {
                let per_k = hash_batch_multik::<IntT>(batch, ks, qual.min_qual);
                for (ik, items) in per_k.into_iter().enumerate() {
                    absorb_into_shards(items, &mut all_shards[ik], &mut buckets);
                }
            });

            all_shards
                .into_iter()
                .zip(out_paths.iter_mut())
                .map(|(shards, out_path)| finish_map_counter::<IntT>(shards, qual, do_fit, out_path))
                .collect::<Vec<_>>()
        }
    };

    timevec.push(Instant::now());
    log::info!(
        "k-mers extracted, counted and filtered for all k in {} s",
        timevec
            .last()
            .unwrap()
            .duration_since(*timevec.get(timevec.len().wrapping_sub(2)).unwrap())
            .as_secs()
    );

    results
        .into_iter()
        .zip(ks)
        .map(|((thedict, maxmindict, themap, histovec, used_min_count), &k)| {
            log::info!("k={k}: minimum count per k-mer used: {used_min_count}");
            PreprocessedK {
                k,
                themap,
                thedict,
                maxmindict,
                histovec,
                used_min_count,
            }
        })
        .collect()
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
    counter: Counter,
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
        match counter {
            Counter::Sort => {
                // Chunking only ever bounded the occurrence buffer, so "no chunking" is simply one
                // unbounded chunk. Guarding here matters: `i_record >= 0` is true on every record, so
                // passing 0 straight through would sort and count after every single read.
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
            }
            Counter::Map => {
                if csize != 0 {
                    log::warn!(
                        "--chunk-size is ignored by --counter map: it buffers no occurrences, so its \
                         memory is bounded by the number of distinct k-mers instead."
                    );
                }
                log::info!("EXPERIMENTAL: counting k-mers into a hash map (no sort)");

                let shards = bulk_preprocessing_standalone_cpu::<IntT>(&all_files, k, qual);
                finish_map_counter::<IntT>(shards, qual, do_fit, out_path)
            }
        }
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

    fn empty_countmap() -> HashMap<u64, (u16, u64, u8), BuildHasherDefault<NoHashHasher<u64>>> {
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
        // Chunking only ever bounded the occurrence buffer, so "no chunking" is one unbounded chunk.
        // The guard matters: `i_record >= 0` is true on every record, so passing 0 straight through
        // would sort and count after every single read.
        //
        // The old separate bulk path (`get_kmers_from_both_files_wasm` + `get_map_wasm`) is gone: it
        // was the same sort-and-count algorithm, and it returned `thedict`/`maxmindict` *unpruned*,
        // still holding every singleton, unlike this one and unlike native.
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
            chunked_processing_wasm::<IntT>(file1, file2, k, qual, &mut tmpvec, csize, do_fit);
        drop(tmpvec);
        histovec.shrink_to_fit();
        (themap, Some(thedict), maxmindict, histovec, used_min_count)
    }
}
