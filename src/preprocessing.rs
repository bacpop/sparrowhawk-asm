//! Some docs should be here

use core::panic;

#[cfg(not(feature = "wasm"))]
use std::{path::PathBuf, time::Instant};

use nohash_hasher::NoHashHasher;
use std::{cell::*, cmp::Ordering, collections::HashMap, hash::BuildHasherDefault};

use rayon::prelude::*;

// use std::process::exit;

#[cfg(not(feature = "wasm"))]
use plotters::prelude::*;

use super::HashInfoSimple;
use super::QualOpts;

// #[cfg(not(feature = "wasm"))]
// use super::bit_encoding::{encode_base, rc_base};

use crate::bit_encoding::UInt;
use crate::bloom_filter::KmerFilter;
use crate::kmer::Kmer;
use crate::logw;
use crate::spectrum_fitter::SpectrumFitter;

/// Tuple for name and list of input files
pub type InputFastx = (String, Vec<String>);

// #[cfg(feature = "wasm")]
// use wasm_bindgen::prelude::*;
#[cfg(feature = "wasm")]
use crate::fastx_wasm::open_fastq;
#[cfg(feature = "wasm")]
use crate::post_state;
#[cfg(feature = "wasm")]
use seq_io::fastq::Record;
#[cfg(feature = "wasm")]
use wasm_bindgen_file_reader::WebSysFile;

// For the fitting, we'll use actually MAXSIZEHISTO - 1
const MAXSIZEHISTO: usize = 500;

// =====================================================================================================

// NOTE: these two functions were implemented to save the whole sequence in memory for GPGPU processing.
// This might be useful again in the future, but now it was just meaning that we were wasting memory!
// I'm disabling them for the moment
//
// #[cfg(not(feature = "wasm"))]
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

// #[cfg(not(feature = "wasm"))]
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

#[cfg(not(feature = "wasm"))]
fn bulk_preprocessing_standalone<IntT, I>(
    input_iters: &mut [I],
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
    I: Iterator<Item=(Vec<u8>, Option<Vec<u8>>)>,
{
    let mut outdict = HashMap::with_hasher(BuildHasherDefault::default());
    let mut minmaxdict = HashMap::with_hasher(BuildHasherDefault::default());

    // let mut itrecord : u32 = 0;                                 // We're using it to add the previous indexes!
    // let mut theseq   : Vec<u64> = Vec::new();
    let theseq: Vec<u64> = Vec::new();

    // Memory usage optimisations
    // let cseq = theseq.capacity();
    // let lseq = theseq.len();
    // if cseq < 2 * lseq {
    //     theseq.reserve_exact(2 * lseq);
    // } else {
    //     theseq.shrink_to(2 * lseq);
    // }

    for (idx, iter) in input_iters.iter_mut().enumerate() {
        log::info!("Getting kmers from file number {idx}...");
        log::info!("Parsing...");

        //     let maxkmers = 200;
        //     let mut numkmers = 0;
        for (seq, seq_qual) in iter {
            // put_these_nts_into_an_efficient_vector(&seqrec.seq(), &mut theseq, (itrecord % 32) as u8);

            let rl = seq.len();
            let kmer_opt = Kmer::<IntT>::new(
                seq.into(),
                rl,
                seq_qual.as_deref(),
                k,
                qual.min_qual,
                true,
                // &itrecord,
            );
            if let Some(mut kmer_it) = kmer_opt {
                //             numkmers += 1;
                let (hc, hnc, b, km) = kmer_it.get_curr_kmerhash_and_bases_and_kmer();
                outvec.push((hc, hnc, b));
                outdict.entry(hc).or_insert(km);
                // let testkm = outdict.entry(hc).or_insert(km);
                // if *testkm != km {
                //     log::debug!("\n\t- COLLISIONS 1 !!! Hash: {:?}", hc);
                //     log::debug!("{:#0258b}\n{:#0258b}", *testkm, km);
                //     ncols += 1;
                // }
                // } else {
                //     log::debug!("\n\t\t- NOT COLLISIONS!!!");
                // }
                minmaxdict.entry(hnc).or_insert(hc);
                while let Some(tmptuple) = kmer_it.get_next_kmer_and_give_us_things() {
                    let (hc, hnc, b, km) = tmptuple;
                    outvec.push((hc, hnc, b));
                    outdict.entry(hc).or_insert(km);
                    // let testkm = outdict.entry(hc).or_insert(km);
                    // if *testkm != km {
                    //     log::debug!("\n\t- COLLISIONS 2 !!! Hash: {:?}", hc);
                    //     log::debug!("{:#0258b}\n{:#0258b}", *testkm, km);
                    // }
                    // } else {
                    //     log::debug!("\n\t\t- NOT COLLISIONS!!!");
                    // }
                    minmaxdict.entry(hnc).or_insert(hc);

                    //                 numkmers += 1;
                    //                 if numkmers >= maxkmers {break};
                }
            }
            //         if numkmers >= maxkmers {itrecord += rl as u32;break};
            // itrecord += rl as u32;
        }
        log::info!("Finished getting kmers");
        //     numkmers = 0;
    }

    //     if (itrecord % 32) != 0 {
    //         let mut tmpu64 = theseq.pop().unwrap();
    //         tmpu64 <<= 2 * (32 - itrecord % 32);
    // //         log::debug!("{:#066b}", tmpu64);
    //         theseq.push(tmpu64);
    //     }

    log::info!("Finished getting kmers from {} file(s)", input_iters.len());
    // log::debug!("Length of seq. vec.: {}, total length of both files: {}", theseq.len(), itrecord);
    // log::debug!("k | Number of collisions =+=+ {} {}", k, ncols);

    (theseq, outdict, minmaxdict)
}

#[cfg(feature = "wasm")]
fn get_kmers_from_both_files_wasm<IntT>(
    file1: &mut WebSysFile,
    file2: Option<&mut WebSysFile>,
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

    if let Some(f2) = file2 {
        let mut reader = open_fastq(f2);

        //     log::debug!("MITAD Length of seq. vec.: {}, total length of first files: {}, THING {}, number of recs: {}", theseq.len(), itrecord, (itrecord % 32) as u8, realit);
        // Memory usage optimisations
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
                // let testkm = outdict.entry(hc).or_insert(km);
                // if *testkm != km {
                //     logw(format!("\n\t- COLLISIONS 3 !!! Hash: {:?}", hc).as_str(), Some("warn"));
                //     logw(format!("{:#0258b}\n{:#0258b}", *testkm, km).as_str(), Some("warn"));
                // }
                // } else {
                //     log::debug!("\n\t\t- NOT COLLISIONS!!!");
                // }
                minmaxdict.entry(hnc).or_insert(hc);
                while let Some(tmptuple) = kmer_it.get_next_kmer_and_give_us_things() {
                    let (hc, hnc, b, km) = tmptuple;
                    outvec.push((hc, hnc, b));
                    outdict.entry(hc).or_insert(km);
                    // let testkm = outdict.entry(hc).or_insert(km);
                    // if *testkm != km {
                    //     logw(format!("\n\t- COLLISIONS 4 !!! Hash: {:?}", hc).as_str(), Some("warn"));
                    //     logw(format!("{:#0258b}\n{:#0258b}", *testkm, km).as_str(), Some("warn"));
                    // }
                    // } else {
                    //     log::debug!("\n\t\t- NOT COLLISIONS!!!");
                    // }
                    minmaxdict.entry(hnc).or_insert(hc);

                    //                 numkmers += 1;
                    //                 if numkmers >= maxkmers {break};
                }
            }
            //         if numkmers >= maxkmers {itrecord += rl as u32;break};
            // itrecord += rl as u32;
            count += 1;
            if percentageblock > 0 && count.is_multiple_of(percentageblock) {
                post_state(&format!(
                    "preprocess:bulk:loop:{:?}:{:?}",
                    count,
                    count / percentageblock * 5
                ));
            }
        }
    }

    logw("Finished getting kmers from the second file", Some("info"));
    post_state("preprocess:bulk:loop:end");

    (outdict, minmaxdict)
}

#[cfg(feature = "wasm")]
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
    HashMap<u64, RefCell<HashInfoSimple>, BuildHasherDefault<NoHashHasher<u64>>>,
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
                outvec.par_sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
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

    if let Some(f2) = file2 {
        let mut reader = open_fastq(f2);

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
                    outvec.par_sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
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

    if i_record > 0 {
        // Processssssss! And reset.
        if !outvec.is_empty() {
            logw("Processing last chunk. Sorting k-mers...", Some("info"));
            outvec.par_sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
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
        for (_, tup) in countmap.iter() {
            if tup.0 as usize > MAXSIZEHISTO {
                histovec[MAXSIZEHISTO - 1] = histovec[MAXSIZEHISTO - 1].saturating_add(1);
            } else {
                histovec[tup.0 as usize - 1] = histovec[tup.0 as usize - 1].saturating_add(1);
            }
        }

        // Remove the last bin, as it might affect the fit, but we want it in the vector to plot it in case the coverage is really
        // large (and so that we can detect it).
        logw("Counting finished. Starting fit...", Some("info"));
        let mut fit = SpectrumFitter::new();
        let result = fit.fit_histogram(histovec[..(MAXSIZEHISTO - 1)].to_vec());
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

        post_state("preprocess:chunked:filtering");
        countmap.retain(|h, tup| {
            if tup.0 >= minc {
                themap.entry(*h).or_insert(RefCell::new(HashInfoSimple {
                    hnc: tup.1,
                    b: tup.2,
                    pre: Vec::new(),
                    post: Vec::new(),
                    counts: tup.0,
                }));
            } else {
                outdict.remove(h);
                minmaxdict.remove(&tup.1);
            }

            false
        });
    } else {
        post_state("preprocess:chunked:filtering");
        countmap.retain(|h, tup| {
            if tup.0 >= minc {
                themap.entry(*h).or_insert(RefCell::new(HashInfoSimple {
                    hnc: tup.1,
                    b: tup.2,
                    pre: Vec::new(),
                    post: Vec::new(),
                    counts: tup.0,
                }));
            } else {
                outdict.remove(h);
                minmaxdict.remove(&tup.1);
            }

            if tup.0 as usize > MAXSIZEHISTO {
                histovec[MAXSIZEHISTO - 1] = histovec[MAXSIZEHISTO - 1].saturating_add(1);
            } else {
                histovec[tup.0 as usize - 1] = histovec[tup.0 as usize - 1].saturating_add(1);
            }

            false
        });
    }

    outdict.shrink_to_fit();
    minmaxdict.shrink_to_fit();
    histovec.shrink_to_fit();
    drop(countmap);

    (outdict, minmaxdict, themap, histovec, minc)
}

#[cfg(feature = "wasm")]
fn bloom_filter_preprocessing_wasm<IntT>(
    file1: &mut WebSysFile,
    file2: Option<&mut WebSysFile>,
    k: usize,
    qual: &QualOpts,
    do_fit: bool,
) -> (
    HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
    HashMap<u64, RefCell<HashInfoSimple>, BuildHasherDefault<NoHashHasher<u64>>>,
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
            let (hc, hnc, b, km) = kmer_it.get_curr_kmerhash_and_bases_and_kmer(); // TODO: potential really small improvement, get only hash for the bloom filter, then get the rest.
            if Ordering::is_eq(kmer_filter.filter(hc, hnc, b)) {
                outdict.entry(hc).or_insert(km);
                minmaxdict.entry(hnc).or_insert(hc);
            }
            while let Some(tmptuple) = kmer_it.get_next_kmer_and_give_us_things() {
                let (hc, hnc, b, km) = tmptuple;
                if Ordering::is_eq(kmer_filter.filter(hc, hnc, b)) {
                    outdict.entry(hc).or_insert(km);
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

    if let Some(f2) = file2 {
        let mut reader = open_fastq(f2);

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
                if Ordering::is_eq(kmer_filter.filter(hc, hnc, b)) {
                    outdict.entry(hc).or_insert(km);
                    minmaxdict.entry(hnc).or_insert(hc);
                }
                while let Some(tmptuple) = kmer_it.get_next_kmer_and_give_us_things() {
                    let (hc, hnc, b, km) = tmptuple;
                    if Ordering::is_eq(kmer_filter.filter(hc, hnc, b)) {
                        outdict.entry(hc).or_insert(km);
                        minmaxdict.entry(hnc).or_insert(hc);
                    }
                }
            }
            count += 1;
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
    let countmap = kmer_filter.get_counts_map();
    countmap.shrink_to_fit();

    // This can be optimised. also better written: I had to repeat the code for the retains, to try to improve slightly the running time in
    // case no autofitting is requested. In any case, it could be improved in the future.
    let mut minc = qual.min_count;
    if do_fit {
        post_state("preprocess:bloom:fitting");
        for (_, tup) in countmap.iter() {
            if tup.0 as usize > MAXSIZEHISTO {
                histovec[MAXSIZEHISTO - 1] = histovec[MAXSIZEHISTO - 1].saturating_add(1);
            } else {
                histovec[tup.0 as usize - 1] = histovec[tup.0 as usize - 1].saturating_add(1);
            }
        }

        // Remove the last bin, as it might affect the fit, but we want it in the vector to plot it in case the coverage is really
        // large (and so that we can detect it).
        logw("Counting finished. Starting fit...", Some("info"));
        let mut fit = SpectrumFitter::new();
        let result = fit.fit_histogram(histovec[..(MAXSIZEHISTO - 1)].to_vec());
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

        post_state("preprocess:bloom:filtering");
        countmap.retain(|h, tup| {
            if tup.0 >= minc {
                themap.entry(*h).or_insert(RefCell::new(HashInfoSimple {
                    hnc: tup.1,
                    b: tup.2,
                    pre: Vec::new(),
                    post: Vec::new(),
                    counts: tup.0,
                }));
            } else {
                outdict.remove(h);
                minmaxdict.remove(&tup.1);
            }

            false
        });
    } else {
        post_state("preprocess:bloom:filtering");
        // I think this part can be improved: now that the min_count error has been corrected, some conditionals could be removed here?
        countmap.retain(|h, tup| {
            if tup.0 >= minc {
                themap.entry(*h).or_insert(RefCell::new(HashInfoSimple {
                    hnc: tup.1,
                    b: tup.2,
                    pre: Vec::new(),
                    post: Vec::new(),
                    counts: tup.0,
                }));
            } else {
                outdict.remove(h);
                minmaxdict.remove(&tup.1);
            }

            if tup.0 as usize > MAXSIZEHISTO {
                histovec[MAXSIZEHISTO - 1] = histovec[MAXSIZEHISTO - 1].saturating_add(1);
            } else {
                histovec[tup.0 as usize - 1] = histovec[tup.0 as usize - 1].saturating_add(1);
            }
            false
        });
    }

    outdict.shrink_to_fit();
    minmaxdict.shrink_to_fit();

    (outdict, minmaxdict, themap, histovec, minc)
}

#[cfg(not(feature = "wasm"))]
fn bloom_filter_preprocessing_standalone<IntT, I>(
    input_iters: &mut [I],
    k: usize,
    qual: &QualOpts,
    do_fit: bool,
    out_path: &mut Option<PathBuf>,
) -> (
    HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
    HashMap<u64, RefCell<HashInfoSimple>, BuildHasherDefault<NoHashHasher<u64>>>,
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
    for (idx, iter) in input_iters.iter_mut().enumerate() {
        log::info!("Getting kmers from file number {idx}...");

        logw("Parsing...", Some("info"));

        for (seq, seq_qual) in iter {
            let rl = seq.len();
            let kmer_opt =
                Kmer::<IntT>::new(seq.into(), rl, seq_qual.as_deref(), k, qual.min_qual, true);
            if let Some(mut kmer_it) = kmer_opt {
                let (hc, hnc, b, km) = kmer_it.get_curr_kmerhash_and_bases_and_kmer(); // TODO: potential really small improvement, get only hash for the bloom filter, then get the rest.
                if Ordering::is_eq(kmer_filter.filter(hc, hnc, b)) {
                    outdict.entry(hc).or_insert(km);
                    minmaxdict.entry(hnc).or_insert(hc);
                }
                while let Some(tmptuple) = kmer_it.get_next_kmer_and_give_us_things() {
                    let (hc, hnc, b, km) = tmptuple;
                    if Ordering::is_eq(kmer_filter.filter(hc, hnc, b)) {
                        outdict.entry(hc).or_insert(km);
                        minmaxdict.entry(hnc).or_insert(hc);
                    }
                }
            }
        }

        logw(
            format!("Finished getting kmers.").as_str(),
            Some("info"),
        );
    }
    log::info!("Finished getting kmers from {} file(s)", input_iters.len());
    log::info!("Finishing filtering...");

    // Now, get themap, histovec, and filter outdict and minmaxdict
    let countmap = kmer_filter.get_counts_map();
    countmap.shrink_to_fit();
    let mut minc;

    // This can be optimised. also better written: I had to repeat the code for the retains, to try to improve slightly the running time in
    // case no autofitting is requested. In any case, it could be improved in the future.
    if do_fit {
        for (_, tup) in countmap.iter() {
            if tup.0 as usize > MAXSIZEHISTO {
                histovec[MAXSIZEHISTO - 1] = histovec[MAXSIZEHISTO - 1].saturating_add(1);
            } else {
                histovec[tup.0 as usize - 1] = histovec[tup.0 as usize - 1].saturating_add(1);
            }
        }

        // Remove the last bin, as it might affect the fit, but we want it in the vector to plot it in case the coverage is really
        // large (and so that we can detect it).
        log::info!("Starting fit...");
        let mut fit = SpectrumFitter::new();

        let result = fit.fit_histogram(histovec[..(MAXSIZEHISTO - 1)].to_vec());
        if let Ok(theres) = result {
            minc = theres as u16;
        } else {
            logw("Fit has not converged. A value of 3 will be used as minimum, as usually this happens when the remaining k-mers go to low values. You should check whether this value is appropiated or not by looking at the k-mer spectrum histogram.", Some("warn"));
            minc = 3;
        }

        if minc == 0 {
            panic!("Fitted min_count value is zero or negative!");
        } else if minc <= 10 {
            logw("Fit has converged to a value smaller than 10. A value of 3 will be used as minimum, as usually this happens when the remaining k-mers go to low values, where the fit might give bad results. You should check whether this value is appropiated or not by looking at the k-mer spectrum histogram.", Some("warn"));
            minc = 3;
        }

        log::info!(
            "Fit done! Minimum count value to be used: {}. Filtering k-mers...",
            minc
        );

        countmap.retain(|h, tup| {
            if tup.0 >= minc {
                themap.entry(*h).or_insert(RefCell::new(HashInfoSimple {
                    hnc: tup.1,
                    b: tup.2,
                    pre: Vec::new(),
                    post: Vec::new(),
                    counts: tup.0,
                }));
            } else {
                outdict.remove(h);
                minmaxdict.remove(&tup.1);
            }

            false
        });
    } else {
        log::info!("Filtering k-mers...");
        minc = qual.min_count;

        countmap.retain(|h, tup| {
            if tup.0 >= minc {
                themap.entry(*h).or_insert(RefCell::new(HashInfoSimple {
                    hnc: tup.1,
                    b: tup.2,
                    pre: Vec::new(),
                    post: Vec::new(),
                    counts: tup.0,
                }));
            } else {
                outdict.remove(h);
                minmaxdict.remove(&tup.1);
            }

            if tup.0 as usize > MAXSIZEHISTO {
                histovec[MAXSIZEHISTO - 1] = histovec[MAXSIZEHISTO - 1].saturating_add(1);
            } else {
                histovec[tup.0 as usize - 1] = histovec[tup.0 as usize - 1].saturating_add(1);
            }
            false
        });
    }

    outdict.shrink_to_fit();
    minmaxdict.shrink_to_fit();

    if out_path.is_some() {
        // Plotting!
        let backend = BitMapBackend::new(out_path.as_ref().unwrap().as_path(), (1280, 960));

        let root = backend.into_drawing_area();

        let _ = root.fill(&WHITE);

        let mut chart = ChartBuilder::on(&root)
            .x_label_area_size(35)
            .y_label_area_size(40)
            .margin(5)
            // .caption("k-mer spectrum", ("sans-serif", 30.0))
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
            // .axis_desc_style(("sans-serif", 15))
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

        // TODO: I have spent too much time trying to draw a vertical line or a rectangle to show in the
        //       histogram the fitted limit. Not more at least until I decide to lose more of my life.
        // https://stackoverflow.com/questions/78776201/how-to-dynamically-use-plotter-segmentvalue

        // backend.draw_rect(
        //     (0i32, 200000i32),
        //     (fitted_min_count as i32, 0i32),
        //     &BLACK,
        //     true,
        // ).unwrap();

        // let testnum = fitted_min_count as i32;
        // let rectangle = Rectangle::new(
        //     [(0, 200000), (5, 0)],
        //     BLUE.mix(0.5).filled(),
        // );
        //
        // chart.draw_series(std::iter::once(rectangle.into_dyn())).unwrap();

        // chart.plotting_area().draw(&rectangle).unwrap();

        // chart.draw_series(LineSeries::new(
        //     [(0i32, 0i32), (fitted_min_count as i32, 200000i32)].iter(),
        //     &BLUE,
        // )).unwrap();

        root.present()
            .expect("Unable to write result to file. Does the output folder exist?");
    }

    (outdict, minmaxdict, themap)
}

#[cfg(not(feature = "wasm"))]
fn get_map_with_counts(
    invec: &[(u64, u64, u8)],
    min_count: u16,
    out_path: &mut Option<PathBuf>,
) -> HashMap<u64, RefCell<HashInfoSimple>, BuildHasherDefault<NoHashHasher<u64>>> {
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
                        .or_insert(RefCell::new(HashInfoSimple {
                            hnc: invec[i - 1].1,
                            b: invec[i - 1].2,
                            pre: Vec::new(),
                            post: Vec::new(),
                            counts: c,
                        }));
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
                .or_insert(RefCell::new(HashInfoSimple {
                    hnc: invec[i - 1].1,
                    b: invec[i - 1].2,
                    pre: Vec::new(),
                    post: Vec::new(),
                    counts: c,
                }));
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
                            .chain(outdict.values().map(|x| (x.borrow().counts as u32, 1))),
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
                        .or_insert(RefCell::new(HashInfoSimple {
                            hnc: invec[i - 1].1,
                            b: invec[i - 1].2,
                            pre: Vec::new(),
                            post: Vec::new(),
                            counts: c,
                        }));
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
                .or_insert(RefCell::new(HashInfoSimple {
                    hnc: invec[i - 1].1,
                    b: invec[i - 1].2,
                    pre: Vec::new(),
                    post: Vec::new(),
                    counts: c,
                }));
        }
    }
    outdict
}

#[cfg(not(feature = "wasm"))]
fn get_map_with_counts_and_fit(
    invec: &mut Vec<(u64, u64, u8)>,
    out_path: &mut Option<PathBuf>,
) -> HashMap<u64, RefCell<HashInfoSimple>, BuildHasherDefault<NoHashHasher<u64>>> {
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
                .or_insert(RefCell::new(HashInfoSimple {
                    hnc: invec[i - 1].1,
                    b: invec[i - 1].2,
                    pre: Vec::new(),
                    post: Vec::new(),
                    counts: c,
                }));

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
        .or_insert(RefCell::new(HashInfoSimple {
            hnc: invec[i - 1].1,
            b: invec[i - 1].2,
            pre: Vec::new(),
            post: Vec::new(),
            counts: c,
        }));

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

    outdict.retain(|_, rc| rc.borrow().counts >= fitted_min_count);
    outdict.shrink_to_fit();

    if out_path.is_some() {
        // Plotting!
        let backend = BitMapBackend::new(out_path.as_ref().unwrap().as_path(), (1280, 960));

        let root = backend.into_drawing_area();

        let _ = root.fill(&WHITE);

        let mut chart = ChartBuilder::on(&root)
            .x_label_area_size(35)
            .y_label_area_size(40)
            .margin(5)
            // .caption("k-mer spectrum", ("sans-serif", 30.0))
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
            // .axis_desc_style(("sans-serif", 15))
            .axis_desc_style(("ibm-plex-sans", 15))
            .draw()
            .unwrap();

        chart
            .draw_series(
                Histogram::vertical(&chart)
                    .style(RED.filled())
                    .data(plotvec.iter().enumerate().map(|(i, x)| (i as u32, *x))),
            )
            .unwrap();

        // TODO: I have spent too much time trying to draw a vertical line or a rectangle to show in the
        //       histogram the fitted limit. Not more at least until I decide to lose more of my life.
        // https://stackoverflow.com/questions/78776201/how-to-dynamically-use-plotter-segmentvalue

        // backend.draw_rect(
        //     (0i32, 200000i32),
        //     (fitted_min_count as i32, 0i32),
        //     &BLACK,
        //     true,
        // ).unwrap();

        // let testnum = fitted_min_count as i32;
        // let rectangle = Rectangle::new(
        //     [(0, 200000), (5, 0)],
        //     BLUE.mix(0.5).filled(),
        // );
        //
        // chart.draw_series(std::iter::once(rectangle.into_dyn())).unwrap();

        // chart.plotting_area().draw(&rectangle).unwrap();

        // chart.draw_series(LineSeries::new(
        //     [(0i32, 0i32), (fitted_min_count as i32, 200000i32)].iter(),
        //     &BLUE,
        // )).unwrap();

        root.present()
            .expect("Unable to write result to file. Does the output folder exist?");
    }

    // log::debug!("Good kmers {}", tmpcounter);
    //     exit(1);

    outdict
}

#[cfg(feature = "wasm")]
fn get_map_wasm(
    invec: &mut Vec<(u64, u64, u8)>,
    min_count: u16,
    do_fit: bool,
) -> (
    HashMap<u64, RefCell<HashInfoSimple>, BuildHasherDefault<NoHashHasher<u64>>>,
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
                    .or_insert(RefCell::new(HashInfoSimple {
                        hnc: invec[i - 1].1,
                        b: invec[i - 1].2,
                        pre: Vec::new(),
                        post: Vec::new(),
                        counts: c,
                    }));

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
            .or_insert(RefCell::new(HashInfoSimple {
                hnc: invec[i - 1].1,
                b: invec[i - 1].2,
                pre: Vec::new(),
                post: Vec::new(),
                counts: c,
            }));

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
        outdict.retain(|_, rc| rc.borrow().counts >= minc);
        outdict.shrink_to_fit();
    } else {
        post_state("preprocess:bulk:filtering");
        while i < invec.len() {
            if tmphash != invec[i].0 {
                if c >= minc {
                    // tmpcounter += 1;
                    outdict
                        .entry(tmphash)
                        .or_insert(RefCell::new(HashInfoSimple {
                            hnc: invec[i - 1].1,
                            b: invec[i - 1].2,
                            pre: Vec::new(),
                            post: Vec::new(),
                            counts: c,
                        }));
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
                .or_insert(RefCell::new(HashInfoSimple {
                    hnc: invec[i - 1].1,
                    b: invec[i - 1].2,
                    pre: Vec::new(),
                    post: Vec::new(),
                    counts: c,
                }));
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

#[cfg(not(feature = "wasm"))]
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
    HashMap<u64, RefCell<HashInfoSimple>, BuildHasherDefault<NoHashHasher<u64>>>,
)
where
    IntT: for<'a> UInt<'a>,
    I: Iterator<Item=(Vec<u8>, Option<Vec<u8>>)>,
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

    for (idx, iter) in input_iters.iter_mut().enumerate() {
        log::info!("Getting kmers from file number {idx}...");

        log::info!("Parsing input...");
        for (seq, seq_qual) in iter {
            let rl = seq.len();
            let kmer_opt =
                Kmer::<IntT>::new(seq.into(), rl, seq_qual.as_deref(), k, qual.min_qual, true);
            if let Some(mut kmer_it) = kmer_opt {
                let (hc, hnc, b, km) = kmer_it.get_curr_kmerhash_and_bases_and_kmer();
                outvec.push((hc, hnc, b));
                outdict.entry(hc).or_insert(km);
                // let testkm = outdict.entry(hc).or_insert(km);
                // if *testkm != km {
                //     ncols += 1;
                // }
                minmaxdict.entry(hnc).or_insert(hc);
                while let Some(tmptuple) = kmer_it.get_next_kmer_and_give_us_things() {
                    let (hc, hnc, b, km) = tmptuple;
                    outvec.push((hc, hnc, b));
                    outdict.entry(hc).or_insert(km);
                    minmaxdict.entry(hnc).or_insert(hc);
                }
            }

            i_record += 1;
            if i_record >= csize {
                // Processssssss! And reset.
                if !outvec.is_empty() {
                    log::debug!("Processing chunk. Sorting k-mers...");
                    outvec.par_sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    log::debug!("k-mers sorted. Counting k-mers...");
                    // Then, do a counting of everything and save the results in a dictionary and return it

                    update_countmap(outvec, &mut countmap);
                }

                // Reset
                outvec.clear();
                i_record = 0;
            }
        }
        log::info!("Finished getting kmers.");
    }

    if i_record > 0 {
        // Processssssss! And reset.
        if !outvec.is_empty() {
            log::info!("Processing last chunk. Sorting k-mers...");
            outvec.par_sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
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
    let mut minc;

    // This can be optimised. also better written: I had to repeat the code for the retains, to try to improve slightly the running time in
    // case no autofitting is requested. In any case, it could be improved in the future.
    if do_fit {
        for (_, tup) in countmap.iter() {
            if tup.0 as usize > MAXSIZEHISTO {
                histovec[MAXSIZEHISTO - 1] = histovec[MAXSIZEHISTO - 1].saturating_add(1);
            } else {
                histovec[tup.0 as usize - 1] = histovec[tup.0 as usize - 1].saturating_add(1);
            }
        }

        // // TEST
        // for i in 0..histovec.len() {
        //     logw(format!("#######{:?}-{:?}", i, histovec[i]).as_str(), Some("info"));
        // }
        // // TEST END

        // Remove the last bin, as it might affect the fit, but we want it in the vector to plot it in case the coverage is really
        // large (and so that we can detect it).
        log::info!("Counting finished. Starting fit...");
        let mut fit = SpectrumFitter::new();
        // let minc = fit.fit_histogram(histovec.clone()[..(MAXSIZEHISTO - 1)].to_vec()).expect("Fit to the k-mer spectrum failed!") as u16;

        let result = fit.fit_histogram(histovec[..(MAXSIZEHISTO - 1)].to_vec());
        if let Ok(theres) = result {
            minc = theres as u16;
        } else {
            logw("Fit has not converged. A value of 3 will be used as minimum, as usually this happens when the remaining k-mers go to low values. You should check whether this value is appropiated or not by looking at the k-mer spectrum histogram.", Some("warn"));
            minc = 3;
        }

        if minc == 0 {
            panic!("Fitted min_count value is zero or negative!");
        } else if minc <= 10 {
            logw("Fit has converged to a value smaller than 10. A value of 3 will be used as minimum, as usually this happens when the remaining k-mers go to low values, where the fit might give bad results. You should check whether this value is appropiated or not by looking at the k-mer spectrum histogram.", Some("warn"));
            minc = 3;
        }

        log::info!(
            "Fit done! Fitted min_count value: {}. Starting filtering...",
            minc
        );

        countmap.retain(|h, tup| {
            if tup.0 >= minc {
                themap.entry(*h).or_insert(RefCell::new(HashInfoSimple {
                    hnc: tup.1,
                    b: tup.2,
                    pre: Vec::new(),
                    post: Vec::new(),
                    counts: tup.0,
                }));
            } else {
                outdict.remove(h);
                minmaxdict.remove(&tup.1);
            }

            false
        });
    } else {
        minc = qual.min_count;

        countmap.retain(|h, tup| {
            if tup.0 >= minc {
                themap.entry(*h).or_insert(RefCell::new(HashInfoSimple {
                    hnc: tup.1,
                    b: tup.2,
                    pre: Vec::new(),
                    post: Vec::new(),
                    counts: tup.0,
                }));
            } else {
                outdict.remove(h);
                minmaxdict.remove(&tup.1);
            }

            if tup.0 as usize > MAXSIZEHISTO {
                histovec[MAXSIZEHISTO - 1] = histovec[MAXSIZEHISTO - 1].saturating_add(1);
            } else {
                histovec[tup.0 as usize - 1] = histovec[tup.0 as usize - 1].saturating_add(1);
            }
            false
        });
    }

    drop(countmap);
    outdict.shrink_to_fit();
    minmaxdict.shrink_to_fit();

    if out_path.is_some() {
        // Plotting!
        let backend = BitMapBackend::new(out_path.as_ref().unwrap().as_path(), (1280, 960));

        let root = backend.into_drawing_area();

        let _ = root.fill(&WHITE);

        let mut chart = ChartBuilder::on(&root)
            .x_label_area_size(35)
            .y_label_area_size(40)
            .margin(5)
            // .caption("k-mer spectrum", ("sans-serif", 30.0))
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
            // .axis_desc_style(("sans-serif", 15))
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

        // TODO: I have spent too much time trying to draw a vertical line or a rectangle to show in the
        //       histogram the fitted limit. Not more at least until I decide to lose more of my life.
        // https://stackoverflow.com/questions/78776201/how-to-dynamically-use-plotter-segmentvalue

        // backend.draw_rect(
        //     (0i32, 200000i32),
        //     (fitted_min_count as i32, 0i32),
        //     &BLACK,
        //     true,
        // ).unwrap();

        // let testnum = fitted_min_count as i32;
        // let rectangle = Rectangle::new(
        //     [(0, 200000), (5, 0)],
        //     BLUE.mix(0.5).filled(),
        // );
        //
        // chart.draw_series(std::iter::once(rectangle.into_dyn())).unwrap();

        // chart.plotting_area().draw(&rectangle).unwrap();

        // chart.draw_series(LineSeries::new(
        //     [(0i32, 0i32), (fitted_min_count as i32, 200000i32)].iter(),
        //     &BLUE,
        // )).unwrap();

        root.present()
            .expect("Unable to write result to file. Does the output folder exist?");
    }

    (outdict, minmaxdict, themap)
}

/// Read fastq files, get the reads, get the k-mers, count them, filter them by count, and get some way of recovering the sequence later.
#[cfg(not(feature = "wasm"))]
pub fn preprocessing_standalone<IntT, I>(
    input_iters: &mut [I],
    k: usize,
    qual: &QualOpts,
    timevec: &mut Option<&mut Vec<Instant>>,
    out_path: &mut Option<PathBuf>,
    csize: usize,
    do_bloom: bool,
    do_fit: bool,
) -> (
    HashMap<u64, RefCell<HashInfoSimple>, BuildHasherDefault<NoHashHasher<u64>>>,
    Vec<u64>,
    HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
)
where
    IntT: for<'a> UInt<'a>,
    I: Iterator<Item=(Vec<u8>, Option<Vec<u8>>)>,
{
    log::info!("Starting preprocessing_standalone with k = {k}");

    if do_bloom {
        // Build indexes
        log::info!("Processing using a Bloom filter");

        let (thedict, maxmindict, themap) =
            bloom_filter_preprocessing_standalone::<IntT, _>(input_iters, k, qual, do_fit, out_path);
        (themap, Vec::new(), thedict, maxmindict)
    } else if csize == 0 {
        // First, we want to fill our mega-vector with all k-mers from both paired-end reads
        log::info!("Processing in bulk");

        let mut tmpvec: Vec<(u64, u64, u8)> = Vec::new();
        let (theseq, thedict, maxmindict) =
            bulk_preprocessing_standalone::<IntT, _>(input_iters, k, qual, &mut tmpvec);

        //exit(0);
        timevec.as_mut().map(|timevec| {
            timevec.push(Instant::now());
            log::info!(
                "k-mers extracted in {} s",
                timevec
                    .last()
                    .unwrap()
                    .duration_since(*timevec.get(timevec.len().wrapping_sub(2)).unwrap())
                    .as_secs()
            );
        });

        // Then, we want to sort it according to the hash
        log::debug!("Number of kmers BEFORE cleaning: {:?}", tmpvec.len());
        log::info!("Sorting vector");
        tmpvec.par_sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        timevec.as_mut().map(|timevec| {
            timevec.push(Instant::now());
            log::info!(
                "k-mers sorted in {} s",
                timevec
                    .last()
                    .unwrap()
                    .duration_since(*timevec.get(timevec.len().wrapping_sub(2)).unwrap())
                    .as_secs()
            );
        });

        // Then, do a counting of everything and save the results in a dictionary and return it
        log::info!("Counting and filtering k-mers");
        let themap = if !do_fit {
            get_map_with_counts(&tmpvec, qual.min_count, out_path)
        } else {
            get_map_with_counts_and_fit(&mut tmpvec, out_path)
        };
        drop(tmpvec);

        timevec.as_mut().map(|timevec| {
            timevec.push(Instant::now());
            log::info!(
                "k-mers counted and filtered in {} s",
                timevec
                    .last()
                    .unwrap()
                    .duration_since(*timevec.get(timevec.len().wrapping_sub(2)).unwrap())
                    .as_secs()
            );
        });

        (themap, theseq, thedict, maxmindict)
    } else {
        log::info!("Processing in chunks of size {}", csize);

        let mut tmpvec: Vec<(u64, u64, u8)> = Vec::new();
        let (thedict, maxmindict, themap) = chunked_preprocessing_standalone::<IntT, _>(
            input_iters,
            k,
            qual,
            &mut tmpvec,
            csize,
            do_fit,
            out_path,
        );
        drop(tmpvec);

        timevec.as_mut().map(|timevec| {
            timevec.push(Instant::now());
            log::info!(
                "Chunked preprocessing done in {} s",
                timevec
                    .last()
                    .unwrap()
                    .duration_since(*timevec.get(timevec.len().wrapping_sub(2)).unwrap())
                    .as_secs()
            );
        });
        (themap, Vec::new(), thedict, maxmindict)
    }
}

#[cfg(feature = "wasm")]
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
    HashMap<u64, RefCell<HashInfoSimple>, BuildHasherDefault<NoHashHasher<u64>>>,
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
        // Build indexes
        logw("Starting preprocessing with k = {k}", Some("info"));

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
        tmpvec.par_sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

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
