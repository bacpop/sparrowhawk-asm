//! Efficient genome assembler for small genomes in Rust
#![warn(missing_docs)]
use std::fmt;

#[cfg(target_family = "wasm")]
use std::{collections::HashMap, hash::BuildHasherDefault};

#[cfg(not(target_family = "wasm"))]
use std::{
    path::{Path, PathBuf},
    time::Instant,
    time::SystemTime,
};

extern crate num_cpus;

/// Construction, assembly, shrinkage, pruning, and collapse of DNA de Bruijn graphs
pub mod graph_works;

/// Native aligned k-mer storage between counting and graph construction.
#[cfg(not(target_family = "wasm"))]
pub mod indexed_kmers;

/// Reads packed once in memory, so a multi-k run parses its input a single time.
#[cfg(not(target_family = "wasm"))]
pub mod read_store;

/// Preprocessing functions of the reads & k-mers
pub mod preprocessing;

/// Candidate base-quality floors, and which of them each k-mer window clears
#[cfg(not(target_family = "wasm"))]
pub mod qual_profile;

/// Declarations and definitions for encode our k-mers efficiently in memory
pub mod bit_encoding;

/// A helper class to obtain k-mers from reads derived from Ska2
pub mod kmer;

/// An implementation of ntHash, based on ntHash 2
pub mod nthash;

/// Contains functions to store the output of the program
pub mod save_functions;

/// Contains different traits that implement various algorithms
pub mod algorithms;

/// Defines a bloom filter (taken from ska.rust!)
pub mod bloom_filter;

/// Fits the k-mer spectrum to automatically get a min_count (taken from ska.rust!
pub mod spectrum_fitter;

#[cfg(target_family = "wasm")]
use nohash_hasher::NoHashHasher;

use crate::graph_works::Assemble;
use bit_encoding::{U256, U512};

#[cfg(not(target_family = "wasm"))]
use crate::bit_encoding::UInt;
#[cfg(not(target_family = "wasm"))]
use crate::preprocessing::InputFastx;

// Re-export core graph types so callers do not need to depend on sparrowhawk-graph directly.
pub use sparrowhawk_graph::{EdgeType, EdgeWeight, HashInfoSimple, Idx};

pub mod cli;

#[cfg(not(target_family = "wasm"))]
use crate::cli::*;

pub mod io_utils;

#[cfg(not(target_family = "wasm"))]
use crate::io_utils::*;

#[cfg(target_family = "wasm")]
use wasm_bindgen::prelude::*;
#[cfg(target_family = "wasm")]
use wasm_bindgen_file_reader::WebSysFile;
#[cfg(target_family = "wasm")]
extern crate console_error_panic_hook;
#[cfg(target_family = "wasm")]
pub mod fastx_wasm;
#[cfg(target_family = "wasm")]
use crate::graph_works::Contigs;

/// Logging wrapper function for the WebAssembly version
#[cfg(target_family = "wasm")]
pub fn logw(text: &str, typ: Option<&str>) {
    if let Some(thetyp) = typ {
        log((String::from("Sparrowhawk::") + thetyp + "::" + text).as_str());
    } else {
        log(text);
    }
}

/// Logging wrapper function for the standalone version
#[cfg(not(target_family = "wasm"))]
pub fn logw(text: &str, typ: Option<&str>) {
    if let Some(realtyp) = typ {
        if realtyp == "info" {
            log::info!("{}", text);
        } else if realtyp == "debug" {
            log::debug!("{}", text);
        } else if realtyp == "trace" {
            log::trace!("{}", text);
        } else if realtyp == "warn" {
            log::warn!("{}", text);
        } else if realtyp == "error" {
            log::error!("{}", text);
        } else {
            println!("{}", text);
        }
    } else {
        println!("{}", text);
    }
}

/// Quality filtering options for FASTQ files
pub struct QualOpts {
    /// Minimum k-mer count across reads to be added
    pub min_count: u16,
    /// Minimum base quality to be added
    pub min_qual: u8,
}

impl fmt::Display for QualOpts {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "min count: {}; minimum quality {} ({});",
            self.min_count,
            self.min_qual,
            self.min_qual.saturating_add(33) as char,
        )
    }
}

#[cfg(not(target_family = "wasm"))]
/// Sets up logging
pub fn set_up_logging(level: log::LevelFilter, outfile: PathBuf) {
    fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "[{} {} {}] {}",
                humantime::format_rfc3339_seconds(SystemTime::now()),
                record.level(),
                record.target(),
                message
            ))
        })
        .level(level)
        .chain(std::io::stdout())
        .chain(fern::log_file(outfile).unwrap())
        .apply()
        .unwrap();
}

/// Everything the `build` pipeline needs, once `IntT` has been chosen.
///
/// The four k-width branches used to be four verbatim copies of the same fifteen-line body, differing
/// only in the integer type. Collapsing them into one generic function means a change to the pipeline
/// is made once instead of four times.
#[cfg(not(target_family = "wasm"))]
struct BuildOpts<'a> {
    input_files: &'a [InputFastx],
    quality: &'a QualOpts,
    chunk_size: usize,
    do_bloom: bool,
    /// Fit `min_count` from the k-mer spectrum rather than using `quality.min_count`.
    do_fit: bool,
    do_bubble_collapse: bool,
    do_dead_end_removal: bool,
    /// Fraction of the stronger branch's coverage below which the weaker branch of a bubble is popped.
    pop_ratio: f32,
    /// Fraction of single-copy coverage below which a branch is an error. `None` keeps the default
    /// for however the coverage was established.
    peak_ratio: Option<f32>,
    /// Flat dead-end threshold in bases, and the k multiple that may raise it; resolved per k.
    tip_length: usize,
    tip_length_kmult: f32,
    /// k multiple bounding the coverage-judged tip band; `None` under `--no-tip-rctc`.
    tip_length_rctc_kmult: Option<f32>,
    /// How many times better covered a junction must be than the tip hanging off it.
    tip_rctc_cutoff: f64,
    do_ec_removal: bool,
    ec_ratio: f64,
    ec_require_both_flanks: bool,
    /// Minimum contig sequence length written to FASTA.
    min_contig_length: usize,
    /// Multi-k: how k-mers carried over from the previous k's contigs are counted.
    contig_counts: ContigCountRule,
    /// Multi-k: also write each intermediate k's contigs.
    keep_intermediate_contigs: bool,
    no_histo: bool,
    output_dir: &'a str,
    output_prefix: &'a str,
    output: PathBuf,
}

#[cfg(not(target_family = "wasm"))]
impl BuildOpts<'_> {
    /// Correction settings for one k: tip thresholds scale with k, coverage comes from its spectrum.
    fn correction(
        &self,
        k: usize,
        peak: preprocessing::PeakSource,
        min_count: u16,
    ) -> algorithms::corrector::CorrectionOpts {
        use algorithms::corrector::{tip_length_nts, CorrectionOpts, CoverageRef};
        CorrectionOpts {
            do_bubble_collapse: self.do_bubble_collapse,
            do_dead_end_removal: self.do_dead_end_removal,
            pop_ratio: self.pop_ratio,
            tip_nts: tip_length_nts(self.tip_length, self.tip_length_kmult, k),
            // Zero empties the band: `.max(limit)` in `remove_dead_paths`.
            tip_rctc_nts: self
                .tip_length_rctc_kmult
                .map_or(0, |m| tip_length_nts(self.tip_length, m, k)),
            tip_rctc_cutoff: self.tip_rctc_cutoff,
            coverage: CoverageRef::new(peak, min_count).with_error_fraction(self.peak_ratio),
            do_ec_removal: self.do_ec_removal,
            ec_ratio: self.ec_ratio,
            ec_require_both_flanks: self.ec_require_both_flanks,
        }
    }

    /// One spectrum plot's path, or `None` under `--no-histo`. `tag` tells the k of a ladder apart.
    fn histo_path(&self, tag: &str) -> Option<PathBuf> {
        (!self.no_histo).then(|| {
            Path::new(self.output_dir).join(format!("{}_kmerspectrum{tag}.png", self.output_prefix))
        })
    }
}

/// Open the inputs and count k-mers at one quality floor. With `store`, every record is also packed
/// into it as it streams past, which is how a multi-k run's first k reads the files for all of them.
#[cfg(not(target_family = "wasm"))]
fn count_reads<IntT>(
    opts: &BuildOpts,
    k: usize,
    quality: &QualOpts,
    floors: Option<&[u8]>,
    store: Option<&mut read_store::ReadStore>,
    timevec: &mut Vec<Instant>,
    out_path_histo: &mut Option<PathBuf>,
) -> Result<preprocessing::PreprocessedK<IntT>, preprocessing::PreprocessingError>
where
    IntT: for<'a> UInt<'a>,
{
    let mut readers = opts
        .input_files
        .iter()
        .flat_map(|(_, files)| {
            files
                .iter()
                .map(|file| {
                    let reader = needletail::parse_fastx_file(file)
                        .unwrap_or_else(|_| panic!("Invalid path/file: {file}"));
                    NeedletailIterator::new(reader)
                })
                .collect::<Vec<NeedletailIterator>>()
        })
        .collect::<Vec<NeedletailIterator>>();

    match store {
        // One chained reader, so the single `&mut` store sees every record, in file order.
        Some(store) => preprocessing::preprocessing_standalone::<IntT, _>(
            &mut [read_store::Tee::new(readers.into_iter().flatten(), store)],
            k,
            quality,
            floors,
            &mut Some(timevec),
            out_path_histo,
            opts.chunk_size,
            opts.do_bloom,
            opts.do_fit,
            None,
        ),
        None => preprocessing::preprocessing_standalone::<IntT, _>(
            &mut readers,
            k,
            quality,
            floors,
            &mut Some(timevec),
            out_path_histo,
            opts.chunk_size,
            opts.do_bloom,
            opts.do_fit,
            None,
        ),
    }
}

/// Run the whole `build` pipeline, monomorphised on the packed-k-mer width.
#[cfg(not(target_family = "wasm"))]
fn run_build<IntT>(
    opts: BuildOpts,
    k: usize,
    timevec: &mut Vec<Instant>,
    out_path_graph: &mut Option<PathBuf>,
) -> Result<(), preprocessing::PreprocessingError>
where
    IntT: for<'a> UInt<'a>,
{
    // The alphabet only, from the head of the first file: a few thousand reads show all 4-5 bins, and
    // no spectrum is built from them, so the depth of the peek does not matter.
    let ladder = qual_profile::floors_from(
        &qual_profile::peek_alphabet(&opts.input_files[0].1[0], qual_profile::PEEK_READS),
        opts.quality.min_qual,
    );
    log::info!("Candidate base-quality floors: {ladder:?}");

    let mut out_path_histo = opts.histo_path("");
    let mut assembly = count_reads::<IntT>(
        &opts,
        k,
        opts.quality,
        Some(&ladder),
        None,
        timevec,
        &mut out_path_histo,
    )?;
    let chosen = assembly.chosen_min_qual;
    if chosen < opts.quality.min_qual {
        // Why it loosened is logged by the estimator, which is the only place that knows.
        log::warn!(
            "Recounting at a base-quality floor of {chosen} instead of {} — this admits more error \
             k-mers.",
            opts.quality.min_qual
        );
        let loosened = QualOpts {
            min_count: opts.quality.min_count,
            min_qual: chosen,
        };
        // Drop pass 1 before pass 2 allocates, or both tables are resident at once.
        drop(assembly);
        assembly = count_reads::<IntT>(
            &opts,
            k,
            &loosened,
            None,
            None,
            timevec,
            &mut out_path_histo,
        )?;
    }

    let mut contigs = graph_works::BasicAsm::assemble::<IntT>(
        k,
        &mut assembly.kmers,
        &mut Some(timevec),
        out_path_graph,
        // The spectrum of the pass that actually produced the k-mers: after a recount, pass 2's,
        // read at the loosened floor the graph's k-mers were counted at.
        opts.correction(k, assembly.genomic_peak, assembly.used_min_count),
    );

    save_functions::save_as_fasta_with_min_contig_length::<IntT>(
        &mut contigs,
        &assembly.kmers,
        k,
        opts.min_contig_length,
        opts.output,
    );
    Ok(())
}

/// What a multi-k run carries from one k to the next.
#[cfg(not(target_family = "wasm"))]
struct LadderState {
    /// The k values to run; a default ladder is settled once the first k has seen every read.
    ks: Vec<usize>,
    default_ladder: bool,
    /// Every read, packed during the first k's pass and replayed by the rest.
    store: read_store::ReadStore,
    /// The quality floor, settled at the first k and kept for the rest.
    quality: QualOpts,
    /// The previous k's contigs, spelled.
    contigs: Vec<Vec<u8>>,
}

/// Iterative multi-k, as SPAdes and GATB-Minia run it: each k is assembled from the reads plus the
/// previous k's contigs, and the last k's contigs are the output. `None` picks the ladder from the reads.
#[cfg(not(target_family = "wasm"))]
fn run_multik(
    opts: BuildOpts,
    requested: Option<&[usize]>,
    timevec: &mut Vec<Instant>,
    out_path_graph: &mut Option<PathBuf>,
) -> Result<(), preprocessing::PreprocessingError> {
    let floors = qual_profile::floors_from(
        &qual_profile::peek_alphabet(&opts.input_files[0].1[0], qual_profile::PEEK_READS),
        opts.quality.min_qual,
    );
    log::info!("Candidate base-quality floors: {floors:?}");
    let mut state = LadderState {
        ks: requested.map_or_else(|| vec![KMER_LADDER_SHORT[0]], <[usize]>::to_vec),
        default_ladder: requested.is_none(),
        store: read_store::ReadStore::new(&floors),
        quality: QualOpts {
            min_count: opts.quality.min_count,
            min_qual: opts.quality.min_qual,
        },
        contigs: Vec::new(),
    };
    let mut step = 0;
    while step < state.ks.len() {
        log::info!("Multi-k step {}: k={}", step + 1, state.ks[step]);
        let outcome = match state.ks[step] {
            3..=32 => run_ladder_step::<u64>(&opts, step, &mut state, timevec, out_path_graph),
            33..=64 => run_ladder_step::<u128>(&opts, step, &mut state, timevec, out_path_graph),
            65..=128 => run_ladder_step::<U256>(&opts, step, &mut state, timevec, out_path_graph),
            _ => run_ladder_step::<U512>(&opts, step, &mut state, timevec, out_path_graph),
        };
        if let Err(error) = outcome {
            // Only the empty-k-mer checks fail here, and a k no read reaches is out of reach for every
            // larger k too, so the ladder ends; the previous k's contigs are still in `state`.
            if step == 0 {
                return Err(error);
            }
            log::warn!(
                "Stopping the multi-k ladder at k={}: {error}. Writing the k={} contigs; no graph \
                 files are written for a stopped ladder.",
                state.ks[step],
                state.ks[step - 1]
            );
            break;
        }
        step += 1;
    }
    save_functions::save_sequences_as_fasta(&state.contigs, opts.min_contig_length, opts.output);
    Ok(())
}

/// One k of a multi-k run: count (from the files at the first k, else from the store), carry the
/// previous contigs in, assemble, and keep this k's contigs for the next.
#[cfg(not(target_family = "wasm"))]
fn run_ladder_step<IntT>(
    opts: &BuildOpts,
    step: usize,
    state: &mut LadderState,
    timevec: &mut Vec<Instant>,
    out_path_graph: &mut Option<PathBuf>,
) -> Result<(), preprocessing::PreprocessingError>
where
    IntT: for<'a> UInt<'a>,
{
    let k = state.ks[step];
    let mut out_path_histo = opts.histo_path(&format!("_k{k}"));
    let mut assembly = if step == 0 {
        let floors = state.store.floors().to_vec();
        let pass1 = count_reads::<IntT>(
            opts,
            k,
            &state.quality,
            Some(&floors),
            Some(&mut state.store),
            timevec,
            &mut out_path_histo,
        )?;
        state.store.finish();
        state.ks = settle_ladder(&state.ks, state.default_ladder, state.store.max_read_len());
        if pass1.chosen_min_qual < state.quality.min_qual {
            log::warn!(
                "Recounting at a base-quality floor of {} instead of {}, for every k — this admits \
                 more error k-mers.",
                pass1.chosen_min_qual,
                state.quality.min_qual
            );
            state.quality.min_qual = pass1.chosen_min_qual;
            // Drop pass 1 before pass 2 allocates, or both tables are resident at once.
            drop(pass1);
            count_store::<IntT>(
                opts,
                k,
                &state.quality,
                &state.store,
                &[],
                timevec,
                &mut out_path_histo,
            )?
        } else {
            pass1
        }
    } else {
        count_store::<IntT>(
            opts,
            k,
            &state.quality,
            &state.store,
            &state.contigs,
            timevec,
            &mut out_path_histo,
        )?
    };
    // Carried into the count-map by now, so the old contigs are dead weight.
    state.contigs = Vec::new();

    let last = step + 1 == state.ks.len();
    let mut graph_path = if last { out_path_graph.take() } else { None };
    let contigs = graph_works::BasicAsm::assemble::<IntT>(
        k,
        &mut assembly.kmers,
        &mut Some(timevec),
        &mut graph_path,
        opts.correction(k, assembly.genomic_peak, assembly.used_min_count),
    );
    state.contigs = save_functions::spell_contigs::<IntT>(&contigs, &assembly.kmers, k);
    log::info!(
        "k={k}: {} contigs, {} bases",
        state.contigs.len(),
        state.contigs.iter().map(Vec::len).sum::<usize>()
    );
    if !last && opts.keep_intermediate_contigs {
        let path =
            Path::new(opts.output_dir).join(format!("{}_k{k}_contigs.fasta", opts.output_prefix));
        save_functions::save_sequences_as_fasta(&state.contigs, opts.min_contig_length, path);
    }
    Ok(())
}

/// Count one k from the store at the settled floor, carrying `contigs` in once the reads set the cutoff.
#[cfg(not(target_family = "wasm"))]
fn count_store<IntT>(
    opts: &BuildOpts,
    k: usize,
    quality: &QualOpts,
    store: &read_store::ReadStore,
    contigs: &[Vec<u8>],
    timevec: &mut Vec<Instant>,
    out_path_histo: &mut Option<PathBuf>,
) -> Result<preprocessing::PreprocessedK<IntT>, preprocessing::PreprocessingError>
where
    IntT: for<'a> UInt<'a>,
{
    let carried = (!contigs.is_empty()).then_some(preprocessing::CarriedContigs {
        seqs: contigs,
        rule: opts.contig_counts,
    });
    preprocessing::preprocessing_standalone::<IntT, _>(
        &mut [store.records()],
        k,
        quality,
        // No ladder: the floor was settled at the first k.
        None,
        &mut Some(timevec),
        out_path_histo,
        // `--chunk-size` is a no-op here, and the first k already warned about it.
        0,
        opts.do_bloom,
        opts.do_fit,
        carried,
    )
}

/// The k values to run once every read is known: the SPAdes-style default for this length, or the
/// requested list less any k not below the longest read. The first k has run, so it always stays.
#[cfg(not(target_family = "wasm"))]
fn settle_ladder(ks: &[usize], default_ladder: bool, max_read_len: usize) -> Vec<usize> {
    let settled: Vec<usize> = if default_ladder {
        default_kmer_ladder(max_read_len)
    } else {
        ks.iter()
            .enumerate()
            .filter(|&(i, &k)| i == 0 || k < max_read_len)
            .map(|(_, &k)| k)
            .collect()
    };
    if settled.len() < ks.len() && !default_ladder {
        log::warn!("Dropping k values not below the longest read ({max_read_len} bp)");
    }
    log::info!("Multi-k ladder for reads up to {max_read_len} bp: k={settled:?}");
    settled
}

#[doc(hidden)]
#[cfg(not(target_family = "wasm"))]
pub fn main() {
    let args = cli_args();

    // log::info!("Starting program!");
    eprintln!("Sparrowhawk");
    let mut timevec = Vec::new();
    timevec.push(Instant::now());
    match &args.command {
        Commands::Build {
            seq_files,
            file_list,
            output_dir,
            output_prefix,
            k,
            min_count,
            min_qual,
            threads,
            no_bloom,
            chunk_size,
            bubble_pop_ratio,
            bubble_peak_ratio,
            no_ec_removal,
            ec_coverage_ratio,
            ec_require_both_flanks,
            tip_length,
            tip_length_kmult,
            tip_length_rctc_kmult,
            tip_rctc_cutoff,
            no_tip_rctc,
            min_contig_length,
            multik_contig_counts,
            keep_intermediate_contigs,
            no_histo,
            no_graphs,
            no_bubble_collapse,
            no_dead_end_removal,
        } => {
            let do_bloom = !*no_bloom;
            if let Err(message) = cli::validate_bloom_min_count(do_bloom, *min_count) {
                eprintln!("error: {message}");
                std::process::exit(2);
            }

            // Create the output directory if it does not exist, so every write below can assume it is
            // there.
            std::fs::create_dir_all(output_dir)
                .unwrap_or_else(|e| panic!("cannot create output directory {output_dir:?}: {e}"));

            let outputlogfile: PathBuf =
                Path::new(output_dir).join(format!("{output_prefix}_log.txt"));
            if args.verbose {
                // set_up_logging(log::LevelFilter::Trace, outputlogfile);
                set_up_logging(log::LevelFilter::Info, outputlogfile);
            } else {
                set_up_logging(log::LevelFilter::Warn, outputlogfile);
            }

            check_threads(*threads);

            // Read input
            let input_files = get_input_list(file_list, seq_files);
            // let input_files = get_input_list(file_list);

            // Fit the min_count from the spectrum unless an explicit value was given.
            let do_fit = min_count.is_none();
            let quality = QualOpts {
                // Only used when do_fit is false; the fit ignores it.
                min_count: min_count.unwrap_or(DEFAULT_MINCOUNT),
                min_qual: min_qual.unwrap_or(DEFAULT_MINQUAL),
            };
            log::info!("Minimum base quality used: {}", quality.min_qual);

            // Build, merge
            // let rc = !*single_strand;

            log::info!("Checking requested threads and creating pool if needed");
            rayon::ThreadPoolBuilder::new()
                .num_threads(*threads)
                .build_global()
                .unwrap();
            log::info!("Beginning processing");
            timevec.push(Instant::now());

            if do_fit {
                log::info!("Minimum count per k-mer will be fitted from the spectrum, per k.");
            } else {
                log::info!(
                    "Minimum count per k-mer to be considered is {}",
                    quality.min_count
                );
            }

            // No extension here: `assemble` sets .dot/.gfa/.gfa2 on this base path later (hence `mut`).
            let mut out_path_graph: Option<PathBuf> = if *no_graphs {
                None
            } else {
                Some(Path::new(output_dir).join(format!("{output_prefix}_graph")))
            };

            let output: PathBuf =
                Path::new(output_dir).join(format!("{output_prefix}_contigs.fasta"));

            if !(*bubble_pop_ratio > 0.0 && *bubble_pop_ratio < 1.0) {
                eprintln!(
                    "error: --bubble-pop-ratio must be strictly between 0 and 1 (got \
                     {bubble_pop_ratio}). It is the fraction of the stronger branch's coverage below \
                     which the weaker branch counts as an error; 1.0 or more pops every bubble, 0.0 or \
                     less pops none."
                );
                std::process::exit(2);
            }
            if let Some(ratio) = bubble_peak_ratio {
                if !(*ratio > 0.0 && *ratio < 1.0) {
                    eprintln!(
                        "error: --bubble-peak-ratio must be strictly between 0 and 1 (got {ratio}). \
                         It is the fraction of single-copy coverage below which a branch counts as an \
                         error."
                    );
                    std::process::exit(2);
                }
            }
            if !(ec_coverage_ratio.is_finite() && *ec_coverage_ratio > 1.0) {
                eprintln!(
                    "error: --ec-coverage-ratio must be finite and greater than 1 (got \
                     {ec_coverage_ratio}). It is how many times a connector's flanks must out-cover \
                     it before the connector is treated as erroneous."
                );
                std::process::exit(2);
            }
            if *tip_length_kmult < 0.0 {
                eprintln!(
                    "error: --tip-length-kmult must be >= 0 (got {tip_length_kmult}). Zero keeps the \
                     flat --tip-length."
                );
                std::process::exit(2);
            }
            if *tip_length_rctc_kmult < 0.0 {
                eprintln!(
                    "error: --tip-length-rctc-kmult must be >= 0 (got {tip_length_rctc_kmult}). \
                     Zero drops the coverage tier, leaving only the length rule."
                );
                std::process::exit(2);
            }
            if !tip_rctc_cutoff.is_finite() || *tip_rctc_cutoff <= 0.0 {
                eprintln!(
                    "error: --tip-rctc-cutoff must be finite and > 0 (got {tip_rctc_cutoff}). It is \
                     how many times better covered a junction must be than the tip hanging off it."
                );
                std::process::exit(2);
            }
            // Each k is seeded with the contigs of the k before it, so the ladder must climb.
            if let Some(ks) = k {
                if ks.windows(2).any(|w| w[1] <= w[0]) {
                    eprintln!("error: -k values must be strictly ascending (got {ks:?}).");
                    std::process::exit(2);
                }
            }
            let opts = BuildOpts {
                input_files: &input_files,
                quality: &quality,
                chunk_size: *chunk_size,
                do_bloom,
                do_fit,
                do_bubble_collapse: !no_bubble_collapse,
                do_dead_end_removal: !no_dead_end_removal,
                pop_ratio: *bubble_pop_ratio,
                peak_ratio: *bubble_peak_ratio,
                do_ec_removal: !no_ec_removal,
                ec_ratio: *ec_coverage_ratio,
                ec_require_both_flanks: *ec_require_both_flanks,
                tip_length: *tip_length,
                tip_length_kmult: *tip_length_kmult,
                tip_length_rctc_kmult: (!no_tip_rctc).then_some(*tip_length_rctc_kmult),
                tip_rctc_cutoff: *tip_rctc_cutoff,
                min_contig_length: *min_contig_length,
                contig_counts: *multik_contig_counts,
                keep_intermediate_contigs: *keep_intermediate_contigs,
                no_histo: *no_histo,
                output_dir,
                output_prefix,
                output,
            };

            let build_result = match k.as_deref() {
                Some(&[single]) => {
                    // The packed k-mer must fit in 2*k bits, so k picks the integer width.
                    if single % 2 == 0 {
                        panic!("Support for even k-mer lengths not implemented");
                    }
                    let width_k = single;
                    match width_k {
                        0..=2 => panic!("kmer length too small (min. 3)"),
                        3..=32 => {
                            log::info!("k={width_k}: using 64-bit representation");
                            run_build::<u64>(opts, single, &mut timevec, &mut out_path_graph)
                        }
                        33..=64 => {
                            log::info!("k={width_k}: using 128-bit representation");
                            run_build::<u128>(opts, single, &mut timevec, &mut out_path_graph)
                        }
                        65..=128 => {
                            log::info!("k={width_k}: using 256-bit representation");
                            run_build::<U256>(opts, single, &mut timevec, &mut out_path_graph)
                        }
                        129..=256 => {
                            log::info!("k={width_k}: using 512-bit representation");
                            run_build::<U512>(opts, single, &mut timevec, &mut out_path_graph)
                        }
                        _ => panic!("kmer length larger than 256 currently not supported."),
                    }
                }
                requested => run_multik(opts, requested, &mut timevec, &mut out_path_graph),
            };
            if let Err(error) = build_result {
                eprintln!("error: {error}");
                std::process::exit(1);
            }
        }
    }

    timevec.push(Instant::now());

    log::info!(
        "Sparrowhawk done in {} s",
        timevec
            .last()
            .unwrap()
            .duration_since(*timevec.first().unwrap())
            .as_secs()
    );
    log::info!("Finishing program!");
}

// ===================================== WebAssembly stuff follows
#[cfg(target_family = "wasm")]
/// Binary dummy function. In the future, we need to completely remove it whenever compilating with the feature "wasm"
pub fn main() {
    panic!("You've compiled Sparrowhawk for WebAssembly support, you cannot use it as a normal binary anymore!");
}

#[cfg(target_family = "wasm")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

    #[wasm_bindgen(js_name = postMessage)]
    fn post_message(data: &JsValue);
}

#[cfg(target_family = "wasm")]
/// Posts a state update message to the main thread via postMessage
pub fn post_state(state: &str) {
    let obj = js_sys::Object::new();
    let _ = js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("assemblyState"),
        &JsValue::from_str(state),
    );
    post_message(&obj.into());
}

#[cfg(target_family = "wasm")]
#[wasm_bindgen]
/// Function that allows to propagate panic error messages when compiling to wasm, see https://github.com/rustwasm/console_error_panic_hook
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[cfg(target_family = "wasm")]
#[wasm_bindgen]
/// Main struct that acts as wrapper of the assembler when compiling to wasm
pub struct AssemblyHelper {
    verbose: bool,
    k: usize,
    min_count: u16,
    min_qual: u8,
    chunk_size: usize,
    do_bloom: bool,
    do_fit: bool,
    no_bubble_collapse: bool,
    no_dead_end_removal: bool,
    /// The CLI's `--bubble-pop-ratio`. Set through `set_bubble_pop_ratio`, not the constructor, so that
    /// existing JS callers keep working and get the same default the CLI does.
    bubble_pop_ratio: f32,
    /// The CLI's `--tip-length-kmult`. Set through `set_tip_length_kmult`, for the same reason.
    tip_length_kmult: f32,
    preprocessed_data: Option<HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>>,
    maxmindict: Option<HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>>,
    seqdict64: Option<HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>>,
    seqdict128: Option<HashMap<u64, u128, BuildHasherDefault<NoHashHasher<u64>>>>,
    seqdict256: Option<HashMap<u64, U256, BuildHasherDefault<NoHashHasher<u64>>>>,
    seqdict512: Option<HashMap<u64, U512, BuildHasherDefault<NoHashHasher<u64>>>>,
    histovec: Option<Vec<u32>>,
    used_min_count: u16,
    contigs: Contigs,
    outfasta: String,
    outdot: String,
    outgfa: String,
    outgfav2: String,
}

#[cfg(target_family = "wasm")]
#[wasm_bindgen]
impl AssemblyHelper {
    /// Constructor/initialiser of the wasm assembler. It also performs the preprocessing.
    pub fn new(
        k: u32,
        verbose: bool,
        min_count: u16,
        min_qual: u8,
        chunk_size: u32,
        do_bloom: bool,
        do_fit: bool,
        no_bubble_collapse: bool,
        no_dead_end_removal: bool,
    ) -> Self {
        if let Err(message) =
            cli::validate_bloom_min_count(do_bloom, (!do_fit).then_some(min_count))
        {
            panic!("{message}");
        }

        let k = k as usize;
        let chunk_size = chunk_size as usize;

        if cfg!(debug_assertions) {
            init_panic_hook();
        }

        // TODO: improve verbose with the creation of a logging class and object that carries the verbose level and affects the logw functions, or something similar.

        logw("Beginning processing", Some("info"));
        post_state("initialised");

        Self {
            verbose,
            k,
            min_count,
            min_qual,
            chunk_size,
            do_bloom,
            do_fit,
            no_bubble_collapse,
            no_dead_end_removal,
            bubble_pop_ratio: algorithms::corrector::DEFAULT_POP_RATIO,
            tip_length_kmult: cli::DEFAULT_TIP_LEN_KMULT,
            preprocessed_data: None,
            maxmindict: None,
            seqdict64: None,
            seqdict128: None,
            seqdict256: None,
            seqdict512: None,
            histovec: None,
            used_min_count: 0,
            contigs: Contigs::default(),
            outfasta: "".to_owned(),
            outdot: "".to_owned(),
            outgfa: "".to_owned(),
            outgfav2: "".to_owned(),
        }
    }

    /// Preprocess read files
    pub fn preprocess(&mut self, file1: web_sys::File, file2: Option<web_sys::File>) {
        post_state("preprocess:start");

        let mut wf1 = WebSysFile::new(file1);
        let mut wf2 = file2.map(|f| WebSysFile::new(f));

        // Read input
        let quality = QualOpts {
            min_count: self.min_count,
            min_qual: self.min_qual,
        };

        logw("Beginning processing", Some("info"));

        let preprocessed_data: HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>;
        let maxmindict: HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>;
        let histovalues: Vec<u32>;
        let used_min_count: u16;
        let mut thedict64 = None;
        let mut thedict128 = None;
        let mut thedict256 = None;
        let mut thedict512 = None;

        if self.k.is_multiple_of(2) {
            panic!("Support for even k-mer lengths not implemented");
        } else if self.k < 3 {
            panic!("kmer length too small (min. 3)");
        } else if self.k <= 32 {
            logw(
                format!("k={}: using 64-bit representation", self.k).as_str(),
                Some("info"),
            );

            (
                preprocessed_data,
                thedict64,
                maxmindict,
                histovalues,
                used_min_count,
            ) = preprocessing::preprocessing_wasm::<u64>(
                &mut wf1,
                wf2.as_mut(),
                self.k,
                &quality,
                self.chunk_size,
                self.do_bloom,
                self.do_fit,
            );

            logw("Preprocessing done!", Some("info"));
        } else if self.k <= 64 {
            logw(
                format!("k={}: using 128-bit representation", self.k).as_str(),
                Some("info"),
            );

            (
                preprocessed_data,
                thedict128,
                maxmindict,
                histovalues,
                used_min_count,
            ) = preprocessing::preprocessing_wasm::<u128>(
                &mut wf1,
                wf2.as_mut(),
                self.k,
                &quality,
                self.chunk_size,
                self.do_bloom,
                self.do_fit,
            );

            logw("Preprocessing done!", Some("info"));
        } else if self.k <= 128 {
            logw(
                format!("k={}: using 256-bit representation", self.k).as_str(),
                Some("info"),
            );

            (
                preprocessed_data,
                thedict256,
                maxmindict,
                histovalues,
                used_min_count,
            ) = preprocessing::preprocessing_wasm::<U256>(
                &mut wf1,
                wf2.as_mut(),
                self.k,
                &quality,
                self.chunk_size,
                self.do_bloom,
                self.do_fit,
            );

            logw("Preprocessing done!", Some("info"));
        } else if self.k <= 256 {
            logw(
                format!("k={}: using 512-bit representation", self.k).as_str(),
                Some("info"),
            );

            (
                preprocessed_data,
                thedict512,
                maxmindict,
                histovalues,
                used_min_count,
            ) = preprocessing::preprocessing_wasm::<U512>(
                &mut wf1,
                wf2.as_mut(),
                self.k,
                &quality,
                self.chunk_size,
                self.do_bloom,
                self.do_fit,
            );

            logw("Preprocessing done!", Some("info"));
        } else {
            panic!("kmer length larger than 256 currently not supported.");
        }

        post_state("preprocess:saving");

        self.preprocessed_data = Some(preprocessed_data);
        self.maxmindict = Some(maxmindict);
        self.seqdict64 = thedict64;
        self.seqdict128 = thedict128;
        self.seqdict256 = thedict256;
        self.seqdict512 = thedict512;
        self.histovec = Some(histovalues);
        self.used_min_count = used_min_count;

        post_state("preprocess:end");
    }

    /// Fraction of counts needed for popping bubble
    pub fn set_bubble_pop_ratio(&mut self, ratio: f32) {
        if ratio > 0.0 && ratio < 1.0 {
            self.bubble_pop_ratio = ratio;
        } else {
            logw(
                format!(
                    "Ignoring --bubble-pop-ratio {ratio}: it must be strictly between 0 and 1. \
                     Keeping {}.",
                    self.bubble_pop_ratio
                )
                .as_str(),
                Some("warn"),
            );
        }
    }

    /// Multiplier scaling the tip-removal threshold with k. Zero keeps the flat floor.
    pub fn set_tip_length_kmult(&mut self, kmult: f32) {
        if kmult >= 0.0 {
            self.tip_length_kmult = kmult;
        } else {
            logw(
                format!(
                    "Ignoring --tip-length-kmult {kmult}: it must be >= 0. Keeping {}.",
                    self.tip_length_kmult
                )
                .as_str(),
                Some("warn"),
            );
        }
    }

    /// Assemble method of the wasm version
    pub fn assemble(&mut self) {
        logw("Starting assembly...", Some("info"));
        let (mut outcontigs, outdot, outgfa, outgfav2) = graph_works::BasicAsm::assemble_wasm(
            self.k,
            self.preprocessed_data.as_mut().unwrap(),
            self.maxmindict.as_mut().unwrap(),
            algorithms::corrector::CorrectionOpts {
                do_bubble_collapse: !self.no_bubble_collapse,
                do_dead_end_removal: !self.no_dead_end_removal,
                pop_ratio: self.bubble_pop_ratio,
                tip_nts: algorithms::corrector::tip_length_nts(
                    cli::DEFAULT_TIP_LEN_NTS,
                    self.tip_length_kmult,
                    self.k,
                ),
                // `remove_dead_paths` is compiled for wasm, so the coverage tier is live here too.
                tip_rctc_nts: algorithms::corrector::tip_length_nts(
                    cli::DEFAULT_TIP_LEN_NTS,
                    cli::DEFAULT_TIP_RCTC_KMULT,
                    self.k,
                ),
                tip_rctc_cutoff: cli::DEFAULT_TIP_RCTC_CUTOFF,
                // `preprocessing_wasm` does not carry a peak out, so the browser states that it knows
                // no coverage and the bubble rule there stays exactly what it is today.
                coverage: algorithms::corrector::CoverageRef::unknown(),
                // wasm does not compile `path_correction`; the loop there never reaches EC removal.
                do_ec_removal: false,
                ec_ratio: algorithms::corrector::DEFAULT_EC_COVERAGE_RATIO,
                ec_require_both_flanks: true,
            },
        );

        post_state("assembly:saving");
        logw("Assembly done!", Some("info"));

        let outfasta: String;

        if self.k.is_multiple_of(2) {
            panic!("Support for even k-mer lengths not implemented");
        } else if self.k < 3 {
            panic!("kmer length too small (min. 3)");
        } else if self.k <= 32 {
            outfasta = save_functions::save_as_fasta_wasm::<u64>(
                &mut outcontigs,
                self.seqdict64.as_ref().unwrap(),
                self.k,
            );
        } else if self.k <= 64 {
            outfasta = save_functions::save_as_fasta_wasm::<u128>(
                &mut outcontigs,
                self.seqdict128.as_ref().unwrap(),
                self.k,
            );
        } else if self.k <= 128 {
            outfasta = save_functions::save_as_fasta_wasm::<U256>(
                &mut outcontigs,
                self.seqdict256.as_ref().unwrap(),
                self.k,
            );
        } else if self.k <= 256 {
            outfasta = save_functions::save_as_fasta_wasm::<U512>(
                &mut outcontigs,
                self.seqdict512.as_ref().unwrap(),
                self.k,
            );
        } else {
            panic!("kmer length larger than 256 currently not supported.");
        }

        logw("Sparrowhawk done!", Some("info"));

        self.contigs = outcontigs;
        self.outfasta = outfasta;
        self.outdot = outdot;
        self.outgfa = outgfa;
        self.outgfav2 = outgfav2;
        post_state("assembly:end");
    }

    /// Getter to obtain the results as JSON of the assembly
    pub fn get_assembly(&self) -> String {
        let mut results = json::JsonValue::new_array();

        results["outfasta"] = json::JsonValue::String(self.outfasta.clone());
        results["outdot"] = json::JsonValue::String(self.outdot.clone());
        results["outgfa"] = json::JsonValue::String(self.outgfa.clone());
        results["outgfav2"] = json::JsonValue::String(self.outgfav2.clone());
        results["ncontigs"] =
            json::JsonValue::Number(self.contigs.contig_sequences.as_ref().unwrap().len().into());

        results.dump()
    }

    /// Getter to obtain the results as JSON of the preprocessing
    pub fn get_preprocessing_info(&self) -> String {
        let mut results = json::JsonValue::new_array();

        logw(
            format!(
                "{} {}",
                self.preprocessed_data.as_ref().unwrap().len(),
                self.histovec.as_ref().unwrap().len()
            )
            .as_str(),
            Some("info"),
        );

        results["nkmers"] =
            json::JsonValue::Number(self.preprocessed_data.as_ref().unwrap().len().into());
        // Only the historical range: the histogram is now 16x wider, and the front-end plots whatever
        // it is handed, so sending all of it would stretch the web spectrum over mostly empty bins.
        results["histo"] = json::JsonValue::Array(
            self.histovec
                .as_ref()
                .unwrap()
                .iter()
                .take(crate::preprocessing::LEGACY_HISTO_RANGE)
                .map(|x| json::JsonValue::Number((*x).into()))
                .collect(),
        );
        results["used_min_count"] = json::JsonValue::Number(self.used_min_count.into());

        logw(results.dump().as_str(), Some("debug"));

        results.dump()
    }
}
