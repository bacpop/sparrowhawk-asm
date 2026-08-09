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

/// Preprocessing functions of the reads & k-mers
pub mod preprocessing;

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

/// Fits the k-mer spectrum to automatically get a min_count (taken from ska.rust!)
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
            (self.min_qual + 33) as char,
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
    k: usize,
    quality: &'a QualOpts,
    chunk_size: usize,
    do_bloom: bool,
    /// Fit `min_count` from the k-mer spectrum rather than using `quality.min_count`.
    do_fit: bool,
    do_bubble_collapse: bool,
    do_dead_end_removal: bool,
    /// Fraction of the stronger branch's coverage below which the weaker branch of a bubble is popped.
    pop_ratio: f32,
    output: PathBuf,
}

/// Run the whole `build` pipeline, monomorphised on the packed-k-mer width.
#[cfg(not(target_family = "wasm"))]
fn run_build<IntT>(
    opts: BuildOpts,
    timevec: &mut Vec<Instant>,
    out_paths_histo: &mut [Option<PathBuf>],
    out_path_graph: &mut Option<PathBuf>,
) where
    IntT: for<'a> UInt<'a>,
{
    let mut assembly = preprocessing::preprocessing_standalone::<IntT>(
        opts.input_files,
        opts.k,
        opts.quality,
        timevec,
        &mut out_paths_histo[0],
        opts.chunk_size,
        opts.do_bloom,
        opts.do_fit,
    );

    let mut contigs = graph_works::BasicAsm::assemble::<IntT>(
        opts.k,
        &mut assembly.themap,
        &mut assembly.maxmindict,
        &assembly.thedict,
        timevec,
        out_path_graph,
        opts.do_bubble_collapse,
        opts.do_dead_end_removal,
        opts.pop_ratio,
    );

    save_functions::save_as_fasta::<IntT>(&mut contigs, &assembly.thedict, opts.k, opts.output);
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
            do_bloom,
            chunk_size,
            bubble_pop_ratio,
            no_histo,
            no_graphs,
            no_bubble_collapse,
            no_dead_end_removal,
        } => {
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
                min_qual: *min_qual,
            };

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

            let mut out_paths_histo: Vec<Option<PathBuf>> = vec![if *no_histo {
                None
            } else {
                Some(Path::new(output_dir).join(format!("{output_prefix}_kmerspectrum.png")))
            }];

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
            let opts = BuildOpts {
                input_files: &input_files,
                k: *k,
                quality: &quality,
                chunk_size: *chunk_size,
                do_bloom: *do_bloom,
                do_fit,
                do_bubble_collapse: !no_bubble_collapse,
                do_dead_end_removal: !no_dead_end_removal,
                pop_ratio: *bubble_pop_ratio,
                output,
            };

            // The packed k-mer must fit in 2*k bits, so k picks the integer width.
            if k % 2 == 0 {
                panic!("Support for even k-mer lengths not implemented");
            }
            let width_k = *k;
            match width_k {
                0..=2 => panic!("kmer length too small (min. 3)"),
                3..=32 => {
                    log::info!("k={width_k}: using 64-bit representation");
                    run_build::<u64>(opts, &mut timevec, &mut out_paths_histo, &mut out_path_graph)
                }
                33..=64 => {
                    log::info!("k={width_k}: using 128-bit representation");
                    run_build::<u128>(opts, &mut timevec, &mut out_paths_histo, &mut out_path_graph)
                }
                65..=128 => {
                    log::info!("k={width_k}: using 256-bit representation");
                    run_build::<U256>(opts, &mut timevec, &mut out_paths_histo, &mut out_path_graph)
                }
                129..=256 => {
                    log::info!("k={width_k}: using 512-bit representation");
                    run_build::<U512>(opts, &mut timevec, &mut out_paths_histo, &mut out_path_graph)
                }
                _ => panic!("kmer length larger than 256 currently not supported."),
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

    /// Assemble method of the wasm version
    pub fn assemble(&mut self) {
        logw("Starting assembly...", Some("info"));
        let (mut outcontigs, outdot, outgfa, outgfav2) = graph_works::BasicAsm::assemble_wasm(
            self.k,
            self.preprocessed_data.as_mut().unwrap(),
            self.maxmindict.as_mut().unwrap(),
            !self.no_bubble_collapse,
            !self.no_dead_end_removal,
            self.bubble_pop_ratio,
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
        results["histo"] = json::JsonValue::Array(
            self.histovec
                .as_ref()
                .unwrap()
                .iter()
                .map(|x| json::JsonValue::Number((*x).into()))
                .collect(),
        );
        results["used_min_count"] = json::JsonValue::Number(self.used_min_count.into());

        logw(results.dump().as_str(), Some("debug"));

        results.dump()
    }
}
