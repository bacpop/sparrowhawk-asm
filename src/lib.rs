//! Efficient genome assembler for small genomes in Rust
#![warn(missing_docs)]
use std::fmt;

// Only the wasm `AssemblyHelper` still names these types directly; on native the k-mer maps now live
// behind `preprocessing::PreprocessedK`.
#[cfg(target_family = "wasm")]
use std::{collections::HashMap, hash::BuildHasherDefault};

#[cfg(not(target_family = "wasm"))]
use std::{path::PathBuf, time::Instant, time::SystemTime};

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

/// Turning a walk of canonical k-mer hashes back into nucleotides
pub mod spelling;

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
/// is made once instead of four times — which is what makes adding a second k a local edit.
#[cfg(not(target_family = "wasm"))]
struct BuildOpts<'a> {
    input_files: &'a [InputFastx],
    /// `[k1]` for a single-k assembly, or `[k1, k2]` with `k2 > k1` for multi-k. `k1` is the assembly
    /// k, whose graph is output; `k2` is the evidence k, used only to correct it.
    ks: &'a [usize],
    quality: &'a QualOpts,
    chunk_size: usize,
    counter: Counter,
    do_bloom: bool,
    auto_min_count: bool,
    do_bubble_collapse: bool,
    do_dead_end_removal: bool,
    extraction: MultiKExtraction,
    /// Multi-k oracle: k-mers of each flanking unitig to use as context around a bubble.
    flank_context: usize,
    /// Multi-k oracle: minimum discriminating evidence-k k-mers before a branch can be judged.
    min_evidence: usize,
    /// Multi-k oracle: maximum evidence nodes a supported branch's discriminating window may span.
    max_nodes: usize,
    /// Multi-k: veto the coverage heuristic on bubbles whose branches are both real.
    protect: bool,
    /// Multi-k: duplicate collapsed repeats the evidence k can resolve.
    resolve: bool,
    /// Multi-k: maximum evidence-driven correction rounds.
    max_rounds: usize,
    /// Multi-k: where to write the machine-readable verdict table.
    stats_path: Option<PathBuf>,
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
    let k1 = opts.ks[0];

    // Build each evidence graph and drop that k's preprocessing with it, one k at a time.
    //
    // This matters much more with a ladder than it did with two k. The evidence side never spells
    // sequence, so its `thedict` is dead weight, and once the graph is built `themap`'s neighbour lists
    // live inside it — so an evidence k's count structures (the big ones) need never coexist with
    // another's. Peak becomes `one k's preprocessing + the evidence graphs built so far`, rather than
    // the sum over every k.
    //
    // `joint` extraction cannot do this: by construction it counts every k in one pass, so all N sets of
    // count structures are live at once. It was already measured to buy nothing (+-3%), so with a ladder
    // it is simply a worse deal, and we say so.
    let mut evidence: Vec<algorithms::multik::EvidenceGraph> = Vec::new();
    let mut assembly;

    if opts.ks.len() > 1 && opts.extraction == MultiKExtraction::Joint {
        log::warn!(
            "--multik-extraction joint counts all {} k in one pass, so every k's count structures are \
             live at once. It saves no time (measured: within 3% of sequential, and slower on the \
             default counter), so with a ladder it only costs memory. Consider `sequential`.",
            opts.ks.len()
        );
        let mut pre = preprocessing::preprocessing_standalone_multik::<IntT>(
            opts.input_files,
            opts.ks,
            opts.quality,
            timevec,
            out_paths_histo,
            opts.chunk_size,
            opts.counter,
            opts.do_bloom,
            opts.auto_min_count,
            opts.extraction,
        );
        // Drain the evidence k from the back, so the assembly k is what is left.
        let mut evs: Vec<_> = pre.split_off(1);
        for ev_pre in evs.drain(..) {
            evidence.push(algorithms::multik::build_evidence::<IntT>(ev_pre));
        }
        assembly = pre.pop().unwrap();
    } else {
        for (i, &k_ev) in opts.ks.iter().enumerate().skip(1) {
            let ev_pre = preprocessing::preprocessing_standalone::<IntT>(
                opts.input_files,
                k_ev,
                opts.quality,
                timevec,
                &mut out_paths_histo[i],
                opts.chunk_size,
                opts.counter,
                opts.do_bloom,
                opts.auto_min_count,
            );
            evidence.push(algorithms::multik::build_evidence::<IntT>(ev_pre));
        }
        assembly = preprocessing::preprocessing_standalone::<IntT>(
            opts.input_files,
            k1,
            opts.quality,
            timevec,
            &mut out_paths_histo[0],
            opts.chunk_size,
            opts.counter,
            opts.do_bloom,
            opts.auto_min_count,
        );
    }

    let multik = if evidence.is_empty() {
        None
    } else {
        Some(graph_works::MultiKCtx {
            evidence: &evidence,
            dict: &assembly.thedict,
            flank_budget: opts.flank_context,
            min_evidence: opts.min_evidence,
            max_nodes: opts.max_nodes,
            protect: opts.protect,
            resolve: opts.resolve,
            max_rounds: opts.max_rounds,
            stats_path: opts.stats_path.clone(),
        })
    };

    let mut contigs = graph_works::BasicAsm::assemble::<IntT>(
        k1,
        &mut assembly.themap,
        &mut assembly.maxmindict,
        timevec,
        out_path_graph,
        opts.do_bubble_collapse,
        opts.do_dead_end_removal,
        false,
        multik,
    );

    save_functions::save_as_fasta::<IntT>(&mut contigs, &assembly.thedict, k1, opts.output);
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
            auto_min_count,
            do_bloom,
            chunk_size,
            counter,
            multik_extraction,
            multik_min_evidence,
            multik_max_nodes,
            multik_flank_context,
            multik_max_rounds,
            multik_protect,
            no_multik_resolve,
            no_histo,
            no_graphs,
            no_bubble_collapse,
            no_dead_end_removal,
            // no_conflictive_links_removal,
        } => {
            let mut outputlogfile: PathBuf = output_dir.into();
            outputlogfile.set_file_name(output_prefix.to_string() + "_log");
            outputlogfile.set_extension("txt");
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
            let quality = QualOpts {
                min_count: *min_count,
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

            if *auto_min_count {
                log::info!("Automatic fitting to extract minimum counts per k-mer will be done.");
            } else {
                log::info!("Minimum count per k-mer to be considered is {}", min_count);
            }

            // One spectrum PNG per k, so the two fits can be compared. The evidence k has materially
            // lower k-mer coverage (fewer k-mers per read, and (1-e)^k error-free probability), so its
            // min_count must be fitted on its own spectrum, never inherited from the assembly k.
            let mut out_paths_histo: Vec<Option<PathBuf>> = k
                .iter()
                .map(|ki| {
                    if *no_histo {
                        return None;
                    }
                    let mut p: PathBuf = output_dir.into();
                    let suffix = if k.len() > 1 {
                        format!("_kmerspectrum_k{ki}")
                    } else {
                        "_kmerspectrum".to_string()
                    };
                    p.set_file_name(output_prefix.to_owned() + &suffix);
                    p.set_extension("png");
                    Some(p)
                })
                .collect();

            let mut out_path_graph: Option<PathBuf>;
            if *no_graphs {
                out_path_graph = None;
            } else {
                out_path_graph = Some(output_dir.into());
                out_path_graph
                    .as_mut()
                    .unwrap()
                    .set_file_name(output_prefix.to_owned() + "_graph");
            }

            let mut output: PathBuf = output_dir.into();
            output.set_file_name(output_prefix.to_string() + "_contigs");
            output.set_extension("fasta");

            // `valid_kmer` already rejects even k and anything outside 3..=256 per value; what it
            // cannot see is the relationship between them.
            // Strictly ascending: the first k is the assembly k, the rest are evidence, and the
            // ladder climbs. Equal or descending values are always a mistake, and silently sorting them
            // would hide it.
            if k.windows(2).any(|w| w[1] <= w[0]) {
                eprintln!(
                    "error: -k values must be strictly ascending (got {k:?}). The first is the assembly \
                     k; every later one is a larger evidence k, whose whole purpose is longer-range read \
                     evidence than the k before it."
                );
                std::process::exit(2);
            }
            if k.len() > 1 {
                log::info!(
                    "Multi-k: assembling at k={}, with evidence ladder k={:?}",
                    k[0],
                    &k[1..]
                );
            }

            let opts = BuildOpts {
                input_files: &input_files,
                ks: k,
                quality: &quality,
                chunk_size: *chunk_size,
                counter: *counter,
                do_bloom: *do_bloom,
                auto_min_count: *auto_min_count,
                do_bubble_collapse: !no_bubble_collapse,
                do_dead_end_removal: !no_dead_end_removal,
                extraction: *multik_extraction,
                flank_context: *multik_flank_context,
                min_evidence: *multik_min_evidence,
                max_nodes: *multik_max_nodes,
                protect: *multik_protect,
                resolve: !no_multik_resolve,
                max_rounds: *multik_max_rounds,
                stats_path: if k.len() > 1 {
                    let mut p: PathBuf = output_dir.into();
                    p.set_file_name(output_prefix.to_owned() + "_multik_stats");
                    p.set_extension("tsv");
                    Some(p)
                } else {
                    None
                },
                output,
            };

            // The packed k-mer must fit in 2*k bits, so k picks the integer width. Both graphs share
            // one width, taken from the LARGER k: it costs a little memory on the assembly k's
            // `thedict` (post-filter, so roughly genome-sized) and saves monomorphising the whole
            // pipeline twice.
            let max_k = *k.iter().max().unwrap();
            if max_k % 2 == 0 {
                panic!("Support for even k-mer lengths not implemented");
            }
            match max_k {
                0..=2 => panic!("kmer length too small (min. 3)"),
                3..=32 => {
                    log::info!("max k={max_k}: using 64-bit representation");
                    run_build::<u64>(opts, &mut timevec, &mut out_paths_histo, &mut out_path_graph)
                }
                33..=64 => {
                    log::info!("max k={max_k}: using 128-bit representation");
                    run_build::<u128>(opts, &mut timevec, &mut out_paths_histo, &mut out_path_graph)
                }
                65..=128 => {
                    log::info!("max k={max_k}: using 256-bit representation");
                    run_build::<U256>(opts, &mut timevec, &mut out_paths_histo, &mut out_path_graph)
                }
                129..=256 => {
                    log::info!("max k={max_k}: using 512-bit representation");
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

    /// Assemble method of the wasm version
    pub fn assemble(&mut self) {
        logw("Starting assembly...", Some("info"));
        let (mut outcontigs, outdot, outgfa, outgfav2) = graph_works::BasicAsm::assemble_wasm(
            self.k,
            self.preprocessed_data.as_mut().unwrap(),
            self.maxmindict.as_mut().unwrap(),
            !self.no_bubble_collapse,
            !self.no_dead_end_removal,
            false,
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
