//! Efficient genome assembler for bacterial species in Rust, browser-ready.
//!
//! Use [`assemble`] with [`AssemblyOptions`] for the complete single-k/multi-k pipeline,
//! or call [`preprocessing::preprocessing`], [`graph_works::BasicAsm`] through
//! [`graph_works::Assemble`], and [`graph_works::spelling::spell_contigs`] separately.
//! Results remain packed; FASTA writing decodes bounded buffers and returns I/O errors.
//! Library entry points do not initialise logging or a global Rayon pool.
//!
//! ```no_run
//! use sparrowhawk_asm::{assemble, AssemblyOptions, InputFiles};
//! use std::io::{self, Write};
//!
//! # #[cfg(not(target_family = "wasm"))]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let reads = InputFiles::new(["reads_1.fastq.gz", "reads_2.fastq.gz"])?;
//! let result = assemble(&reads, &AssemblyOptions::default())?;
//! let mut output = io::BufWriter::new(io::stdout().lock());
//! result.write_fasta(&mut output)?;
//! output.flush()?;
//! # Ok(())
//! # }
//! ```
//!
//! Sources must be repeatable; see [`IteratorSource`] for custom parsers. Qualities are
//! PHRED+33 bytes, while quality thresholds are numerical PHRED scores. Component
//! preprocessing keeps a fixed quality floor; the complete API chooses floors and recounts.
//! Accepted k values and biological early stops are available on [`Assembly`]. Input,
//! spelling, and graph callback failures propagate through [`AssemblyError`].
//!
//! See `examples/api_usage.rs` for automatic assembly and
//! `examples/api_usage_step_by_step.rs` for the manual component workflow.
#![warn(missing_docs)]

mod api;
mod cli;
pub mod graph_works;
mod pipeline;
pub mod preprocessing;
pub mod utils;

use crate::cli::*;
#[cfg(target_family = "wasm")]
use crate::pipeline::Preprocessed;
use crate::pipeline::{preprocess_step, run_ladder, BuildOpts, LadderState};
#[cfg(not(target_family = "wasm"))]
use crate::utils::io_utils::get_input_list;
use crate::utils::{clock::Instant, io_utils, logging, reads};
use std::fmt;
#[cfg(not(target_family = "wasm"))]
use std::path::{Path, PathBuf};

// Re-export core graph types so callers do not need to depend on sparrowhawk-graph directly.
pub use api::{assemble, assemble_with_graph, Assembly, AssemblyOptions};
pub use sparrowhawk_graph::{
    CarryType, DbgGraph, EdgeId, EdgeType, EdgeWeight, Idx, NodeId, NodeStruct, SerializedContigs,
};
pub use utils::error::AssemblyError;
#[cfg(target_family = "wasm")]
pub use utils::reads::BrowserFiles;
#[cfg(not(target_family = "wasm"))]
pub use utils::reads::InputFiles;
pub use utils::reads::{IteratorSource, MaybeSend, OwnedRecord, ReadSource};

#[cfg(target_family = "wasm")]
use std::io::Write;
#[cfg(target_family = "wasm")]
use wasm_bindgen::prelude::*;

/// Display bins returned to the browser; fitting uses the full preprocessing histogram.
#[cfg(target_family = "wasm")]
const BROWSER_HISTOGRAM_BINS: usize = 500;
#[cfg(target_family = "wasm")]
const _: () = assert!(BROWSER_HISTOGRAM_BINS <= preprocessing::MAXSIZEHISTO);

/// Quality filtering options for FASTQ files
#[derive(Clone, Copy, Debug)]
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

/// Natively a run's log already says where it is.
#[cfg(not(target_family = "wasm"))]
pub(crate) fn post_state(_state: &str) {}

/// Writes the corrected graph, before its collapse, as `<base>.dot`, `.gfa` and `.gfa2`.
#[cfg(not(target_family = "wasm"))]
fn write_graph_files(graph: &DbgGraph, base: &Path) -> std::io::Result<()> {
    use std::io::Write;
    // Append extensions: replacing one would discard dots in the caller's prefix.
    for (extension, render) in [
        ("dot", DbgGraph::get_dot_string as fn(&DbgGraph) -> String),
        ("gfa", DbgGraph::get_gfa_string),
        ("gfa2", DbgGraph::get_gfa2_string),
    ] {
        let mut path = base.as_os_str().to_owned();
        path.push(format!(".{extension}"));
        let mut out = std::io::BufWriter::new(std::fs::File::create(PathBuf::from(path))?);
        let text = render(graph);
        out.write_all(text.as_bytes())?;
        out.flush()?;
    }
    Ok(())
}

#[doc(hidden)]
#[cfg(not(target_family = "wasm"))]
pub fn run_cli() {
    let args = cli_args();

    // log::info!("Starting program!");
    eprintln!("Sparrowhawk");
    if let Err(message) = args.validate() {
        eprintln!("error: {message}");
        std::process::exit(2);
    }
    let Args {
        seq_files,
        file_list,
        output_dir,
        output_prefix,
        k,
        min_count,
        min_qual,
        two_bit_reads,
        threads,
        keep_intermediate_contigs,
        no_histo,
        debug_quality_floors,
        no_graphs,
        verbose,
        ..
    } = &args;
    let mut timevec = Vec::new();
    timevec.push(Instant::now());

    // Create the output directory if it does not exist, so every write below can assume it is
    // there.
    std::fs::create_dir_all(output_dir)
        .unwrap_or_else(|e| panic!("cannot create output directory {output_dir:?}: {e}"));

    let outputlogfile: PathBuf = Path::new(output_dir).join(format!("{output_prefix}_log.txt"));
    if *verbose {
        logging::init(log::LevelFilter::Info, &outputlogfile);
    } else {
        logging::init(log::LevelFilter::Warn, &outputlogfile);
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
    // An explicit --min-qual is used as is, as an explicit --min-count is.
    let fixed_min_qual = min_qual.is_some();
    if *two_bit_reads {
        log::info!("--two-bit-reads: Q0 at every k; no quality-floor selection or recounting");
    } else if fixed_min_qual {
        log::info!("Base-quality floor fixed at {}", quality.min_qual);
    } else {
        log::info!(
            "Base-quality floors chosen automatically, at most {}",
            quality.min_qual
        );
    }

    // Set up multithreading pool
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

    // No extension here: `write_graph_files` appends .dot/.gfa/.gfa2 to this base path.
    let out_path_graph: Option<PathBuf> = if *no_graphs {
        None
    } else {
        Some(Path::new(output_dir).join(format!("{output_prefix}_graph")))
    };

    let output: PathBuf = Path::new(output_dir).join(format!("{output_prefix}_contigs.fasta"));

    // Finished checks, let's put all the options in a convenient struct...
    let mut opts = BuildOpts::from(&AssemblyOptions::from(&args));
    opts.keep_intermediate_contigs = *keep_intermediate_contigs;
    opts.no_histo = *no_histo;
    opts.debug_quality_floors = *debug_quality_floors;
    opts.output_dir = output_dir;
    opts.output_prefix = output_prefix;

    // And do final preparations
    let reads = reads::InputFiles::new(
        input_files
            .iter()
            .flat_map(|(_, files)| files.iter().cloned()),
    )
    .unwrap_or_else(|error| {
        eprintln!("error: {error}");
        std::process::exit(1)
    });
    let mut write_graph = out_path_graph
        .as_deref()
        .map(|path| move |graph: &DbgGraph| write_graph_files(graph, path));
    let graph_sink = write_graph
        .as_mut()
        .map(|write| write as &mut dyn FnMut(&DbgGraph) -> std::io::Result<()>);

    // A single k runs as a ladder of one step; only multi-k packs the reads for later steps.
    let built = LadderState::new(&opts, &reads, k.as_deref(), true).and_then(|mut ladder| {
        preprocess_step(&opts, &reads, 0, &mut ladder, &mut timevec).and_then(|first| {
            run_ladder(&opts, &reads, &mut ladder, first, &mut timevec, graph_sink)
        })
    });

    // In any case, save output contigs
    match built {
        Ok(contigs) => io_utils::save_sequences_as_fasta(&contigs, opts.min_contig_length, output),
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(1);
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

//
//
// ===================================== WebAssembly stuff follows
//
#[cfg(target_family = "wasm")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = log)]
    fn console_log(s: &str);
    #[wasm_bindgen(js_namespace = console, js_name = warn)]
    fn console_warn(s: &str);
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn console_error(s: &str);
    #[wasm_bindgen(js_name = postMessage)]
    fn post_message(data: &JsValue);
    #[wasm_bindgen(js_namespace = performance, js_name = now)]
    fn performance_now() -> f64;
}

#[cfg(target_family = "wasm")]
/// Posts a state update message to the main thread via postMessage
pub(crate) fn post_state(state: &str) {
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
/// The assembler's handle for the page: set up once, then `preprocess` and `assemble`.
pub struct AssemblyHelper {
    k: usize,
    /// Multi-k when set: the requested k values, or empty for the default ladder.
    multik: Option<Vec<usize>>,
    quality: QualOpts,
    /// Whether `quality.min_qual` was given, and is used as is rather than as the ladder's top.
    fixed_min_qual: bool,
    do_bloom: bool,
    do_fit: bool,
    /// The CLI's `--min-contig-length`: shorter contigs are left out of the FASTA.
    min_contig_length: usize,
    /// The CLI's `--bubble-pop-ratio`, set through `set_bubble_pop_ratio`.
    bubble_pop_ratio: f32,
    /// The CLI's `--dead-end-length-kmult`, set through `set_dead_end_length_kmult`.
    dead_end_length_kmult: f32,
    /// The page's files, kept so a multi-k run can read them again.
    reads: Option<reads::BrowserFiles>,
    /// The first k's count, from `preprocess` until `assemble`.
    preprocessed: Option<Preprocessed>,
    ladder: Option<LadderState>,
    /// What `get_preprocessing_info` reports, `spectrum_k` being the k counted.
    histo: Vec<u32>,
    nkmers: usize,
    used_min_count: u16,
    spectrum_k: usize,
    ncontigs: usize,
    /// The last assembly's outputs, gzipped, until the page takes them.
    fasta_gz: Vec<u8>,
    dot_gz: Vec<u8>,
    gfa_gz: Vec<u8>,
    gfa2_gz: Vec<u8>,
}

#[cfg(target_family = "wasm")]
#[wasm_bindgen]
impl AssemblyHelper {
    /// Set up an assembly. `verbose` logs progress to the console, as `-v` does natively. `min_qual`
    /// is used as is when given, and otherwise chosen automatically, as `--min-qual` natively.
    pub fn new(
        k: u32,
        verbose: bool,
        min_count: u16,
        min_qual: Option<u8>,
        do_bloom: bool,
        do_fit: bool,
        min_contig_length: u32,
    ) -> Result<AssemblyHelper, JsError> {
        if cfg!(debug_assertions) {
            init_panic_hook();
        }
        logging::init(if verbose {
            log::LevelFilter::Info
        } else {
            log::LevelFilter::Warn
        });
        cli::validate_bloom_min_count(do_bloom, (!do_fit).then_some(min_count))
            .map_err(JsError::new)?;
        let k = cli::check_k(k as usize).map_err(|message| JsError::new(&message))?;
        post_state("initialised");
        Ok(Self {
            k,
            multik: None,
            quality: QualOpts {
                min_count,
                min_qual: min_qual.unwrap_or(DEFAULT_MINQUAL),
            },
            fixed_min_qual: min_qual.is_some(),
            do_bloom,
            do_fit,
            min_contig_length: min_contig_length as usize,
            bubble_pop_ratio: graph_works::corrector::DEFAULT_POP_RATIO,
            dead_end_length_kmult: cli::DEFAULT_DEAD_END_LEN_KMULT,
            reads: None,
            preprocessed: None,
            ladder: None,
            histo: Vec::new(),
            nkmers: 0,
            used_min_count: 0,
            spectrum_k: k,
            ncontigs: 0,
            fasta_gz: Vec::new(),
            dot_gz: Vec::new(),
            gfa_gz: Vec::new(),
            gfa2_gz: Vec::new(),
        })
    }

    /// Assemble at several k, each seeded with the previous k's contigs, instead of at one. An empty
    /// list is the default ladder, settled from the read length; otherwise odd, ascending k values.
    pub fn set_multik(&mut self, ks: Vec<u32>) -> Result<(), JsError> {
        let ks: Vec<usize> = ks.into_iter().map(|k| k as usize).collect();
        cli::check_ladder(&ks).map_err(|message| JsError::new(&message))?;
        self.multik = Some(ks);
        Ok(())
    }

    /// Fraction of counts needed for popping bubble
    pub fn set_bubble_pop_ratio(&mut self, ratio: f32) {
        if ratio > 0.0 && ratio < 1.0 {
            self.bubble_pop_ratio = ratio;
        } else {
            log::warn!(
                "Ignoring --bubble-pop-ratio {ratio}: it must be strictly between 0 and 1. \
                 Keeping {}.",
                self.bubble_pop_ratio
            );
        }
    }

    /// Multiplier scaling the dead-end removal threshold with k. Zero keeps the flat floor.
    pub fn set_dead_end_length_kmult(&mut self, kmult: f32) {
        if kmult.is_finite() && kmult >= 0.0 {
            self.dead_end_length_kmult = kmult;
        } else {
            log::warn!(
                "Ignoring --dead-end-length-kmult {kmult}: it must be finite and >= 0. Keeping {}.",
                self.dead_end_length_kmult
            );
        }
    }

    /// Parse, filter, and count the first k-value's kmers from the read files (one, or a pair).
    pub fn preprocess(
        &mut self,
        file1: web_sys::File,
        file2: Option<web_sys::File>,
    ) -> Result<(), JsError> {
        post_state("preprocess:start");
        let reads = reads::BrowserFiles::new(std::iter::once(file1).chain(file2).collect())
            .map_err(|error| JsError::new(&error.to_string()))?;
        let opts = self
            .build_opts()
            .map_err(|error| JsError::new(&error.to_string()))?;
        let mut timevec = vec![Instant::now()];
        // A single k is a ladder of one step; an empty multi-k list requests the default ladder.
        let requested = match &self.multik {
            None => Some(vec![self.k]),
            Some(ks) => (!ks.is_empty()).then(|| ks.clone()),
        };
        let mut ladder = LadderState::new(&opts, &reads, requested.as_deref(), false)
            .map_err(|error| JsError::new(&error.to_string()))?;
        self.spectrum_k = ladder.ks[0];
        let preprocessed = preprocess_step(&opts, &reads, 0, &mut ladder, &mut timevec)
            .map_err(|error| JsError::new(&error.to_string()))?;
        self.ladder = Some(ladder);
        let (histovec, nkmers, used_min_count) = preprocessed.summary();
        // Limit the display payload; spectrum fitting has already used the full histogram.
        self.histo = histovec
            .iter()
            .take(BROWSER_HISTOGRAM_BINS)
            .copied()
            .collect();
        self.nkmers = nkmers;
        self.used_min_count = used_min_count;
        self.preprocessed = Some(preprocessed);
        self.reads = Some(reads);
        post_state("preprocess:end");
        Ok(())
    }

    /// Assemble what `preprocess` obtained, and every later k of a multi-k run.
    pub fn assemble(&mut self) -> Result<(), JsError> {
        let preprocessed = self
            .preprocessed
            .take()
            .ok_or_else(|| JsError::new("preprocess must run before assemble"))?;
        let opts = self
            .build_opts()
            .map_err(|error| JsError::new(&error.to_string()))?;
        let mut timevec = vec![Instant::now()];
        let mut graph_text = None;
        let mut keep_graph = |graph: &DbgGraph| {
            graph_text = Some((
                graph.get_dot_string(),
                graph.get_gfa_string(),
                graph.get_gfa2_string(),
            ));
            Ok(())
        };
        let ladder = self
            .ladder
            .as_mut()
            .ok_or_else(|| JsError::new("preprocess must run before assemble"))?;
        let reads = self
            .reads
            .as_ref()
            .ok_or_else(|| JsError::new("preprocess must run before assemble"))?;
        let contigs = run_ladder(
            &opts,
            reads,
            ladder,
            preprocessed,
            &mut timevec,
            Some(&mut keep_graph),
        )
        .map_err(|error| JsError::new(&error.to_string()))?;
        post_state("assembly:saving");
        let mut written = 0;
        self.fasta_gz = io_utils::gzipped(|out| {
            written = io_utils::write_fasta(&contigs, opts.min_contig_length, out)?;
            Ok(())
        });
        self.ncontigs = written;
        log::info!(
            "Omitted {} contigs shorter than {} nt from FASTA output",
            contigs.len() - written,
            opts.min_contig_length
        );
        let (dot, gfa, gfa2) = graph_text.unwrap_or_default();
        self.dot_gz = io_utils::gzipped(|out| out.write_all(dot.as_bytes()));
        self.gfa_gz = io_utils::gzipped(|out| out.write_all(gfa.as_bytes()));
        self.gfa2_gz = io_utils::gzipped(|out| out.write_all(gfa2.as_bytes()));
        post_state("assembly:end");
        Ok(())
    }

    /// The assembly's metadata as JSON; the output buffers are handed over separately.
    pub fn get_assembly(&self) -> String {
        let mut results = json::JsonValue::new_object();
        results["ncontigs"] = json::JsonValue::Number(self.ncontigs.into());
        results.dump()
    }

    /// The contigs as gzipped FASTA. Handed over once: the helper keeps no copy.
    pub fn take_fasta_gz(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.fasta_gz)
    }

    /// The corrected graph, before its collapse, as gzipped DOT. Handed over once.
    pub fn take_dot_gz(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.dot_gz)
    }

    /// The corrected graph, before its collapse, as gzipped GFA 1.1. Handed over once.
    pub fn take_gfa_gz(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.gfa_gz)
    }

    /// The corrected graph, before its collapse, as gzipped GFA 2. Handed over once.
    pub fn take_gfa2_gz(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.gfa2_gz)
    }

    /// Getter to obtain the results as JSON of the preprocessing
    pub fn get_preprocessing_info(&self) -> String {
        let mut results = json::JsonValue::new_object();

        results["nkmers"] = json::JsonValue::Number(self.nkmers.into());
        results["histo"] = json::JsonValue::Array(
            self.histo
                .iter()
                .map(|&x| json::JsonValue::Number(x.into()))
                .collect(),
        );
        results["used_min_count"] = json::JsonValue::Number(self.used_min_count.into());
        // In a multi-k run, the spectrum is the first k's; the later ones are counted in `assemble`.
        results["k"] = json::JsonValue::Number(self.spectrum_k.into());
        results["multik"] = json::JsonValue::Boolean(self.multik.is_some());

        log::debug!("{}", results.dump());

        results.dump()
    }
}

#[cfg(target_family = "wasm")]
impl AssemblyHelper {
    /// The CLI's defaults, with the page's own settings over them.
    fn build_opts(&self) -> Result<BuildOpts<'static>, AssemblyError> {
        let options = AssemblyOptions {
            k_values: self.multik.clone().unwrap_or_else(|| vec![self.k]),
            min_count: (!self.do_fit).then_some(self.quality.min_count),
            min_qual: self.fixed_min_qual.then_some(self.quality.min_qual),
            use_bloom: self.do_bloom,
            min_contig_length: self.min_contig_length,
            pop_ratio: self.bubble_pop_ratio,
            dead_end_length_kmult: self.dead_end_length_kmult,
            ..AssemblyOptions::default()
        };
        options.validate()?;
        Ok(BuildOpts::from(&options))
    }
}
