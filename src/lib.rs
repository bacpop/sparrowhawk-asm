//! Efficient genome assembler for bacterial species in Rust, browser-ready
#![warn(missing_docs)]
use std::fmt;
use std::path::{Path, PathBuf};

use crate::clock::Instant;

/// A clock that also works in the browser
pub mod clock;

/// The logger, natively to stdout and the log file, in the browser to the console
pub mod logging;

/// Where reads come from, read again as often as a run needs
pub mod reads;

/// Construction, assembly, shrinkage, pruning, and collapse of DNA de Bruijn graphs
pub mod graph_works;

/// Aligned k-mer storage between counting and graph construction.
pub mod indexed_kmers;

/// Reads packed once in memory, so a multi-k run parses its input a single time.
pub mod read_store;

/// Preprocessing functions of the reads & k-mers
pub mod preprocessing;

/// Candidate base-quality floors, and which of them each k-mer window clears
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

/// Fits the k-mer spectrum to automatically get a min_count (taken from ska.rust!)
pub mod spectrum_fitter;

use crate::bit_encoding::UInt;
use crate::graph_works::Assemble;
use crate::reads::ReadSource;
use bit_encoding::{U256, U512};
use sparrowhawk_graph::DbgGraph;

// Re-export core graph types so callers do not need to depend on sparrowhawk-graph directly.
pub use sparrowhawk_graph::{EdgeType, EdgeWeight, HashInfoSimple, Idx};

pub mod cli;

use crate::cli::*;

pub mod io_utils;

#[cfg(not(target_family = "wasm"))]
use crate::io_utils::*;

#[cfg(target_family = "wasm")]
use wasm_bindgen::prelude::*;
#[cfg(target_family = "wasm")]
extern crate console_error_panic_hook;
#[cfg(target_family = "wasm")]
pub mod fastx_wasm;

/// Quality filtering options for FASTQ files
#[derive(Clone, Copy)]
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

/// Everything a build needs besides its reads, on both targets. The browser writes no files, so it
/// leaves the output fields empty and `no_histo` set.
struct BuildOpts<'a> {
    quality: QualOpts,
    /// Count at `quality.min_qual` alone, rather than at the reads' ladder of floors topped by it.
    fixed_min_qual: bool,
    /// Enabled by default
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
    debug_quality_floors: bool,
    output_dir: &'a str,
    output_prefix: &'a str,
}

impl BuildOpts<'_> {
    /// Correction settings for one k: tip thresholds scale with k, coverage comes from its spectrum. Small helper that gets the info from buildopts
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
        (!self.no_histo || self.debug_quality_floors).then(|| {
            Path::new(self.output_dir).join(format!("{}_kmerspectrum{tag}.png", self.output_prefix))
        })
    }
}

#[cfg(any(test, target_family = "wasm"))]
impl BuildOpts<'static> {
    /// The CLI's defaults, writing no files: the browser's settings, and the tests' starting point.
    fn with_defaults(quality: QualOpts, do_bloom: bool, do_fit: bool) -> Self {
        use algorithms::corrector::{DEFAULT_EC_COVERAGE_RATIO, DEFAULT_POP_RATIO};
        Self {
            quality,
            fixed_min_qual: false,
            do_bloom,
            do_fit,
            do_bubble_collapse: true,
            do_dead_end_removal: true,
            pop_ratio: DEFAULT_POP_RATIO,
            peak_ratio: None,
            tip_length: DEFAULT_TIP_LEN_NTS,
            tip_length_kmult: DEFAULT_TIP_LEN_KMULT,
            tip_length_rctc_kmult: Some(DEFAULT_TIP_RCTC_KMULT),
            tip_rctc_cutoff: DEFAULT_TIP_RCTC_CUTOFF,
            do_ec_removal: true,
            ec_ratio: DEFAULT_EC_COVERAGE_RATIO,
            ec_require_both_flanks: false,
            min_contig_length: DEFAULT_MIN_CONTIG_LENGTH_NTS,
            contig_counts: ContigCountRule::Floor,
            keep_intermediate_contigs: false,
            no_histo: true,
            debug_quality_floors: false,
            output_dir: "",
            output_prefix: "",
        }
    }
}

/// One k's counted k-mers, at the packed width its k needs.
enum CountedK {
    W64(preprocessing::PreprocessedK<u64>),
    W128(preprocessing::PreprocessedK<u128>),
    W256(preprocessing::PreprocessedK<U256>),
    W512(preprocessing::PreprocessedK<U512>),
}

/// Count one k's k-mers in one pass over `records`, carrying `carried` in once the reads have set the
/// cutoff. `prior`: the first count's selection when this is a recount; see `keep_refit_or_prior`.
#[allow(clippy::too_many_arguments)]
fn count<IntT, I>(
    opts: &BuildOpts,
    k: usize,
    quality: &QualOpts,
    floors: Option<&[u8]>,
    records: I,
    carried: Option<preprocessing::CarriedContigs>,
    prior: Option<(u16, preprocessing::PeakSource)>,
    timevec: &mut Vec<Instant>,
    out_path_histo: &mut Option<PathBuf>,
) -> Result<preprocessing::PreprocessedK<IntT>, preprocessing::PreprocessingError>
where
    IntT: for<'a> UInt<'a>,
    I: Iterator<Item = reads::OwnedRecord> + reads::MaybeSend,
{
    preprocessing::preprocessing_standalone::<IntT, _>(
        &mut [records],
        k,
        quality,
        floors,
        &mut Some(timevec),
        out_path_histo,
        opts.do_bloom,
        opts.do_fit,
        carried,
        opts.debug_quality_floors,
        prior,
    )
}

/// Count one k at the strict floor of the reads' quality ladder, and once more at a looser floor if
/// the spectrum asks for it.
fn count_single<S: ReadSource>(
    opts: &BuildOpts,
    reads: &S,
    k: usize,
    timevec: &mut Vec<Instant>,
) -> Result<CountedK, preprocessing::PreprocessingError> {
    match k {
        3..=32 => {
            log::info!("k={k}: using 64-bit representation");
            count_single_at::<u64, S>(opts, reads, k, timevec).map(CountedK::W64)
        }
        33..=64 => {
            log::info!("k={k}: using 128-bit representation");
            count_single_at::<u128, S>(opts, reads, k, timevec).map(CountedK::W128)
        }
        65..=128 => {
            log::info!("k={k}: using 256-bit representation");
            count_single_at::<U256, S>(opts, reads, k, timevec).map(CountedK::W256)
        }
        _ => {
            log::info!("k={k}: using 512-bit representation");
            count_single_at::<U512, S>(opts, reads, k, timevec).map(CountedK::W512)
        }
    }
}

/// The base-quality floors a k is counted at, ascending: the given floor alone when it is fixed, which
/// can never be loosened, or else the ladder the reads' qualities allow, topped by it.
fn quality_floors<S: ReadSource>(opts: &BuildOpts, reads: &S) -> Vec<u8> {
    let floors = if opts.fixed_min_qual {
        vec![opts.quality.min_qual]
    } else {
        // The alphabet only, from the head of the reads: a few thousand show all 4-5 bins, and no
        // spectrum is built from them, so the depth of the peek does not matter.
        qual_profile::floors_from(
            &qual_profile::peek_alphabet(reads.records(), qual_profile::PEEK_READS),
            opts.quality.min_qual,
        )
    };
    log::info!("Candidate base-quality floors: {floors:?}");
    floors
}

fn count_single_at<IntT, S>(
    opts: &BuildOpts,
    reads: &S,
    k: usize,
    timevec: &mut Vec<Instant>,
) -> Result<preprocessing::PreprocessedK<IntT>, preprocessing::PreprocessingError>
where
    IntT: for<'a> UInt<'a>,
    S: ReadSource,
{
    let ladder = quality_floors(opts, reads);
    // The ladder's strict floor, not the default: with `[0]`, no quality filtering.
    let quality = QualOpts {
        min_count: opts.quality.min_count,
        min_qual: ladder[ladder.len() - 1],
    };
    let mut out_path_histo = opts.histo_path("");
    let first = count::<IntT, _>(
        opts,
        k,
        &quality,
        Some(&ladder),
        reads.records(),
        None,
        None,
        timevec,
        &mut out_path_histo,
    )?;
    let chosen = first.chosen_min_qual;
    if chosen >= quality.min_qual {
        return Ok(first);
    }
    // Why it loosened is logged by the estimator, which is the only place that knows.
    log::warn!(
        "Recounting at a base-quality floor of {chosen} instead of {} — this admits more error \
         k-mers.",
        quality.min_qual
    );
    post_state(&format!("preprocess:recount:{chosen}"));
    let loosened = QualOpts {
        min_count: quality.min_count,
        min_qual: chosen,
    };
    // The sketch's selection stands if the refit at this floor loses the separation, as in multi-k.
    let prior = Some((first.used_min_count, first.genomic_peak));
    // Drop pass 1 before pass 2 allocates, or both tables are resident at once.
    drop(first);
    count::<IntT, _>(
        opts,
        k,
        &loosened,
        None,
        reads.records(),
        None,
        prior,
        timevec,
        &mut out_path_histo,
    )
}

/// Assemble one counted k and spell its contigs.
fn assemble_single(
    opts: &BuildOpts,
    counted: CountedK,
    timevec: &mut Vec<Instant>,
    graph_sink: Option<&mut dyn FnMut(&DbgGraph)>,
) -> Vec<Vec<u8>> {
    match counted {
        CountedK::W64(mut c) => assemble_counted(opts, &mut c, timevec, graph_sink),
        CountedK::W128(mut c) => assemble_counted(opts, &mut c, timevec, graph_sink),
        CountedK::W256(mut c) => assemble_counted(opts, &mut c, timevec, graph_sink),
        CountedK::W512(mut c) => assemble_counted(opts, &mut c, timevec, graph_sink),
    }
}

/// Build, correct and collapse one k's graph, then spell the contigs from its k-mers.
fn assemble_counted<IntT: for<'a> UInt<'a>>(
    opts: &BuildOpts,
    counted: &mut preprocessing::PreprocessedK<IntT>,
    timevec: &mut Vec<Instant>,
    graph_sink: Option<&mut dyn FnMut(&DbgGraph)>,
) -> Vec<Vec<u8>> {
    let k = counted.k;
    // The spectrum of the pass that produced the k-mers: after a recount, pass 2's.
    let correction = opts.correction(k, counted.genomic_peak, counted.used_min_count);
    let contigs = graph_works::BasicAsm::assemble::<IntT>(
        k,
        &mut counted.kmers,
        &mut Some(timevec),
        graph_sink,
        correction,
    );
    save_functions::spell_contigs::<IntT>(&contigs, &counted.kmers, k)
}

/// How a multi-k run reads its input again after the first k.
enum Replay {
    /// Every read, packed during the first k's pass and replayed by the rest (native).
    Store(Box<read_store::ReadStore>),
    /// The input itself, read again for every k (the browser), so only the read lengths are kept.
    Reread(read_store::ReadStats),
}

impl Replay {
    fn stats(&self) -> read_store::ReadStats {
        match self {
            Self::Store(store) => store.stats(),
            Self::Reread(stats) => *stats,
        }
    }
}

/// What a multi-k run carries from one k to the next.
struct LadderState {
    /// The k values to run; a default ladder is settled once the first k has seen every read.
    ks: Vec<usize>,
    default_ladder: bool,
    /// The reads' quality ladder, ascending.
    floors: Vec<u8>,
    replay: Replay,
    /// Index of the strictest floor later k may use. Lowered only once two consecutive resolved k chose
    /// below it (see [`lowered_floor_index`]), and never raised.
    floor_index: usize,
    /// The floor the last resolved k chose; unresolved k leave it unchanged.
    last_choice: Option<u8>,
    /// Current quality/min-count settings. `min_qual` is the ceiling, `floors[floor_index]`.
    quality: QualOpts,
    /// The previous k's contigs, spelled.
    contigs: Vec<Vec<u8>>,
}

impl LadderState {
    /// A ladder over `reads`: the `requested` k values, or the default ladder, settled after the first
    /// k. `in_memory` packs the reads during the first k; otherwise every k reads them again.
    fn new<S: ReadSource>(
        opts: &BuildOpts,
        reads: &S,
        requested: Option<&[usize]>,
        in_memory: bool,
    ) -> Self {
        let floors = quality_floors(opts, reads);
        if let Some(ks) = requested {
            post_ladder(ks);
        }
        let replay = if in_memory {
            Replay::Store(Box::new(read_store::ReadStore::new(&floors)))
        } else {
            Replay::Reread(read_store::ReadStats::default())
        };
        Self {
            ks: requested.map_or_else(|| vec![DEFAULT_KMER_LADDER[0]], <[usize]>::to_vec),
            default_ladder: requested.is_none(),
            floor_index: floors.len() - 1,
            last_choice: None,
            quality: QualOpts {
                min_count: opts.quality.min_count,
                // The strict floor, as `record_floor_choice` keeps it: with `[0]`, no quality filtering.
                min_qual: floors[floors.len() - 1],
            },
            contigs: Vec::new(),
            floors,
            replay,
        }
    }

    /// Record the floor this k chose. An unresolved k is skipped: its loosest-floor fallback is not a
    /// selection, so it neither lowers the ceiling nor counts towards a pair.
    fn record_floor_choice(&mut self, k: usize, chosen: u8, unresolved: bool) {
        if unresolved {
            return;
        }
        let floors = &self.floors;
        let lowered = lowered_floor_index(floors, self.floor_index, self.last_choice, chosen);
        if lowered != self.floor_index {
            log::info!(
                "A floor below {} was chosen at two consecutive k (the last at k={k}); later k values \
                 will not go above {}.",
                floors[self.floor_index],
                floors[lowered]
            );
            self.floor_index = lowered;
            self.quality.min_qual = floors[lowered];
        }
        self.last_choice = Some(chosen);
    }
}

/// A k after the first with a single-copy peak below this, fitted or fallback, is not assembled: the
/// ladder stops on the previous contigs. At 3, a 13x simulation gained a misassembly.
const MIN_LADDER_PEAK: u32 = 4;

/// Whether a k after the first has too little coverage to assemble; an unknown peak counts as too low.
fn peak_too_low_for_ladder(step: usize, peak: preprocessing::PeakSource) -> bool {
    step > 0 && peak.value().is_none_or(|p| p < MIN_LADDER_PEAK)
}

/// Length-weighted mean length of the contigs of at least `min_len` bases; 0 when there are none.
fn aun(contigs: &[Vec<u8>], min_len: usize) -> f64 {
    let (sum, sum_sq) = contigs
        .iter()
        .map(|c| c.len() as u64)
        .filter(|&l| l >= min_len as u64)
        .fold((0u64, 0u64), |(s, q), l| (s + l, q + l * l));
    if sum == 0 {
        0.0
    } else {
        sum_sq as f64 / sum as f64
    }
}

/// Count ladder step `step` at the packed width its k needs.
fn count_step<S: ReadSource>(
    opts: &BuildOpts,
    reads: &S,
    step: usize,
    state: &mut LadderState,
    timevec: &mut Vec<Instant>,
) -> Result<CountedK, preprocessing::PreprocessingError> {
    let k = state.ks[step];
    // The default ladder is only settled from the read lengths once its first k has been counted.
    if state.default_ladder && step == 0 {
        log::info!("Multi-k step 1: k={k} (the rest of the ladder is set after it)");
    } else {
        log::info!("Multi-k step {} of {}: k={k}", step + 1, state.ks.len());
    }
    post_state(&format!("multik:step:{}:{k}", step + 1));
    match k {
        3..=32 => ladder_count::<u64, S>(opts, reads, step, state, timevec).map(CountedK::W64),
        33..=64 => ladder_count::<u128, S>(opts, reads, step, state, timevec).map(CountedK::W128),
        65..=128 => ladder_count::<U256, S>(opts, reads, step, state, timevec).map(CountedK::W256),
        _ => ladder_count::<U512, S>(opts, reads, step, state, timevec).map(CountedK::W512),
    }
}

/// The first k reads the input, packing or measuring it, and settles the ladder; a later k counts from
/// the store or reads the input again. Then the floor is settled, recounting if it loosened.
fn ladder_count<IntT, S>(
    opts: &BuildOpts,
    reads: &S,
    step: usize,
    state: &mut LadderState,
    timevec: &mut Vec<Instant>,
) -> Result<preprocessing::PreprocessedK<IntT>, preprocessing::PreprocessingError>
where
    IntT: for<'a> UInt<'a>,
    S: ReadSource,
{
    let k = state.ks[step];
    let mut out_path_histo = opts.histo_path(&format!("_k{k}"));
    let active_floors = active_quality_floors(&state.floors, state.floor_index).to_vec();
    let strict_floor = state.floors[state.floor_index];
    let quality = state.quality;
    let first = if step == 0 {
        let floors = Some(active_floors.as_slice());
        let pass1 = match &mut state.replay {
            Replay::Store(store) => count::<IntT, _>(
                opts,
                k,
                &quality,
                floors,
                read_store::Tee::new(reads.records(), &mut **store),
                None,
                None,
                timevec,
                &mut out_path_histo,
            )?,
            Replay::Reread(stats) => count::<IntT, _>(
                opts,
                k,
                &quality,
                floors,
                read_store::Tee::new(reads.records(), stats),
                None,
                None,
                timevec,
                &mut out_path_histo,
            )?,
        };
        if let Replay::Store(store) = &mut state.replay {
            store.finish();
        }
        let stats = state.replay.stats();
        state.ks = settle_ladder(
            &state.ks,
            state.default_ladder,
            stats.average_read_len().unwrap_or(0.0),
            stats.max_read_len(),
        );
        post_ladder(&state.ks);
        pass1
    } else {
        count_again::<IntT, S>(
            opts,
            reads,
            k,
            &quality,
            Some(&active_floors),
            state,
            None,
            timevec,
            &mut out_path_histo,
        )?
    };
    let assembly = settle_floor_and_recount::<IntT, S>(
        opts,
        reads,
        k,
        state,
        first,
        strict_floor,
        timevec,
        &mut out_path_histo,
    )?;
    if peak_too_low_for_ladder(step, assembly.genomic_peak) {
        return Err(preprocessing::PreprocessingError::peak_too_low(
            k,
            assembly.genomic_peak,
        ));
    }
    Ok(assembly)
}

/// Count k again, from the store or by reading the input again, carrying the previous contigs in.
#[allow(clippy::too_many_arguments)]
fn count_again<IntT, S>(
    opts: &BuildOpts,
    reads: &S,
    k: usize,
    quality: &QualOpts,
    floors: Option<&[u8]>,
    state: &LadderState,
    prior: Option<(u16, preprocessing::PeakSource)>,
    timevec: &mut Vec<Instant>,
    out_path_histo: &mut Option<PathBuf>,
) -> Result<preprocessing::PreprocessedK<IntT>, preprocessing::PreprocessingError>
where
    IntT: for<'a> UInt<'a>,
    S: ReadSource,
{
    let carried = (!state.contigs.is_empty()).then_some(preprocessing::CarriedContigs {
        seqs: &state.contigs,
        rule: opts.contig_counts,
    });
    match &state.replay {
        Replay::Store(store) => count::<IntT, _>(
            opts,
            k,
            quality,
            floors,
            store.records(),
            carried,
            prior,
            timevec,
            out_path_histo,
        ),
        Replay::Reread(_) => count::<IntT, _>(
            opts,
            k,
            quality,
            floors,
            reads.records(),
            carried,
            prior,
            timevec,
            out_path_histo,
        ),
    }
}

/// Record the floor this k's first count chose and, if it is below `strict_floor`, recount there,
/// refitting the whole spectrum; the sketch's cutoff and peak stand only if the refit loses the
/// separation. The first count is dropped before the recount allocates, so both tables are never
/// resident at once.
#[allow(clippy::too_many_arguments)]
fn settle_floor_and_recount<IntT, S>(
    opts: &BuildOpts,
    reads: &S,
    k: usize,
    state: &mut LadderState,
    first: preprocessing::PreprocessedK<IntT>,
    strict_floor: u8,
    timevec: &mut Vec<Instant>,
    out_path_histo: &mut Option<PathBuf>,
) -> Result<preprocessing::PreprocessedK<IntT>, preprocessing::PreprocessingError>
where
    IntT: for<'a> UInt<'a>,
    S: ReadSource,
{
    if !opts.do_fit {
        return Ok(first);
    }
    state.record_floor_choice(k, first.chosen_min_qual, first.min_count_unresolved);
    if first.chosen_min_qual >= strict_floor {
        return Ok(first);
    }
    let (floor, min_count, peak) = (
        first.chosen_min_qual,
        first.used_min_count,
        first.genomic_peak,
    );
    log::warn!("Recounting k={k} at a base-quality floor of {floor}.");
    post_state(&format!("preprocess:recount:{floor}"));
    drop(first);
    let quality = QualOpts {
        min_count,
        min_qual: floor,
    };
    count_again::<IntT, S>(
        opts,
        reads,
        k,
        &quality,
        None,
        state,
        Some((min_count, peak)),
        timevec,
        out_path_histo,
    )
}

/// Assemble ladder step `step` at its width.
fn assemble_step(
    opts: &BuildOpts,
    step: usize,
    state: &mut LadderState,
    counted: CountedK,
    timevec: &mut Vec<Instant>,
    graph_sink: &mut Option<&mut dyn FnMut(&DbgGraph)>,
) -> Result<(), preprocessing::PreprocessingError> {
    match counted {
        CountedK::W64(c) => ladder_assemble(opts, step, state, c, timevec, graph_sink),
        CountedK::W128(c) => ladder_assemble(opts, step, state, c, timevec, graph_sink),
        CountedK::W256(c) => ladder_assemble(opts, step, state, c, timevec, graph_sink),
        CountedK::W512(c) => ladder_assemble(opts, step, state, c, timevec, graph_sink),
    }
}

/// Assemble one step and keep its contigs for the next k. The graph goes to `graph_sink` only at the
/// last k, and only when its contigs are sure to be kept.
fn ladder_assemble<IntT: for<'a> UInt<'a>>(
    opts: &BuildOpts,
    step: usize,
    state: &mut LadderState,
    mut assembly: preprocessing::PreprocessedK<IntT>,
    timevec: &mut Vec<Instant>,
    graph_sink: &mut Option<&mut dyn FnMut(&DbgGraph)>,
) -> Result<(), preprocessing::PreprocessingError> {
    let k = state.ks[step];
    // An unresolved cutoff must beat the previous contigs, so they are kept until it has; otherwise
    // they are carried into the count-map by now and dead weight.
    let unresolved = step > 0 && assembly.min_count_unresolved;
    let previous = unresolved.then(|| std::mem::take(&mut state.contigs));
    state.contigs = Vec::new();

    let last = step + 1 == state.ks.len();
    // A k that may still be rejected sends no graph, as a stopped ladder never has.
    let sink = if last && !unresolved {
        graph_sink.take()
    } else {
        None
    };
    state.contigs = assemble_counted(opts, &mut assembly, timevec, sink);
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
    if let Some(previous) = previous {
        let (new, old) = (
            aun(&state.contigs, opts.min_contig_length),
            aun(&previous, opts.min_contig_length),
        );
        if new <= old {
            state.contigs = previous;
            return Err(preprocessing::PreprocessingError::no_aun_gain(k, new, old));
        }
        log::info!(
            "k={k}: kept despite an unresolved cutoff, as auN rose from {old:.1} to {new:.1}."
        );
    }
    Ok(())
}

/// Assemble the ladder from its first k, already counted, to its last; each later k is counted with
/// the previous contigs carried in. A k that fails ends the ladder on the previous contigs.
fn run_ladder<S: ReadSource>(
    opts: &BuildOpts,
    reads: &S,
    state: &mut LadderState,
    first: CountedK,
    timevec: &mut Vec<Instant>,
    mut graph_sink: Option<&mut dyn FnMut(&DbgGraph)>,
) -> Result<Vec<Vec<u8>>, preprocessing::PreprocessingError> {
    let mut counted = Ok(first);
    let mut step = 0;
    loop {
        let outcome =
            counted.and_then(|c| assemble_step(opts, step, state, c, timevec, &mut graph_sink));
        if let Err(error) = outcome {
            // A k with no usable k-mers, too low a peak, or an unresolved cutoff that did not raise
            // auN ends the ladder. In every case the previous contigs are in `state`.
            if step == 0 {
                return Err(error);
            }
            log::warn!(
                "Stopping the multi-k ladder at k={}: {error}. Keeping the k={} contigs; a stopped \
                 ladder writes no graph.",
                state.ks[step],
                state.ks[step - 1]
            );
            break;
        }
        step += 1;
        if step == state.ks.len() {
            break;
        }
        counted = count_step(opts, reads, step, state, timevec);
    }
    Ok(std::mem::take(&mut state.contigs))
}

/// Eligible floors are the current strict floor and all rungs below it.
fn active_quality_floors(floors: &[u8], strict_index: usize) -> &[u8] {
    &floors[..=strict_index]
}

/// The ceiling for later k: the higher of two consecutive choices once both are below the current
/// ceiling, otherwise unchanged. The chosen floors come from the ladder, so `position` finds them.
fn lowered_floor_index(floors: &[u8], ceiling: usize, previous: Option<u8>, chosen: u8) -> usize {
    let top = floors[ceiling];
    match previous {
        Some(previous) if previous < top && chosen < top => floors
            .iter()
            .position(|&floor| floor == previous.max(chosen))
            .unwrap_or(ceiling),
        _ => ceiling,
    }
}

/// The k values to run once every read is known: the default up to the longest read less
/// [`READ_LEN_MARGIN`], or the requested list without k values not below the longest read. The first k
/// has run, so it always stays.
fn settle_ladder(
    ks: &[usize],
    default_ladder: bool,
    average_read_len: f64,
    max_read_len: usize,
) -> Vec<usize> {
    let ceiling = max_read_len.saturating_sub(READ_LEN_MARGIN);
    let settled: Vec<usize> = if default_ladder {
        default_kmer_ladder(ceiling)
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
    if default_ladder {
        log::info!(
            "Multi-k ladder: average_read_len={average_read_len:.1} bp, \
             longest read={max_read_len} bp, ceiling={ceiling} bp, k={settled:?}"
        );
    } else {
        log::info!("Multi-k ladder for reads up to {max_read_len} bp: k={settled:?}");
    }
    settled
}

/// Natively a run's log already says where it is.
#[cfg(not(target_family = "wasm"))]
pub(crate) fn post_state(_state: &str) {}

/// Tells the page every k of a multi-k ladder, so it can say how far along the ladder a run is.
fn post_ladder(ks: &[usize]) {
    let ks: Vec<String> = ks.iter().map(usize::to_string).collect();
    post_state(&format!("multik:ladder:{}", ks.join(",")));
}

/// Writes the corrected graph, before its collapse, as `<base>.dot`, `.gfa` and `.gfa2`.
#[cfg(not(target_family = "wasm"))]
fn write_graph_files(graph: &DbgGraph, base: &Path) {
    log::info!(
        "Saving graph (post-shrink, pre-collapse, w/o one-node contigs) as DOT, GFAv1.1, and GFAv2 \
         files..."
    );
    let open = |extension: &str| {
        set_ostream(&Some(
            base.with_extension(extension)
                .into_os_string()
                .into_string()
                .unwrap(),
        ))
    };
    graph.write_to_dot(&mut open("dot"));
    graph.write_to_gfa(&mut open("gfa"));
    graph.write_to_gfa2(&mut open("gfa2"));
    log::info!("Done.");
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
            debug_quality_floors,
            no_graphs,
            no_bubble_collapse,
            no_dead_end_removal,
        } => {
            let do_bloom = !*no_bloom;
            if let Err(message) = cli::validate_bloom_min_count(do_bloom, *min_count) {
                eprintln!("error: {message}");
                std::process::exit(2);
            }
            if *debug_quality_floors && min_count.is_some() {
                eprintln!(
                    "error: --debug-quality-floors requires automatic min-count selection; remove --min-count"
                );
                std::process::exit(2);
            }
            if *debug_quality_floors && min_qual.is_some() {
                eprintln!(
                    "error: --debug-quality-floors requires automatic base-quality floors; remove --min-qual"
                );
                std::process::exit(2);
            }

            // Create the output directory if it does not exist, so every write below can assume it is
            // there.
            std::fs::create_dir_all(output_dir)
                .unwrap_or_else(|e| panic!("cannot create output directory {output_dir:?}: {e}"));

            let outputlogfile: PathBuf =
                Path::new(output_dir).join(format!("{output_prefix}_log.txt"));
            if args.verbose {
                logging::init(log::LevelFilter::Info, &outputlogfile);
            } else {
                logging::init(log::LevelFilter::Warn, &outputlogfile);
            }
            if *chunk_size != 0 {
                log::warn!(
                    "--chunk-size is ignored: k-mers are counted into a hash map, which buffers no \
                     occurrences, so memory is bounded by the number of distinct k-mers instead."
                );
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
            if fixed_min_qual {
                log::info!("Base-quality floor fixed at {}", quality.min_qual);
            } else {
                log::info!(
                    "Base-quality floors chosen automatically, at most {}",
                    quality.min_qual
                );
            }

            // Set up multithreadding pool
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

            // No extension here: `write_graph_files` sets .dot/.gfa/.gfa2 on this base path.
            let out_path_graph: Option<PathBuf> = if *no_graphs {
                None
            } else {
                Some(Path::new(output_dir).join(format!("{output_prefix}_graph")))
            };

            let output: PathBuf =
                Path::new(output_dir).join(format!("{output_prefix}_contigs.fasta"));
            
            // Checks for invalid values of parameter
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
                if let Err(message) = cli::check_ladder(ks) {
                    eprintln!("error: {message}");
                    std::process::exit(2);
                }
            }

            // Finished checks, let's put all the options in a convenient struct...
            let opts = BuildOpts {
                quality,
                fixed_min_qual,
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
                debug_quality_floors: *debug_quality_floors,
                output_dir,
                output_prefix,
            };
            
            // And do final preparations
            let reads = reads::InputFiles(&input_files);
            let mut write_graph = out_path_graph
                .as_deref()
                .map(|path| move |graph: &DbgGraph| write_graph_files(graph, path));
            let graph_sink = write_graph
                .as_mut()
                .map(|write| write as &mut dyn FnMut(&DbgGraph));

            // ...and RUN!!
            let built = match k.as_deref() {
                // single-k
                Some(&[single]) => count_single(&opts, &reads, single, &mut timevec)
                    .map(|counted| assemble_single(&opts, counted, &mut timevec, graph_sink)),
                // multi-k
                requested => {
                    let mut ladder = LadderState::new(&opts, &reads, requested, true);
                    count_step(&opts, &reads, 0, &mut ladder, &mut timevec).and_then(|first| {
                        run_ladder(&opts, &reads, &mut ladder, first, &mut timevec, graph_sink)
                    })
                }
            };
            
            // In any case, save output contigs
            match built {
                Ok(contigs) => save_functions::save_sequences_as_fasta(
                    &contigs,
                    opts.min_contig_length,
                    output,
                ),
                Err(error) => {
                    eprintln!("error: {error}");
                    std::process::exit(1);
                }
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

//
//
// ===================================== WebAssembly stuff follows
//
#[cfg(target_family = "wasm")]
/// Binary dummy function. In the future, we need to completely remove it whenever compilating with the feature "wasm"
pub fn main() {
    panic!("You've compiled Sparrowhawk for WebAssembly support, you cannot use it as a normal binary anymore!");
}

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
    /// The CLI's `--tip-length-kmult`, set through `set_tip_length_kmult`.
    tip_length_kmult: f32,
    /// The page's files, kept so a multi-k run can read them again.
    reads: Option<reads::BrowserFiles>,
    /// The first k's count, from `preprocess` until `assemble`.
    counted: Option<CountedK>,
    ladder: Option<LadderState>,
    /// What `get_preprocessing_info` reports, `spectrum_k` being the k counted.
    histo: Vec<u32>,
    nkmers: usize,
    used_min_count: u16,
    spectrum_k: usize,
    ncontigs: usize,
    outfasta: String,
    outdot: String,
    outgfa: String,
    outgfav2: String,
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
            bubble_pop_ratio: algorithms::corrector::DEFAULT_POP_RATIO,
            tip_length_kmult: cli::DEFAULT_TIP_LEN_KMULT,
            reads: None,
            counted: None,
            ladder: None,
            histo: Vec::new(),
            nkmers: 0,
            used_min_count: 0,
            spectrum_k: k,
            ncontigs: 0,
            outfasta: String::new(),
            outdot: String::new(),
            outgfa: String::new(),
            outgfav2: String::new(),
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

    /// Multiplier scaling the tip-removal threshold with k. Zero keeps the flat floor.
    pub fn set_tip_length_kmult(&mut self, kmult: f32) {
        if kmult >= 0.0 {
            self.tip_length_kmult = kmult;
        } else {
            log::warn!(
                "Ignoring --tip-length-kmult {kmult}: it must be >= 0. Keeping {}.",
                self.tip_length_kmult
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
        let reads = reads::BrowserFiles(std::iter::once(file1).chain(file2).collect());
        let opts = self.build_opts();
        let mut timevec = vec![Instant::now()];
        let counted = match &self.multik {
            None => {
                self.spectrum_k = self.k;
                count_single(&opts, &reads, self.k, &mut timevec)?
            }
            Some(ks) => {
                let requested = (!ks.is_empty()).then_some(ks.as_slice());
                let mut ladder = LadderState::new(&opts, &reads, requested, false);
                // Before the first count settles the rest of the ladder.
                self.spectrum_k = ladder.ks[0];
                let first = count_step(&opts, &reads, 0, &mut ladder, &mut timevec)?;
                self.ladder = Some(ladder);
                first
            }
        };
        let (histovec, nkmers, used_min_count) = counted.summary();
        // Only the historical range: the page plots whatever it is handed.
        self.histo = histovec
            .iter()
            .take(preprocessing::LEGACY_HISTO_RANGE)
            .copied()
            .collect();
        self.nkmers = nkmers;
        self.used_min_count = used_min_count;
        self.counted = Some(counted);
        self.reads = Some(reads);
        post_state("preprocess:end");
        Ok(())
    }

    /// Assemble what `preprocess` obtained, and every later k of a multi-k run.
    pub fn assemble(&mut self) -> Result<(), JsError> {
        let counted = self
            .counted
            .take()
            .ok_or_else(|| JsError::new("preprocess must run before assemble"))?;
        let opts = self.build_opts();
        let mut timevec = vec![Instant::now()];
        let mut graph_text = None;
        let mut keep_graph = |graph: &DbgGraph| {
            graph_text = Some((
                graph.get_dot_string(),
                graph.get_gfa_string(),
                graph.get_gfa2_string(),
            ));
        };
        let contigs = match (self.ladder.as_mut(), self.reads.as_ref()) {
            (Some(ladder), Some(reads)) => run_ladder(
                &opts,
                reads,
                ladder,
                counted,
                &mut timevec,
                Some(&mut keep_graph),
            )?,
            _ => assemble_single(&opts, counted, &mut timevec, Some(&mut keep_graph)),
        };
        post_state("assembly:saving");
        let kept = save_functions::at_least(&contigs, opts.min_contig_length);
        log::info!(
            "Omitted {} contigs shorter than {} nt from FASTA output",
            contigs.len() - kept.len(),
            opts.min_contig_length
        );
        self.ncontigs = kept.len();
        self.outfasta = save_functions::fasta_string(&kept);
        (self.outdot, self.outgfa, self.outgfav2) = graph_text.unwrap_or_default();
        post_state("assembly:end");
        Ok(())
    }

    /// Getter to obtain the results as JSON of the assembly
    /// TODO: do as we do in ska.rust, so that the output is directly written to disk? Also, compression? Though, is it preferable to keep the contigs compressed/uncompressed, or the graph from which they come? In this case probably I'll need to keep the strings better, but it should be better still to offer them compressed (and we'll save some space as well).
    pub fn get_assembly(&self) -> String {
        let mut results = json::JsonValue::new_array();

        results["outfasta"] = json::JsonValue::String(self.outfasta.clone());
        results["outdot"] = json::JsonValue::String(self.outdot.clone());
        results["outgfa"] = json::JsonValue::String(self.outgfa.clone());
        results["outgfav2"] = json::JsonValue::String(self.outgfav2.clone());
        results["ncontigs"] = json::JsonValue::Number(self.ncontigs.into());

        results.dump()
    }

    /// Getter to obtain the results as JSON of the preprocessing
    pub fn get_preprocessing_info(&self) -> String {
        let mut results = json::JsonValue::new_array();

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
    fn build_opts(&self) -> BuildOpts<'static> {
        BuildOpts {
            fixed_min_qual: self.fixed_min_qual,
            min_contig_length: self.min_contig_length,
            pop_ratio: self.bubble_pop_ratio,
            tip_length_kmult: self.tip_length_kmult,
            ..BuildOpts::with_defaults(self.quality, self.do_bloom, self.do_fit)
        }
    }
}

#[cfg(target_family = "wasm")]
impl CountedK {
    /// The spectrum, the number of k-mers kept, and the cutoff, for the page.
    fn summary(&self) -> (&[u32], usize, u16) {
        match self {
            Self::W64(c) => (&c.histovec, c.kmers.len(), c.used_min_count),
            Self::W128(c) => (&c.histovec, c.kmers.len(), c.used_min_count),
            Self::W256(c) => (&c.histovec, c.kmers.len(), c.used_min_count),
            Self::W512(c) => (&c.histovec, c.kmers.len(), c.used_min_count),
        }
    }
}

//
//
//
//
// ===================================================================
// ============================================================= TESTS
// ==================================================================
#[cfg(test)]
mod multik_tests {
    use super::{
        active_quality_floors, aun, count_step, lowered_floor_index, quality_floors, read_store,
        run_ladder, settle_ladder, BuildOpts, LadderState, QualOpts, Replay,
    };
    use crate::cli::{DEFAULT_MINCOUNT, DEFAULT_MINQUAL};
    use crate::clock::Instant;

    #[test]
    fn aun_weights_lengths_over_the_output_filter() {
        let contigs = vec![vec![b'A'; 600], vec![b'A'; 400], vec![b'A'; 1000]];
        let want = (600.0 * 600.0 + 1000.0 * 1000.0) / 1600.0;
        assert!((aun(&contigs, 500) - want).abs() < 1e-9);
        assert_eq!(aun(&contigs, 2000), 0.0);
    }

    fn state_with(floors: &[u8]) -> LadderState {
        LadderState {
            ks: vec![21],
            default_ladder: true,
            floors: floors.to_vec(),
            replay: Replay::Reread(read_store::ReadStats::default()),
            floor_index: floors.len() - 1,
            last_choice: None,
            quality: QualOpts {
                min_count: 0,
                min_qual: floors[floors.len() - 1],
            },
            contigs: Vec::new(),
        }
    }

    /// The ceiling for later k falls only once two consecutive k chose below it, and then to the higher of
    /// the two choices.
    #[test]
    fn the_floor_ceiling_needs_two_consecutive_choices_below_it() {
        let floors = [0u8, 14, 21];
        // (ceiling index, previous choice, this choice, new ceiling index)
        for (ceiling, previous, chosen, want) in [
            (2, None, 14, 2),
            (2, Some(21), 14, 2),
            (2, Some(14), 21, 2),
            (2, Some(14), 14, 1),
            (2, Some(14), 0, 1),
            (2, Some(0), 14, 1),
            (1, Some(0), 0, 0),
        ] {
            assert_eq!(
                lowered_floor_index(&floors, ceiling, previous, chosen),
                want,
                "{previous:?} then {chosen} under index {ceiling}"
            );
        }
        assert_eq!(active_quality_floors(&floors, 2), &[0, 14, 21]);
        assert_eq!(active_quality_floors(&floors, 1), &[0, 14]);
    }

    #[test]
    fn an_unresolved_k_neither_lowers_the_ceiling_nor_breaks_a_pair() {
        let mut state = state_with(&[0, 14, 21]);
        state.record_floor_choice(41, 14, false);
        assert_eq!((state.floor_index, state.quality.min_qual), (2, 21));
        state.record_floor_choice(55, 0, true);
        assert_eq!((state.floor_index, state.last_choice), (2, Some(14)));
        state.record_floor_choice(71, 14, false);
        assert_eq!((state.floor_index, state.quality.min_qual), (1, 14));
    }

    #[test]
    fn default_ladder_stops_ten_bases_under_the_longest_read() {
        assert_eq!(
            settle_ladder(&[21], true, 143.6, 151),
            vec![21, 31, 41, 55, 71, 91, 111, 131, 141]
        );
        assert_eq!(settle_ladder(&[21], true, 22.2, 22), vec![21]);
    }

    #[test]
    fn explicit_ladder_keeps_its_existing_longest_read_filter() {
        assert_eq!(settle_ladder(&[21, 31, 55], false, 0.0, 55), vec![21, 31]);
    }

    /// The browser reads its input again for every k, while native packs it once; both must assemble
    /// the same contigs.
    #[test]
    fn rereading_the_input_assembles_what_the_store_does() {
        let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
        let files = vec![(
            "window".to_owned(),
            ["ecoli_window_1.fastq.gz", "ecoli_window_2.fastq.gz"]
                .map(|f| data.join(f).to_string_lossy().into_owned())
                .to_vec(),
        )];
        let reads = crate::reads::InputFiles(&files);
        let opts = BuildOpts::with_defaults(
            QualOpts {
                min_count: DEFAULT_MINCOUNT,
                min_qual: DEFAULT_MINQUAL,
            },
            true,
            true,
        );
        let run = |in_memory| {
            let mut timevec = vec![Instant::now()];
            let mut ladder = LadderState::new(&opts, &reads, None, in_memory);
            let first = count_step(&opts, &reads, 0, &mut ladder, &mut timevec).unwrap();
            run_ladder(&opts, &reads, &mut ladder, first, &mut timevec, None).unwrap()
        };
        assert_eq!(run(false), run(true));
    }

    /// A given floor is the only one counted at, so it can never be loosened; without one, the
    /// fixture's qualities give a ladder topped by the default.
    #[test]
    fn a_fixed_floor_is_the_only_floor() {
        let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
        let files = vec![(
            "window".to_owned(),
            vec![data
                .join("ecoli_window_1.fastq.gz")
                .to_string_lossy()
                .into_owned()],
        )];
        let reads = crate::reads::InputFiles(&files);
        let mut opts = BuildOpts::with_defaults(
            QualOpts {
                min_count: DEFAULT_MINCOUNT,
                min_qual: 25,
            },
            true,
            true,
        );
        assert!(quality_floors(&opts, &reads).len() > 1);
        opts.fixed_min_qual = true;
        assert_eq!(quality_floors(&opts, &reads), vec![25]);
    }
}
