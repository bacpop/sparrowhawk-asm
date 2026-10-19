//! Shared single-k and multi-k preprocessing, recounting, assembly, and ladder state.

use crate::cli::*;
use crate::graph_works::{self, Assemble};
use crate::preprocessing::{
    self,
    parsercounter::PassInput,
    read_store::{self, PackedSeqs},
};
use crate::utils::{
    bit_encoding::{UInt, U256, U512},
    clock::Instant,
    io_utils, qual_profile,
    reads::{self, ReadSource},
};
use crate::{post_state, QualOpts};
use sparrowhawk_graph::DbgGraph;
use std::path::{Path, PathBuf};

/// A k after the first with a single-copy peak below this, fitted or fallback, is not assembled: the
/// ladder stops on the previous contigs. At 3, a 13x simulation gained a misassembly.
const MIN_LADDER_PEAK: u32 = 4;

/// Everything a build needs besides its reads, on both targets. The browser writes no files, so it
/// leaves the output fields empty and `no_histo` set.
pub(crate) struct BuildOpts<'a> {
    pub(crate) quality: QualOpts,
    /// Count at `quality.min_qual` alone, rather than at the reads' ladder of floors topped by it.
    pub(crate) fixed_min_qual: bool,
    /// Native multi-k: whole reads in two bits, with Q0 at every k.
    pub(crate) two_bit_reads: bool,
    /// Enabled by default
    pub(crate) do_bloom: bool,
    /// Fit `min_count` from the k-mer spectrum rather than using `quality.min_count`.
    pub(crate) do_fit: bool,
    pub(crate) do_bubble_collapse: bool,
    pub(crate) do_dead_end_removal: bool,
    /// Fraction of the stronger branch's coverage below which the weaker branch of a bubble is popped.
    pub(crate) pop_ratio: f32,
    /// Fraction of single-copy coverage below which a branch is an error. `None` keeps the default
    /// for however the coverage was established.
    pub(crate) peak_ratio: Option<f32>,
    /// Flat dead-end threshold in bases, and the k multiple that may raise it; resolved per k.
    pub(crate) dead_end_length: usize,
    pub(crate) dead_end_length_kmult: f32,
    /// k multiple bounding dead-end removal by coverage; `None` under `--no-dead-end-coverage`.
    pub(crate) dead_end_coverage_kmult: Option<f32>,
    /// How many times better covered a junction must be than the dead end hanging off it.
    pub(crate) dead_end_coverage_ratio: f64,
    pub(crate) do_ec_removal: bool,
    pub(crate) ec_ratio: f64,
    pub(crate) ec_require_both_flanks: bool,
    /// Minimum contig sequence length written to FASTA.
    pub(crate) min_contig_length: usize,
    /// Multi-k: also write each intermediate k's contigs.
    pub(crate) keep_intermediate_contigs: bool,
    pub(crate) no_histo: bool,
    pub(crate) debug_quality_floors: bool,
    pub(crate) output_dir: &'a str,
    pub(crate) output_prefix: &'a str,
}

impl BuildOpts<'_> {
    /// Correction settings for one k: dead-end thresholds scale with k, coverage comes from its spectrum. Small helper that gets the info from buildopts
    fn correction(
        &self,
        k: usize,
        peak: preprocessing::PeakSource,
        min_count: u16,
    ) -> graph_works::corrector::CorrectionOpts {
        use graph_works::corrector::{dead_end_length_nts, CorrectionOpts, CoverageRef};
        CorrectionOpts {
            do_bubble_collapse: self.do_bubble_collapse,
            do_dead_end_removal: self.do_dead_end_removal,
            pop_ratio: self.pop_ratio,
            dead_end_nts: dead_end_length_nts(self.dead_end_length, self.dead_end_length_kmult, k),
            // Zero empties the band: `.max(limit)` in `remove_dead_paths`.
            dead_end_coverage_nts: self
                .dead_end_coverage_kmult
                .map_or(0, |m| dead_end_length_nts(self.dead_end_length, m, k)),
            dead_end_coverage_ratio: self.dead_end_coverage_ratio,
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
    pub(crate) fn with_defaults(quality: QualOpts, do_bloom: bool, do_fit: bool) -> Self {
        use graph_works::corrector::{DEFAULT_EC_COVERAGE_RATIO, DEFAULT_POP_RATIO};
        Self {
            quality,
            fixed_min_qual: false,
            two_bit_reads: false,
            do_bloom,
            do_fit,
            do_bubble_collapse: true,
            do_dead_end_removal: true,
            pop_ratio: DEFAULT_POP_RATIO,
            peak_ratio: None,
            dead_end_length: DEFAULT_DEAD_END_LEN_NTS,
            dead_end_length_kmult: DEFAULT_DEAD_END_LEN_KMULT,
            dead_end_coverage_kmult: Some(DEFAULT_DEAD_END_COVERAGE_KMULT),
            dead_end_coverage_ratio: DEFAULT_DEAD_END_COVERAGE_RATIO,
            do_ec_removal: true,
            ec_ratio: DEFAULT_EC_COVERAGE_RATIO,
            ec_require_both_flanks: false,
            min_contig_length: DEFAULT_MIN_CONTIG_LENGTH_NTS,
            keep_intermediate_contigs: false,
            no_histo: true,
            debug_quality_floors: false,
            output_dir: "",
            output_prefix: "",
        }
    }
}

// Preprocessing, one k at a time

/// One k's preprocessing, at the packed width its k needs.
pub(crate) enum Preprocessed {
    W64(preprocessing::PreprocessedK<u64>),
    W128(preprocessing::PreprocessedK<u128>),
    W256(preprocessing::PreprocessedK<U256>),
    W512(preprocessing::PreprocessedK<U512>),
}

/// The base-quality floors a k is counted at, ascending: the given floor alone when it is fixed, which
/// can never be loosened, or else the ladder the reads' qualities allow, topped by it.
fn quality_floors<S: ReadSource>(opts: &BuildOpts, reads: &S) -> Vec<u8> {
    let floors = if opts.two_bit_reads {
        vec![0]
    } else if opts.fixed_min_qual {
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

/// One pass of the preprocessing (parse, filter, count, choose the cutoff, filter by it) of one k.
/// Carry `carried` in once the reads set the cutoff. `prior` is the first pass's selection on a recount;
/// see `keep_refit_or_prior`.
#[allow(clippy::too_many_arguments)]
fn preprocess_pass<IntT, I>(
    opts: &BuildOpts,
    k: usize,
    quality: &QualOpts,
    floors: Option<&[u8]>,
    input: PassInput<'_, I>,
    carried: Option<&PackedSeqs>,
    prior: Option<(u16, preprocessing::PeakSource)>,
    timevec: &mut Vec<Instant>,
    out_path_histo: &mut Option<PathBuf>,
) -> Result<preprocessing::PreprocessedK<IntT>, preprocessing::PreprocessingError>
where
    IntT: for<'a> UInt<'a>,
    I: Iterator<Item = reads::OwnedRecord> + reads::MaybeSend,
{
    preprocessing::preprocessing::<IntT, _>(
        input,
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

/// Count ladder step `step` at the packed width its k needs.
pub(crate) fn preprocess_step<S: ReadSource>(
    opts: &BuildOpts,
    reads: &S,
    step: usize,
    state: &mut LadderState,
    timevec: &mut Vec<Instant>,
) -> Result<Preprocessed, preprocessing::PreprocessingError> {
    let k = state.ks[step];
    if state.multik {
        // The default ladder is settled from the read lengths once its first k has been counted.
        if state.default_ladder && step == 0 {
            log::info!("Multi-k step 1: k={k} (the rest of the ladder is set after it)");
        } else {
            log::info!("Multi-k step {} of {}: k={k}", step + 1, state.ks.len());
        }
        post_state(&format!("multik:step:{}:{k}", step + 1));
    }
    let bits = match k {
        3..=32 => 64,
        33..=64 => 128,
        65..=128 => 256,
        _ => 512,
    };
    log::info!("k={k}: using {bits}-bit representation");
    match k {
        3..=32 => {
            preprocess_step_at::<u64, S>(opts, reads, step, state, timevec).map(Preprocessed::W64)
        }
        33..=64 => {
            preprocess_step_at::<u128, S>(opts, reads, step, state, timevec).map(Preprocessed::W128)
        }
        65..=128 => {
            preprocess_step_at::<U256, S>(opts, reads, step, state, timevec).map(Preprocessed::W256)
        }
        _ => {
            preprocess_step_at::<U512, S>(opts, reads, step, state, timevec).map(Preprocessed::W512)
        }
    }
}

/// The first k reads the input, packing or measuring it, and settles the ladder; a later k counts from
/// the store or reads the input again. Then the floor is settled, recounting if it loosened.
fn preprocess_step_at<IntT, S>(
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
    let tag = if state.multik {
        format!("_k{k}")
    } else {
        String::new()
    };
    let mut out_path_histo = opts.histo_path(&tag);
    let active_floors = state.floors[..=state.floor_index].to_vec();
    let strict_floor = state.floors[state.floor_index];
    let quality = state.quality;
    let first = if step == 0 {
        let floors = Some(active_floors.as_slice());
        let pass1 = match &mut state.replay {
            Replay::Store(store) => preprocess_pass::<IntT, _>(
                opts,
                k,
                &quality,
                floors,
                PassInput::Text(read_store::Tee::new(reads.records(), &mut **store)),
                None,
                None,
                timevec,
                &mut out_path_histo,
            )?,
            Replay::Reread(stats) => preprocess_pass::<IntT, _>(
                opts,
                k,
                &quality,
                floors,
                PassInput::Text(read_store::Tee::new(reads.records(), stats)),
                None,
                None,
                timevec,
                &mut out_path_histo,
            )?,
        };
        if let Replay::Store(store) = &mut state.replay {
            store.finish();
        }
        if state.multik {
            let stats = state.replay.stats();
            state.ks = settle_ladder(
                &state.ks,
                state.default_ladder,
                stats.average_read_len().unwrap_or(0.0),
                stats.max_read_len(),
            );
            post_ladder(&state.ks);
        }
        pass1
    } else {
        preprocess_again::<IntT, S>(
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
    let assembly = settle_floor::<IntT, S>(
        opts,
        reads,
        k,
        state,
        first,
        strict_floor,
        timevec,
        &mut out_path_histo,
    )?;
    if step > 0
        && assembly
            .genomic_peak
            .value()
            .is_none_or(|peak| peak < MIN_LADDER_PEAK)
    {
        return Err(preprocessing::PreprocessingError::peak_too_low(
            k,
            assembly.genomic_peak,
        ));
    }
    Ok(assembly)
}

/// Count k again, from the store or by reading the input again, carrying the previous contigs in.
#[allow(clippy::too_many_arguments)]
fn preprocess_again<IntT, S>(
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
    let carried = (!state.contigs.is_empty()).then_some(&state.contigs);
    let input = match &state.replay {
        Replay::Store(store) => PassInput::Stored(store),
        Replay::Reread(_) => PassInput::Text(reads.records()),
    };
    preprocess_pass::<IntT, _>(
        opts,
        k,
        quality,
        floors,
        input,
        carried,
        prior,
        timevec,
        out_path_histo,
    )
}

/// Record the floor this k's first count chose and, if it is below `strict_floor`, recount there,
/// refitting the whole spectrum; the sketch's cutoff and peak stand only if the refit loses the
/// separation. The first count is dropped before the recount allocates, so both tables are never
/// resident at once.
#[allow(clippy::too_many_arguments)]
fn settle_floor<IntT, S>(
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
    log::warn!(
        "Recounting k={k} at a base-quality floor of {floor} instead of {strict_floor}: this admits \
         more error k-mers."
    );
    post_state(&format!("preprocess:recount:{floor}"));
    drop(first);
    let quality = QualOpts {
        min_count,
        min_qual: floor,
    };
    preprocess_again::<IntT, S>(
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

// Assembling one k

/// Assemble ladder step `step` at its width.
fn assemble_step(
    opts: &BuildOpts,
    step: usize,
    state: &mut LadderState,
    counted: Preprocessed,
    timevec: &mut Vec<Instant>,
    graph_sink: &mut Option<&mut dyn FnMut(&DbgGraph)>,
) -> Result<(), preprocessing::PreprocessingError> {
    match counted {
        Preprocessed::W64(c) => assemble_step_at(opts, step, state, c, timevec, graph_sink),
        Preprocessed::W128(c) => assemble_step_at(opts, step, state, c, timevec, graph_sink),
        Preprocessed::W256(c) => assemble_step_at(opts, step, state, c, timevec, graph_sink),
        Preprocessed::W512(c) => assemble_step_at(opts, step, state, c, timevec, graph_sink),
    }
}

/// Assemble one step and keep its contigs for the next k. The graph goes to `graph_sink` only at the
/// last k, and only when its contigs are sure to be kept.
fn assemble_step_at<IntT: for<'a> UInt<'a>>(
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
    let previous = unresolved.then(|| std::mem::replace(&mut state.contigs, PackedSeqs::new(true)));
    state.contigs = PackedSeqs::new(true);

    let last = step + 1 == state.ks.len();
    // A k that may still be rejected sends no graph, as a stopped ladder never has.
    let sink = if last && !unresolved {
        graph_sink.take()
    } else {
        None
    };
    let correction = opts.correction(k, assembly.genomic_peak, assembly.used_min_count);
    let contigs = graph_works::BasicAsm::assemble::<IntT>(
        k,
        &mut assembly.kmers,
        &mut Some(timevec),
        sink,
        correction,
    );
    state.contigs = crate::graph_works::spelling::spell_contigs::<IntT>(
        &contigs,
        |hash| assembly.kmers.get_packed(hash).copied(),
        k,
    );
    log::info!(
        "k={k}: {} contigs, {} bases",
        state.contigs.len(),
        state.contigs.iter().map(|s| s.len()).sum::<usize>()
    );
    if !last && opts.keep_intermediate_contigs {
        let path =
            Path::new(opts.output_dir).join(format!("{}_k{k}_contigs.fasta", opts.output_prefix));
        io_utils::save_sequences_as_fasta(&state.contigs, opts.min_contig_length, path);
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

// The ladder

/// How a run reads its input again for a recount or a later k.
enum Replay {
    /// Every read, packed during the first k's pass and replayed by the rest (native).
    Store(Box<read_store::ReadStore>),
    /// The input itself, read again for every pass: browser runs and native single-k runs.
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

/// What a run carries from one k to the next. A single k is a ladder of one step.
pub(crate) struct LadderState {
    /// Whether this is a multi-k run. Single-k uses the same flow without multi-k progress or names.
    multik: bool,
    /// The k values to run; a default ladder is settled once the first k has seen every read.
    pub(crate) ks: Vec<usize>,
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
    contigs: PackedSeqs,
}

impl LadderState {
    /// A ladder over `reads`: the `requested` k values, or the default ladder, settled after the first
    /// k. `in_memory` packs the reads during the first k of multi-k; otherwise passes read them again.
    pub(crate) fn new<S: ReadSource>(
        opts: &BuildOpts,
        reads: &S,
        requested: Option<&[usize]>,
        in_memory: bool,
    ) -> Self {
        let floors = quality_floors(opts, reads);
        let multik = requested.is_none_or(|ks| ks.len() > 1);
        if let Some(ks) = requested.filter(|_| multik) {
            post_ladder(ks);
        }
        let replay = if in_memory && multik {
            Replay::Store(Box::new(read_store::ReadStore::new(
                &floors,
                opts.two_bit_reads,
            )))
        } else {
            Replay::Reread(read_store::ReadStats::default())
        };
        Self {
            multik,
            ks: requested.map_or_else(|| vec![DEFAULT_KMER_LADDER[0]], <[usize]>::to_vec),
            default_ladder: requested.is_none(),
            floor_index: floors.len() - 1,
            last_choice: None,
            quality: QualOpts {
                min_count: opts.quality.min_count,
                // The strict floor, as `record_floor_choice` keeps it: with `[0]`, no quality filtering.
                min_qual: floors[floors.len() - 1],
            },
            contigs: PackedSeqs::new(true),
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

/// Length-weighted mean length of the contigs of at least `min_len` bases; 0 when there are none.
fn aun(contigs: &PackedSeqs, min_len: usize) -> f64 {
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

/// Assemble the ladder from its first k, already counted, to its last; each later k is counted with
/// the previous contigs carried in. A k that fails ends the ladder on the previous contigs.
pub(crate) fn run_ladder<S: ReadSource>(
    opts: &BuildOpts,
    reads: &S,
    state: &mut LadderState,
    first: Preprocessed,
    timevec: &mut Vec<Instant>,
    mut graph_sink: Option<&mut dyn FnMut(&DbgGraph)>,
) -> Result<PackedSeqs, preprocessing::PreprocessingError> {
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
        counted = preprocess_step(opts, reads, step, state, timevec);
    }
    Ok(std::mem::replace(&mut state.contigs, PackedSeqs::new(true)))
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

/// Tells the page every k of a multi-k ladder, so it can say how far along the ladder a run is.
fn post_ladder(ks: &[usize]) {
    let ks: Vec<String> = ks.iter().map(usize::to_string).collect();
    post_state(&format!("multik:ladder:{}", ks.join(",")));
}

#[cfg(target_family = "wasm")]
impl Preprocessed {
    /// The spectrum, the number of k-mers kept, and the cutoff, for the page.
    pub(crate) fn summary(&self) -> (&[u32], usize, u16) {
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
        aun, lowered_floor_index, preprocess_step, quality_floors, read_store, run_ladder,
        settle_ladder, BuildOpts, LadderState, QualOpts, Replay,
    };
    use crate::cli::{DEFAULT_MINCOUNT, DEFAULT_MINQUAL};
    use crate::preprocessing::read_store::PackedSeqs;
    use crate::utils::clock::Instant;

    #[test]
    fn aun_weights_lengths_over_the_output_filter() {
        let mut contigs = PackedSeqs::new(true);
        for len in [600, 400, 1000] {
            contigs.push_contig(&vec![0; len]);
        }
        contigs.finish();
        let want = (600.0 * 600.0 + 1000.0 * 1000.0) / 1600.0;
        assert!((aun(&contigs, 500) - want).abs() < 1e-9);
        assert_eq!(aun(&contigs, 2000), 0.0);
    }

    fn state_with(floors: &[u8]) -> LadderState {
        LadderState {
            multik: true,
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
            contigs: PackedSeqs::new(true),
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
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn rereading_the_input_assembles_what_the_store_does() {
        let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
        let files = vec![(
            "window".to_owned(),
            ["ecoli_window_1.fastq.gz", "ecoli_window_2.fastq.gz"]
                .map(|f| data.join(f).to_string_lossy().into_owned())
                .to_vec(),
        )];
        let reads = crate::utils::reads::InputFiles(&files);
        let mut opts = BuildOpts::with_defaults(
            QualOpts {
                min_count: DEFAULT_MINCOUNT,
                min_qual: DEFAULT_MINQUAL,
            },
            true,
            true,
        );
        let run = |opts: &BuildOpts, in_memory| {
            let mut timevec = vec![Instant::now()];
            let mut ladder = LadderState::new(opts, &reads, None, in_memory);
            let first = preprocess_step(opts, &reads, 0, &mut ladder, &mut timevec).unwrap();
            run_ladder(opts, &reads, &mut ladder, first, &mut timevec, None).unwrap()
        };
        // The automatic ladder, and a fixed floor, whose 4-bit store holds a single class.
        for fixed in [None, Some(20)] {
            if let Some(floor) = fixed {
                opts.quality.min_qual = floor;
                opts.fixed_min_qual = true;
            }
            assert_eq!(
                run(&opts, false)
                    .iter()
                    .map(|s| s.ascii())
                    .collect::<Vec<_>>(),
                run(&opts, true)
                    .iter()
                    .map(|s| s.ascii())
                    .collect::<Vec<_>>(),
                "fixed floor {fixed:?}"
            );
        }
    }

    /// A given floor is the only one counted at, so it can never be loosened; without one, the
    /// fixture's qualities give a ladder topped by the default.
    #[cfg(not(target_family = "wasm"))]
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
        let reads = crate::utils::reads::InputFiles(&files);
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
        let ladder = LadderState::new(&opts, &reads, Some(&[21, 31]), true);
        assert_eq!(ladder.floors, vec![25]);
        assert!(matches!(ladder.replay, Replay::Store(_)));
        let single = LadderState::new(&opts, &reads, Some(&[31]), true);
        assert!(!single.multik);
        assert!(matches!(single.replay, Replay::Reread(_)));
    }
    #[test]
    fn two_bit_reads_use_q0_from_the_first_k_and_never_reread() {
        use crate::utils::reads::{OwnedRecord, ReadSource};
        use std::cell::Cell;
        struct Reads {
            records: Vec<OwnedRecord>,
            opens: Cell<usize>,
        }
        impl ReadSource for Reads {
            type Records<'a> = std::vec::IntoIter<OwnedRecord>;
            fn records(&self) -> Self::Records<'_> {
                self.opens.set(self.opens.get() + 1);
                self.records.clone().into_iter()
            }
        }
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let genome: Vec<_> = (0..2000)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                b"ACGT"[((seed >> 33) & 3) as usize]
            })
            .collect();
        let records: Vec<_> = (0..=1850)
            .step_by(5)
            .flat_map(|i| {
                let seq = genome[i..i + 150].to_vec();
                let qual = (0..150)
                    .map(|j| if j % 7 == 0 { 35 } else { 70 })
                    .collect::<Vec<_>>();
                [(seq.clone(), Some(qual.clone())), (seq, Some(qual))]
            })
            .collect();
        let run = |two_bit| {
            let reads = Reads {
                records: records.clone(),
                opens: Cell::new(0),
            };
            let mut opts = BuildOpts::with_defaults(
                QualOpts {
                    min_count: 2,
                    min_qual: 0,
                },
                true,
                true,
            );
            opts.two_bit_reads = two_bit;
            opts.fixed_min_qual = !two_bit;
            assert_eq!(quality_floors(&opts, &reads), [0]);
            assert_eq!(reads.opens.get(), 0, "Q0 must not peek at input qualities");
            let mut state = LadderState::new(&opts, &reads, Some(&[21, 31]), true);
            assert_eq!(state.floors, [0]);
            let mut times = vec![Instant::now()];
            let first = preprocess_step(&opts, &reads, 0, &mut state, &mut times).unwrap();
            match &first {
                super::Preprocessed::W64(c) => assert_eq!(c.chosen_min_qual, 0),
                _ => panic!("k=21 must use u64"),
            }
            let result = run_ladder(&opts, &reads, &mut state, first, &mut times, None).unwrap();
            assert_eq!(
                reads.opens.get(),
                1,
                "packed replay must not reopen input or recount"
            );
            assert_eq!(state.floors, [0]);
            assert_eq!(state.floor_index, 0);
            assert_eq!(state.quality.min_qual, 0);
            result.iter().map(|s| s.ascii()).collect::<Vec<_>>()
        };
        let expected = run(false);
        assert!(!expected.is_empty());
        assert_eq!(run(true), expected);
    }
}
