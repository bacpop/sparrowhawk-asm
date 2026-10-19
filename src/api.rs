//! Complete assembly without CLI global initialisation or automatic output files.

use crate::{
    cli::*,
    graph_works::corrector::{CorrectionOpts, DEFAULT_EC_COVERAGE_RATIO, DEFAULT_POP_RATIO},
    pipeline::{preprocess_step, run_ladder, BuildOpts, LadderState},
    preprocessing::{
        read_store::{PackedSeq, PackedSeqs},
        PeakSource,
    },
    utils::{bit_encoding::UInt, clock::Instant},
    AssemblyError, DbgGraph, QualOpts, ReadSource,
};
use std::io;

/// Settings shared by the library, CLI, and browser assembly pipeline.
#[derive(Clone, Debug)]
pub struct AssemblyOptions {
    /// Empty selects the automatic ladder; one k selects single-k; several select an explicit ladder.
    pub k_values: Vec<usize>,
    /// Fixed minimum count, or `None` to fit it independently at each k.
    pub min_count: Option<u16>,
    /// Fixed numerical PHRED floor, or `None` for automatic quality-floor selection.
    pub min_qual: Option<u8>,
    /// Use memory-saving Bloom counting, with possible false positives.
    pub use_bloom: bool,
    /// Native multi-k only: store ACGT reads in two bits and use Q0 without quality recounts.
    pub two_bit_reads: bool,
    /// Inclusive minimum contig length for result inspection and FASTA output.
    pub min_contig_length: usize,
    /// Enable bubble and bulge correction.
    pub do_bubble_collapse: bool,
    /// Enable dead-end removal.
    pub do_dead_end_removal: bool,
    /// Weak-to-strong coverage fraction for popping bubbles, strictly between zero and one.
    pub pop_ratio: f32,
    /// Optional error fraction of single-copy coverage; otherwise use the provenance-specific default.
    pub peak_ratio: Option<f32>,
    /// Flat dead-end length threshold in bases.
    pub dead_end_length: usize,
    /// k multiplier that may raise the flat dead-end threshold.
    pub dead_end_length_kmult: f32,
    /// k multiplier for coverage-based dead-end removal; `None` disables it.
    pub dead_end_coverage_kmult: Option<f32>,
    /// Required junction-to-dead-end coverage ratio.
    pub dead_end_coverage_ratio: f64,
    /// Enable erroneous-connection removal.
    pub do_ec_removal: bool,
    /// Required flank-to-connector coverage ratio, greater than one.
    pub ec_ratio: f64,
    /// Require strong alternatives at both connector flanks.
    pub ec_require_both_flanks: bool,
}

impl Default for AssemblyOptions {
    fn default() -> Self {
        Self {
            k_values: Vec::new(),
            min_count: None,
            min_qual: None,
            use_bloom: true,
            two_bit_reads: false,
            min_contig_length: DEFAULT_MIN_CONTIG_LENGTH_NTS,
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
        }
    }
}

impl AssemblyOptions {
    /// Reject invalid settings before opening input or allocating a counting pass.
    pub fn validate(&self) -> Result<(), AssemblyError> {
        let invalid = |message: &str| AssemblyError::InvalidOptions(message.into());
        check_ladder(&self.k_values).map_err(AssemblyError::InvalidOptions)?;
        validate_bloom_min_count(self.use_bloom, self.min_count).map_err(invalid)?;
        if self.two_bit_reads {
            if cfg!(target_family = "wasm") {
                return Err(invalid("--two-bit-reads is available only natively"));
            }
            if self.k_values.len() == 1 {
                return Err(invalid("--two-bit-reads requires multi-k mode"));
            }
            if self.min_qual.is_some_and(|q| q > 0) {
                return Err(invalid("--two-bit-reads requires Q0"));
            }
        }
        if !(self.pop_ratio > 0.0 && self.pop_ratio < 1.0) {
            return Err(invalid(
                "--bubble-pop-ratio must be strictly between 0 and 1",
            ));
        }
        if self
            .peak_ratio
            .is_some_and(|ratio| !(ratio > 0.0 && ratio < 1.0))
        {
            return Err(invalid(
                "--bubble-peak-ratio must be strictly between 0 and 1",
            ));
        }
        if !self.dead_end_length_kmult.is_finite()
            || self.dead_end_length_kmult < 0.0
            || self
                .dead_end_coverage_kmult
                .is_some_and(|m| !m.is_finite() || m < 0.0)
        {
            return Err(invalid(
                "dead-end length multipliers must be finite and non-negative",
            ));
        }
        if !self.dead_end_coverage_ratio.is_finite() || self.dead_end_coverage_ratio <= 0.0 {
            return Err(invalid(
                "--dead-end-coverage-ratio must be finite and greater than 0",
            ));
        }
        if !self.ec_ratio.is_finite() || self.ec_ratio <= 1.0 {
            return Err(invalid(
                "--ec-coverage-ratio must be finite and greater than 1",
            ));
        }
        Ok(())
    }

    /// Resolve the same per-k correction settings used by the complete pipeline.
    pub fn correction_for(
        &self,
        k: usize,
        peak: PeakSource,
        min_count: u16,
    ) -> Result<CorrectionOpts, AssemblyError> {
        self.validate()?;
        check_k(k).map_err(AssemblyError::InvalidOptions)?;
        Ok(BuildOpts::from(self).correction(k, peak, min_count))
    }
}

pub(crate) fn validate_packed_k<IntT: for<'a> UInt<'a>>(k: usize) -> Result<(), AssemblyError> {
    check_k(k).map_err(AssemblyError::InvalidOptions)?;
    let width = IntT::zero_init().n_bits();
    if 2 * k > width as usize {
        return Err(AssemblyError::InvalidOptions(format!(
            "k={k} needs {} bits, but the packed type has {width}",
            2 * k
        )));
    }
    Ok(())
}

pub(crate) fn validate_correction(correction: &CorrectionOpts) -> Result<(), AssemblyError> {
    AssemblyOptions {
        pop_ratio: correction.pop_ratio,
        peak_ratio: Some(correction.coverage.error_fraction),
        dead_end_coverage_ratio: correction.dead_end_coverage_ratio,
        ec_ratio: correction.ec_ratio,
        ..AssemblyOptions::default()
    }
    .validate()
}

impl<'a> From<&AssemblyOptions> for BuildOpts<'a> {
    fn from(o: &AssemblyOptions) -> Self {
        Self {
            quality: QualOpts {
                min_count: o.min_count.unwrap_or(DEFAULT_MINCOUNT),
                min_qual: if o.two_bit_reads {
                    0
                } else {
                    o.min_qual.unwrap_or(DEFAULT_MINQUAL)
                },
            },
            fixed_min_qual: o.min_qual.is_some() || o.two_bit_reads,
            two_bit_reads: o.two_bit_reads,
            do_bloom: o.use_bloom,
            do_fit: o.min_count.is_none(),
            do_bubble_collapse: o.do_bubble_collapse,
            do_dead_end_removal: o.do_dead_end_removal,
            pop_ratio: o.pop_ratio,
            peak_ratio: o.peak_ratio,
            dead_end_length: o.dead_end_length,
            dead_end_length_kmult: o.dead_end_length_kmult,
            dead_end_coverage_kmult: o.dead_end_coverage_kmult,
            dead_end_coverage_ratio: o.dead_end_coverage_ratio,
            do_ec_removal: o.do_ec_removal,
            ec_ratio: o.ec_ratio,
            ec_require_both_flanks: o.ec_require_both_flanks,
            min_contig_length: o.min_contig_length,
            keep_intermediate_contigs: false,
            no_histo: true,
            debug_quality_floors: false,
            output_dir: "",
            output_prefix: "",
        }
    }
}

// Internal shorthand; public signatures spell out the callback contract.
pub(crate) type GraphSink<'a> = &'a mut dyn FnMut(&DbgGraph) -> io::Result<()>;

/// Completed contigs stored in two bits per base, with no fully decoded assembly copy.
pub struct Assembly {
    sequences: PackedSeqs,
    min_contig_length: usize,
    completed_k_values: Vec<usize>,
    stopped_at: Option<(usize, String)>,
}

impl Assembly {
    /// Borrow contigs clearing the inclusive minimum-length filter.
    pub fn contigs(&self) -> impl Iterator<Item = PackedSeq<'_>> + '_ {
        self.sequences
            .iter()
            .filter(move |seq| seq.len() >= self.min_contig_length)
    }
    /// Number of contigs clearing the length filter.
    pub fn len(&self) -> usize {
        self.contigs().count()
    }
    /// Whether no contig clears the length filter.
    pub fn is_empty(&self) -> bool {
        self.contigs().next().is_none()
    }
    /// Accepted k values, excluding rejected or read-length-truncated steps.
    pub fn completed_k_values(&self) -> &[usize] {
        &self.completed_k_values
    }
    /// The rejected k and its biological reason, if the ladder stopped early.
    pub fn stopped_at(&self) -> Option<(usize, &str)> {
        self.stopped_at
            .as_ref()
            .map(|(k, reason)| (*k, reason.as_str()))
    }
    /// Write filtered FASTA using a bounded decoder; returns the number of records written.
    pub fn write_fasta<W: io::Write>(&self, writer: &mut W) -> io::Result<usize> {
        self.sequences.write_fasta(self.min_contig_length, writer)
    }
}

/// Run the shared assembly pipeline without initialising logging or a global thread pool.
pub fn assemble<S: ReadSource>(
    reads: &S,
    options: &AssemblyOptions,
) -> Result<Assembly, AssemblyError> {
    assemble_inner(reads, options, None)
}

/// Assemble with a borrowed corrected graph callback, before collapse at the final resolved k.
/// This allows you to e.g. save the graph before we collapse it, or check it out yourself
/// Biologically stopped ladders and unresolved final steps send no graph. Callback errors are fatal.
pub fn assemble_with_graph<S: ReadSource>(
    reads: &S,
    options: &AssemblyOptions,
    graph_sink: &mut dyn FnMut(&DbgGraph) -> io::Result<()>,
) -> Result<Assembly, AssemblyError> {
    assemble_inner(reads, options, Some(graph_sink))
}

fn assemble_inner<S: ReadSource>(
    reads: &S,
    options: &AssemblyOptions,
    graph_sink: Option<GraphSink<'_>>,
) -> Result<Assembly, AssemblyError> {
    options.validate()?;
    let opts = BuildOpts::from(options);
    let requested = (!options.k_values.is_empty()).then_some(options.k_values.as_slice());
    let mut times = vec![Instant::now()];
    let mut state = LadderState::new(&opts, reads, requested, !cfg!(target_family = "wasm"))?;
    let first = preprocess_step(&opts, reads, 0, &mut state, &mut times)?;
    let sequences = run_ladder(&opts, reads, &mut state, first, &mut times, graph_sink)?;
    Ok(Assembly {
        sequences,
        min_contig_length: options.min_contig_length,
        completed_k_values: state.ks[..state.completed_steps].to_vec(),
        stopped_at: state.stopped_at.take(),
    })
}

#[cfg(not(target_family = "wasm"))]
impl From<&crate::cli::Args> for AssemblyOptions {
    fn from(a: &crate::cli::Args) -> Self {
        Self {
            k_values: a.k.clone().unwrap_or_default(),
            min_count: a.min_count,
            min_qual: a.min_qual,
            use_bloom: !a.no_bloom,
            two_bit_reads: a.two_bit_reads,
            min_contig_length: a.min_contig_length,
            do_bubble_collapse: !a.no_bubble_collapse,
            do_dead_end_removal: !a.no_dead_end_removal,
            pop_ratio: a.bubble_pop_ratio,
            peak_ratio: a.bubble_peak_ratio,
            dead_end_length: a.dead_end_length,
            dead_end_length_kmult: a.dead_end_length_kmult,
            dead_end_coverage_kmult: (!a.no_dead_end_coverage).then_some(a.dead_end_coverage_kmult),
            dead_end_coverage_ratio: a.dead_end_coverage_ratio,
            do_ec_removal: !a.no_ec_removal,
            ec_ratio: a.ec_coverage_ratio,
            ec_require_both_flanks: a.ec_require_both_flanks,
        }
    }
}
