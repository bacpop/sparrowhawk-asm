//! Command line interface, built using [`crate::clap` with `Derive`](https://docs.rs/clap/latest/clap/_derive/_tutorial/index.html)
//!
//! The browser shares only the defaults and checks; the parser is native-only.
#[cfg(not(target_family = "wasm"))]
use clap::{ArgGroup, Parser};

/// Default k-mer size
pub const DEFAULT_KMER: usize = 31;
/// Default minimum k-mer count for FASTQ files
pub const DEFAULT_MINCOUNT: u16 = 5;
/// Default minimum base quality (PHRED score) for FASTQ files
pub const DEFAULT_MINQUAL: u8 = 20;
/// Default dead-end removal by length: dead ends shorter than this many bases are pruned
pub const DEFAULT_DEAD_END_LEN_NTS: usize = 100;
/// Default dead-end removal by length: the threshold also scales with k by this multiplier, taken from
/// Minia/SPAdes
pub const DEFAULT_DEAD_END_LEN_KMULT: f32 = 2.5;
/// Default dead-end removal by coverage: dead ends up to this many times k bases are also pruned when
/// the junction they hang off is much better covered
pub const DEFAULT_DEAD_END_COVERAGE_KMULT: f32 = 10.0;
/// Default dead-end removal by coverage: how many times a dead end's coverage its junction's other
/// neighbours must average, SPAdes/Minia-derived
pub const DEFAULT_DEAD_END_COVERAGE_RATIO: f64 = 2.0;
/// Default minimum contig sequence length written to FASTA, in nucleotides
pub const DEFAULT_MIN_CONTIG_LENGTH_NTS: usize = 500;
/// Default output directory
pub const DEFAULT_OUTPUT_DIR: &str = "./";
/// Default output prefix
pub const DEFAULT_OUTPUT_PREFIX: &str = "sphk";
/// Smallest minimum count supported by Bloom filtering and automatic Bloom fitting.
pub(crate) const MIN_BLOOM_COUNT: u16 = 2;
/// Fixed anchors for the automatic multi-k ladder.
pub const DEFAULT_KMER_LADDER: [usize; 5] = [21, 31, 41, 55, 71];
/// K increment used to extend the automatic ladder beyond its fixed anchors.
pub const KMER_LADDER_INCREMENT: usize = 20;
/// Largest k the packed k-mer types hold: 512 bits fit 256 bases, and k must be odd.
pub const MAX_K: usize = 255;
/// Smallest gain a clamped last rung must add over the previous one to be worth an assembly pass.
pub const MIN_CLAMPED_STEP: usize = 8;
/// Bases the automatic ladder's ceiling stays under the longest read, so a full-length read still
/// yields 11 k-mers at the last k.
pub const READ_LEN_MARGIN: usize = 10;

/// The automatic ladder, bounded by `max_k` (the longest read less [`READ_LEN_MARGIN`]) and by [`MAX_K`].
///
/// The step past the ceiling becomes the largest odd k under it if that adds [`MIN_CLAMPED_STEP`];
/// one past [`MAX_K`] always does. The initial 21 is retained as the bootstrap k.
pub fn default_kmer_ladder(max_k: usize) -> Vec<usize> {
    let ceiling = max_k.min(MAX_K);
    let mut ladder: Vec<usize> = DEFAULT_KMER_LADDER
        .into_iter()
        .filter(|&k| k <= ceiling)
        .collect();
    if ladder.is_empty() {
        ladder.push(DEFAULT_KMER_LADDER[0]);
    }

    let mut next = DEFAULT_KMER_LADDER[DEFAULT_KMER_LADDER.len() - 1] + KMER_LADDER_INCREMENT;
    while next <= ceiling {
        ladder.push(next);
        next += KMER_LADDER_INCREMENT;
    }
    let top = if ceiling.is_multiple_of(2) {
        ceiling.saturating_sub(1)
    } else {
        ceiling
    };
    let last = ladder[ladder.len() - 1];
    if top > last && (top == MAX_K || top - last >= MIN_CLAMPED_STEP) {
        ladder.push(top);
    }
    ladder
}

/// An odd k within 3..=MAX_K.
pub fn check_k(k: usize) -> Result<usize, String> {
    if !(3..=MAX_K).contains(&k) || k.is_multiple_of(2) {
        Err("K-mer must be an odd number between 3 and 255 (inclusive)".to_string())
    } else {
        Ok(k)
    }
}

/// Each k of a ladder is seeded with the previous k's contigs, so the ladder must climb.
pub fn check_ladder(ks: &[usize]) -> Result<(), String> {
    for &k in ks {
        check_k(k)?;
    }
    if ks.windows(2).any(|w| w[1] <= w[0]) {
        return Err(format!(
            "-k values must be strictly ascending (got {ks:?})."
        ));
    }
    Ok(())
}

#[doc(hidden)]
#[cfg(not(target_family = "wasm"))]
fn valid_kmer(s: &str) -> Result<usize, String> {
    let k: usize = s
        .parse()
        .map_err(|_| format!("`{s}` isn't a valid k-mer"))?;
    check_k(k)
}

#[doc(hidden)]
#[cfg(not(target_family = "wasm"))]
fn valid_cpus(s: &str) -> Result<usize, String> {
    let threads: usize = s
        .parse()
        .map_err(|_| format!("`{s}` isn't a valid number of cores"))?;
    if threads < 1 {
        Err("Threads must be one or higher".to_string())
    } else {
        Ok(threads)
    }
}

/// Reject Bloom-filter configurations whose low thresholds do not populate the count map.
pub(crate) fn validate_bloom_min_count(
    do_bloom: bool,
    explicit_min_count: Option<u16>,
) -> Result<(), &'static str> {
    if do_bloom && explicit_min_count.is_some_and(|min_count| min_count < MIN_BLOOM_COUNT) {
        Err(
            "Bloom counting does not support explicit --min-count 0 or 1; use --min-count >= 2, \
             omit --min-count for automatic fitting, or disable Bloom counting (native CLI: \
             --no-bloom)",
        )
    } else {
        Ok(())
    }
}

/// Prints a warning if more threads than available have been requested
#[cfg(not(target_family = "wasm"))]
pub fn check_threads(threads: usize) {
    let max_threads = std::thread::available_parallelism().map_or(1, usize::from);
    if threads > max_threads {
        log::warn!("{threads} threads is greater than available cores {max_threads}");
    }
}

/// Command-line options
#[cfg(not(target_family = "wasm"))]
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(group(
    ArgGroup::new("input")
        .required(true)
        .args(["seq_files", "file_list"]),
))]
pub struct Args {
    /// List of input FASTQ files
    #[arg(group = "input")]
    pub seq_files: Option<Vec<String>>,

    /// File listing input file (tab separated name TAB fastq1,fastq2)
    #[arg(short, group = "input")]
    pub file_list: Option<String>,

    /// Output directory
    #[arg(long, default_value_t = DEFAULT_OUTPUT_DIR.to_string())]
    pub output_dir: String,

    /// Output prefix
    #[arg(long, default_value_t = DEFAULT_OUTPUT_PREFIX.to_string())]
    pub output_prefix: String,

    /// K-mer size(s), comma-separated and strictly ascending: `-k 31`, or `-k 21,31,55,91` for an
    /// iterative multi-k assembly. If omitted, use 21,31,41,55,71, then add 20 at a time up to 10 bases
    /// under the longest read (at most 255), plus a last k clamped to that ceiling.
    #[arg(short, value_parser = valid_kmer, value_delimiter = ',')]
    pub k: Option<Vec<usize>>,

    /// Minimum k-mer count. If omitted, it is FITTED from the k-mer spectrum, separately for each k.
    #[arg(long)]
    pub min_count: Option<u16>,

    /// Minimum k-mer base quality, inclusive (with reads). If omitted, it is chosen automatically
    /// at each k from the reads' quality ladder, topped at 20; if given, it is used as is at every k.
    #[arg(long)]
    pub min_qual: Option<u8>,

    /// Native multi-k only: store whole reads in two bits and use Q0 from the first k onwards,
    /// without quality-floor recounts. Panics on non-ACGT bases, including N. Incompatible with a
    /// positive --min-qual, --debug-quality-floors, or a single explicit k.
    #[arg(long, default_value_t = false)]
    pub two_bit_reads: bool,

    /// Number of CPU threads
    #[arg(long, value_parser = valid_cpus, default_value_t = 1)]
    pub threads: usize,

    /// Disable the default Bloom counting and use exact k-mer counts. Bloom counting uses less memory
    /// and may be faster, but can add false-positive matches. Required for explicit --min-count 0 or 1.
    #[arg(long, default_value_t = false)]
    pub no_bloom: bool,

    /// DEPRECATED: now ignored, with warning.
    #[arg(long, default_value_t = 0)]
    pub chunk_size: usize,

    /// Fraction of the stronger branch's coverage below which the weaker branch of a bubble is
    /// treated as an error and popped.
    #[arg(long, default_value_t = crate::algorithms::corrector::DEFAULT_POP_RATIO)]
    pub bubble_pop_ratio: f32,

    /// Fraction of the fitted single-copy coverage below which a bubble branch is treated as an
    /// error, whatever its sibling carries. Ignored when no coverage could be established.
    #[arg(long)]
    pub bubble_peak_ratio: Option<f32>,

    /// Do not remove erroneous connections: short weakly-covered paths joining two branch points.
    #[arg(long, default_value_t = false)]
    pub no_ec_removal: bool,

    /// Erroneous-connection removal: a connector goes when both its flanking branches carry this
    /// many times its coverage.
    #[arg(long, default_value_t = crate::algorithms::corrector::DEFAULT_EC_COVERAGE_RATIO)]
    pub ec_coverage_ratio: f64,

    /// Erroneous-connection removal: require both flanks over the ratio rather than either.
    #[arg(long, default_value_t = false)]
    pub ec_require_both_flanks: bool,

    /// Dead-end removal by length: dead ends shorter than this many bases are pruned. The threshold
    /// actually used is `max(--dead-end-length, --dead-end-length-kmult * k)`.
    #[arg(long, default_value_t = DEFAULT_DEAD_END_LEN_NTS)]
    pub dead_end_length: usize,

    /// Dead-end removal by length: scale that threshold with k. Zero disables the scaling.
    #[arg(long, default_value_t = DEFAULT_DEAD_END_LEN_KMULT)]
    pub dead_end_length_kmult: f32,

    /// Dead-end removal by coverage: also prune dead ends up to `--dead-end-coverage-kmult * k` bases
    /// when the junction they hang off is much better covered. Zero disables it.
    #[arg(long, default_value_t = DEFAULT_DEAD_END_COVERAGE_KMULT)]
    pub dead_end_coverage_kmult: f32,

    /// Dead-end removal by coverage: a longer dead end goes when its junction's other neighbours
    /// average this many times its coverage.
    #[arg(long, default_value_t = DEFAULT_DEAD_END_COVERAGE_RATIO)]
    pub dead_end_coverage_ratio: f64,

    /// Disable dead-end removal by coverage: only remove dead ends by length.
    #[arg(long, default_value_t = false)]
    pub no_dead_end_coverage: bool,

    /// Minimum contig sequence length written to FASTA, in nucleotides. Contigs with exactly
    /// this length are retained.
    #[arg(long, default_value_t = DEFAULT_MIN_CONTIG_LENGTH_NTS)]
    pub min_contig_length: usize,

    /// Multi-k: also write each intermediate k's contigs as `<prefix>_k<k>_contigs.fasta`.
    #[arg(long, default_value_t = false)]
    pub keep_intermediate_contigs: bool,

    /// By default, Sparrowhawk draws the k-mer spectrum twice, linear and logarithmic (_log), each as
    /// PNG and SVG, beside the contigs. Use this argument to disable it.
    #[arg(long, default_value_t = false)]
    pub no_histo: bool,

    /// Evaluate and plot every quality floor for diagnosis. This does not alter floor selection;
    /// it overrides --no-histo and writes per-floor plots and spectrum TSVs.
    #[arg(long, default_value_t = false)]
    pub debug_quality_floors: bool,

    /// By default, Sparrowhawk will extract the graph just before collapse and save it in your output folder
    /// in the DOT, GFAv1.1 and GFAv2 formats. Use this argument if you want it to not do this
    #[arg(long, default_value_t = false)]
    pub no_graphs: bool,

    /// Do not solve bubbles in the graph
    #[arg(long, default_value_t = false)]
    pub no_bubble_collapse: bool,

    /// Do not remove dead ends in the graph
    #[arg(long, default_value_t = false)]
    pub no_dead_end_removal: bool,

    /// Show progress messages
    #[arg(short, long)]
    pub verbose: bool,
}

#[cfg(not(target_family = "wasm"))]
impl Args {
    /// Reject option values a run cannot use, before anything is created or logged.
    pub fn validate(&self) -> Result<(), String> {
        if self.two_bit_reads {
            if self.k.as_ref().is_some_and(|ks| ks.len() == 1) {
                return Err("--two-bit-reads requires multi-k mode; omit -k or provide at least two k values".into());
            }
            if self.min_qual.is_some_and(|q| q > 0) {
                return Err("--two-bit-reads requires Q0; remove --min-qual or set it to 0".into());
            }
            if self.debug_quality_floors {
                return Err(
                    "--two-bit-reads fixes Q0 and is incompatible with --debug-quality-floors"
                        .into(),
                );
            }
        }
        validate_bloom_min_count(!self.no_bloom, self.min_count)?;
        if self.debug_quality_floors && self.min_count.is_some() {
            return Err(
                "--debug-quality-floors requires automatic min-count selection; remove --min-count"
                    .into(),
            );
        }
        if self.debug_quality_floors && self.min_qual.is_some() {
            return Err(
                "--debug-quality-floors requires automatic base-quality floors; remove --min-qual"
                    .into(),
            );
        }
        if !(self.bubble_pop_ratio > 0.0 && self.bubble_pop_ratio < 1.0) {
            return Err(format!(
                "--bubble-pop-ratio must be strictly between 0 and 1 (got {}). It is the fraction of \
                 the stronger branch's coverage below which the weaker branch counts as an error; 1.0 \
                 or more pops every bubble, 0.0 or less pops none.",
                self.bubble_pop_ratio
            ));
        }
        if let Some(ratio) = self.bubble_peak_ratio.filter(|r| !(*r > 0.0 && *r < 1.0)) {
            return Err(format!(
                "--bubble-peak-ratio must be strictly between 0 and 1 (got {ratio}). It is the \
                 fraction of single-copy coverage below which a branch counts as an error."
            ));
        }
        if !(self.ec_coverage_ratio.is_finite() && self.ec_coverage_ratio > 1.0) {
            return Err(format!(
                "--ec-coverage-ratio must be finite and greater than 1 (got {}). It is how many times \
                 a connector's flanks must out-cover it before the connector is treated as erroneous.",
                self.ec_coverage_ratio
            ));
        }
        if self.dead_end_length_kmult < 0.0 {
            return Err(format!(
                "--dead-end-length-kmult must be >= 0 (got {}). Zero keeps the flat \
                 --dead-end-length.",
                self.dead_end_length_kmult
            ));
        }
        if self.dead_end_coverage_kmult < 0.0 {
            return Err(format!(
                "--dead-end-coverage-kmult must be >= 0 (got {}). Zero drops dead-end removal by \
                 coverage, leaving only the length rule.",
                self.dead_end_coverage_kmult
            ));
        }
        if !self.dead_end_coverage_ratio.is_finite() || self.dead_end_coverage_ratio <= 0.0 {
            return Err(format!(
                "--dead-end-coverage-ratio must be finite and > 0 (got {}). It is how many times \
                 better covered a junction must be than the dead end hanging off it.",
                self.dead_end_coverage_ratio
            ));
        }
        // Each k is seeded with the contigs of the k before it, so the ladder must climb.
        if let Some(ks) = &self.k {
            check_ladder(ks)?;
        }
        Ok(())
    }
}

/// Function to parse command line args into [`Args`] struct
#[cfg(not(target_family = "wasm"))]
pub fn cli_args() -> Args {
    Args::parse()
}

//
//
//
//
// ===================================================================
// ============================================================= TESTS
// ==================================================================
#[cfg(test)]
mod tests {
    use super::default_kmer_ladder;

    /// The default ladder: fixed anchors under the ceiling, then steps of 20, then a last rung clamped to
    /// the largest odd k under the ceiling when it adds at least 8 (always at MAX_K); 21 always stays.
    #[test]
    fn default_ladder_cases() {
        const ANCHORS: [usize; 5] = [21, 31, 41, 55, 71];
        let with = |tail: &[usize]| [&ANCHORS[..], tail].concat();
        // (ceiling, ladder)
        for (ceiling, want) in [
            (20, vec![21]),
            (60, vec![21, 31, 41, 55]),
            (65, vec![21, 31, 41, 55, 65]),
            (78, with(&[])),
            (90, with(&[89])),
            (135, with(&[91, 111, 131])),
            (138, with(&[91, 111, 131])),
            (141, with(&[91, 111, 131, 141])),
            (225, with(&[91, 111, 131, 151, 171, 191, 211, 225])),
            (240, with(&[91, 111, 131, 151, 171, 191, 211, 231, 239])),
            (254, with(&[91, 111, 131, 151, 171, 191, 211, 231, 251])),
            (
                270,
                with(&[91, 111, 131, 151, 171, 191, 211, 231, 251, 255]),
            ),
            (
                1000,
                with(&[91, 111, 131, 151, 171, 191, 211, 231, 251, 255]),
            ),
        ] {
            assert_eq!(default_kmer_ladder(ceiling), want, "ceiling {ceiling}");
        }
    }
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn two_bit_reads_require_multik_q0_without_quality_diagnostics() {
        use clap::Parser;
        let parse = |extra: &[&str]| {
            super::Args::try_parse_from(
                ["sparrowhawk-asm", "reads.fastq", "--two-bit-reads"]
                    .into_iter()
                    .chain(extra.iter().copied()),
            )
            .unwrap()
        };
        for extra in [&[][..], &["-k", "21,31"][..], &["--min-qual", "0"][..]] {
            assert!(parse(extra).validate().is_ok());
        }
        for (extra, message) in [
            (&["-k", "31"][..], "multi-k"),
            (&["--min-qual", "20"][..], "Q0"),
            (&["--debug-quality-floors"][..], "incompatible"),
        ] {
            assert!(parse(extra).validate().unwrap_err().contains(message));
        }
    }
}
