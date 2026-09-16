//! Command line interface, built using [`crate::clap` with `Derive`](https://docs.rs/clap/latest/clap/_derive/_tutorial/index.html)
use std::fmt;

use clap::{ArgGroup, Parser, Subcommand, ValueEnum};

/// Default k-mer size
pub const DEFAULT_KMER: usize = 31;
/// Default minimum k-mer count for FASTQ files
pub const DEFAULT_MINCOUNT: u16 = 5;
/// Default minimum base quality (PHRED score) for FASTQ files
pub const DEFAULT_MINQUAL: u8 = 20;
/// Default flat tip-removal threshold, in bases
pub const DEFAULT_TIP_LEN_NTS: usize = 100;
/// Default tip-removal k multiplier, matching Minia's `-tip-len-topo-kmult`
pub const DEFAULT_TIP_LEN_KMULT: f32 = 2.5;
/// Default minimum contig body length written to FASTA, in nucleotides
pub const DEFAULT_MIN_CONTIG_LENGTH_NTS: usize = 500;
/// Default output directory
pub const DEFAULT_OUTPUT_DIR: &str = "./";
/// Default output prefix
pub const DEFAULT_OUTPUT_PREFIX: &str = "sphk";
/// Smallest minimum count supported by Bloom filtering and automatic Bloom fitting.
pub(crate) const MIN_BLOOM_COUNT: u16 = 2;


#[doc(hidden)]
fn valid_kmer(s: &str) -> Result<usize, String> {
    let k: usize = s
        .parse()
        .map_err(|_| format!("`{s}` isn't a valid k-mer"))?;
    if !(3..=256).contains(&k) || k.is_multiple_of(2) {
        Err("K-mer must be an odd number between 3 and 255 (inclusive)".to_string())
    } else {
        Ok(k)
    }
}

#[doc(hidden)]
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
            "--do-bloom does not support --min-count 0 or 1; use --min-count >= 2, omit \
             --min-count to fit automatically, or remove --do-bloom",
        )
    } else {
        Ok(())
    }
}

/// Prints a warning if more threads than available have been requested
pub fn check_threads(threads: usize) {
    let max_threads = num_cpus::get();
    if threads > max_threads {
        log::warn!("{threads} threads is greater than available cores {max_threads}");
    }
}

/// Possible output file types
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum FileType {
    /// Variant call format
    Vcf,
    /// FASTA alignment
    Aln,
}

/// Possible variant filters
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum FilterType {
    /// Output all variants
    NoFilter,
    /// Filter constant bases
    NoConst,
    /// Filter any site with an ambiguous base
    NoAmbig,
    /// Filter constant bases, and any ambiguous bases
    NoAmbigOrConst,
}

/// As text, for use in logging messages
impl fmt::Display for FilterType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Self::NoFilter => write!(f, "No filtering"),
            Self::NoConst => write!(f, "No constant sites"),
            Self::NoAmbig => write!(f, "No ambiguous sites"),
            Self::NoAmbigOrConst => write!(f, "No constant sites or ambiguous bases"),
        }
    }
}

/// Options that apply to all subcommands
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct Args {
    #[doc(hidden)]
    #[command(subcommand)]
    pub command: Commands,

    /// Show progress messages
    #[arg(short, long, global = true)]
    pub verbose: bool,
}

/// Subcommands and their specific options
#[derive(Subcommand)]
pub enum Commands {
    #[command(group(
        ArgGroup::new("input")
            .required(true)
            .args(["seq_files", "file_list"]),
    ))]
    /// Assemble these input fastq files
    Build {
        /// List of input FASTQ files
        #[arg(group = "input")]
        seq_files: Option<Vec<String>>,

        /// File listing input file (tab separated name TAB fastq1,fastq2)
        #[arg(short, group = "input")]
        file_list: Option<String>,

        /// Output directory
        #[arg(long, default_value_t = DEFAULT_OUTPUT_DIR.to_string())]
        output_dir: String,

        /// Output prefix
        #[arg(long, default_value_t = DEFAULT_OUTPUT_PREFIX.to_string())]
        output_prefix: String,

        /// K-mer size
        #[arg(short, value_parser = valid_kmer, default_value_t = DEFAULT_KMER)]
        k: usize,

        /// Minimum k-mer count. If omitted, it is FITTED from the k-mer spectrum, separately for each k.
        #[arg(long)]
        min_count: Option<u16>,

        /// Minimum k-mer base quality, inclusive (with reads). Defaults to 20 at every k.
        #[arg(long)]
        min_qual: Option<u8>,

        /// Number of CPU threads
        #[arg(long, value_parser = valid_cpus, default_value_t = 1)]
        threads: usize,

        /// Use, instead of the default filtering, a Bloom filter. This will use less memory and be faster, but will add
        /// false positive matches to the counting. Explicit --min-count values 0 and 1 are not supported with Bloom filtering.
        #[arg(long, default_value_t = false)]
        do_bloom: bool,

        /// Set a value for the chunks of the reads during preprocessing. A value of zero ignores chunking.
        /// Native builds now count into a hash map, which buffers no occurrences, so this is a no-op
        /// there and is kept only so existing command lines still parse.
        #[arg(long, default_value_t = 100000)]
        chunk_size: usize,

        /// Fraction of the stronger branch's coverage below which the weaker branch of a bubble is
        /// treated as an error and popped.
        #[arg(long, default_value_t = crate::algorithms::corrector::DEFAULT_POP_RATIO)]
        bubble_pop_ratio: f32,

        /// Tip removal: dead-end paths shorter than this many bases are pruned. The threshold actually
        /// used is `max(--tip-length, --tip-length-kmult * k)`.
        #[arg(long, default_value_t = DEFAULT_TIP_LEN_NTS)]
        tip_length: usize,

        /// Tip removal: scale that threshold with k, as Minia does. Zero keeps the flat `--tip-length`,
        /// which is the historical behaviour.
        #[arg(long, default_value_t = DEFAULT_TIP_LEN_KMULT)]
        tip_length_kmult: f32,

        /// Minimum contig body length written to FASTA, in nucleotides. Contigs with exactly this
        /// length are retained.
        #[arg(long, default_value_t = DEFAULT_MIN_CONTIG_LENGTH_NTS)]
        min_contig_length: usize,

        /// By default, Sparrowhawk will draw your k-mer spectrum histogram and save it as PNG in the same folder
        /// where the contigs output will be. Use this argument if you want it to not do this
        #[arg(long, default_value_t = false)]
        no_histo: bool,

        /// By default, Sparrowhawk will extract the graph just before collapse and save it in your output folder
        /// in the DOT, GFAv1.1 and GFAv2 formats. Use this argument if you want it to not do this
        #[arg(long, default_value_t = false)]
        no_graphs: bool,

        /// Do not solve bubbles in the graph
        #[arg(long, default_value_t = false)]
        no_bubble_collapse: bool,

        /// Do not remove dead endsin the graph
        #[arg(long, default_value_t = false)]
        no_dead_end_removal: bool,
    },
}

/// Function to parse command line args into [`Args`] struct
pub fn cli_args() -> Args {
    Args::parse()
}
