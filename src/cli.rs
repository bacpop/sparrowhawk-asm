//! Command line interface, built using [`crate::clap` with `Derive`](https://docs.rs/clap/latest/clap/_derive/_tutorial/index.html)
use std::fmt;

use clap::{ArgGroup, Parser, Subcommand, ValueEnum};

/// Default k-mer size
pub const DEFAULT_KMER: usize = 31;
/// Default minimum k-mer count for FASTQ files
pub const DEFAULT_MINCOUNT: u16 = 5;
/// Default minimum base quality (PHRED score) for FASTQ files
pub const DEFAULT_MINQUAL: u8 = 20;
/// Default output directory
pub const DEFAULT_OUTPUT_DIR: &str = "./";
/// Default output prefix
pub const DEFAULT_OUTPUT_PREFIX: &str = "sphk";

/// How k-mer occurrences are turned into k-mer counts.
///
/// Both counters are exact and must produce identical results. They differ only in what their cost
/// scales with, which is why both are kept: the choice is data-dependent.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Counter {
    /// DEPRECATED, slated for removal. Buffer every k-mer occurrence, sort it, and run-length count.
    /// This is strictly more work than `Map`: it builds the *same* distinct-k-mer map, and on top of it
    /// keeps an occurrence buffer (bounded by `--chunk-size`) that it then sorts. Measured 1.7-2.1x
    /// slower than `Map`, at equal memory where the distinct-k-mer map dominates and ~20% more where it
    /// does not. Kept only to reproduce the old behaviour.
    Sort,
    /// Count into a hash map keyed by the canonical hash: no occurrence buffer, no sort, so memory
    /// scales with the number of *distinct* k-mers. Validated on six real datasets (172x-862x): faster
    /// and leaner than `Sort`, with byte-identical results. The default.
    Map,
}

impl fmt::Display for Counter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Sort => write!(f, "sort"),
            Self::Map => write!(f, "map"),
        }
    }
}

/// How k-mers are extracted when more than one k is requested.
///
/// Both produce byte-for-byte identical results; they differ only in when the reads are touched.
///
/// Measured on the 1M-read simulation (12 threads, k=31,63), joint is a **wash**: 3.4% *slower* than
/// sequential with the default sort counter, 1.8% faster with the map counter. Reading the files once
/// instead of twice sounds like it should pay, but the I/O and record decode are a small share of the
/// work next to the hashing — which is inherently once per k, because ntHash's rolling state depends
/// on k. Sequential is therefore the default: never slower on the default counter, and much leaner on
/// memory (at `--chunk-size 0`, joint held two unbounded occurrence buffers: 4.7 GB against 3.6 GB).
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum MultiKExtraction {
    /// One pass over the reads per k. The default: at least as fast as `joint` on the default counter,
    /// and it lets each k's count structures be released before the next k is counted.
    Sequential,
    /// Read the files once, hashing every k from each batch of records. Saves the file read, the record
    /// decode and the per-record owned copy — but not the hashing. Worth trying if your reads are on
    /// slow or remote storage, where one fewer full pass over the files may actually matter; the
    /// benchmark above ran from page cache, so it understates that case.
    Joint,
}

impl fmt::Display for MultiKExtraction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Joint => write!(f, "joint"),
            Self::Sequential => write!(f, "sequential"),
        }
    }
}

#[doc(hidden)]
fn valid_kmer(s: &str) -> Result<usize, String> {
    let k: usize = s
        .parse()
        .map_err(|_| format!("`{s}` isn't a valid k-mer"))?;
    if !(3..=256).contains(&k) || k.is_multiple_of(2) {
        Err("K-mer must an odd number between 5 and 128 (inclusive)".to_string())
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

        /// K-mer size(s), comma-separated and strictly ascending.
        ///
        /// One value gives a standard single-k assembly. Two or more give a multi-k assembly: the FIRST
        /// is the *assembly* k, whose graph is corrected and output, and the rest are larger *evidence*
        /// k, used only to decide which branches of the assembly graph the reads actually support.
        ///
        /// A ladder is not redundant even though the largest evidence k resolves a superset of what the
        /// smaller ones do. Evidence k needs (k-1) bases of *unambiguous* flank to be usable at all, and
        /// in a tangled small-k graph that is often available for a small evidence k and not a large
        /// one — and each resolution lengthens the unitigs that the next k up then needs.
        ///
        /// For example `-k 31,63`, or `-k 19,31,45,63`.
        ///
        /// Note the values are comma-separated, not space-separated: a space-separated list would be
        /// ambiguous against the positional list of input FASTQ files.
        #[arg(short, value_parser = valid_kmer, value_delimiter = ',', num_args = 1..,
              default_values_t = vec![DEFAULT_KMER])]
        k: Vec<usize>,

        /// Minimum k-mer count. If omitted, it is FITTED from the k-mer spectrum, separately for each k.
        #[arg(long)]
        min_count: Option<u16>,

        /// Minimum k-mer quality (with reads)
        #[arg(long, default_value_t = DEFAULT_MINQUAL)]
        min_qual: u8,

        /// Number of CPU threads
        #[arg(long, value_parser = valid_cpus, default_value_t = 1)]
        threads: usize,

        /// Use, instead of the default filtering, a Bloom filter. This will use less memory and be faster, but will add
        /// false positive matches to the counting, making possible that a k-mer is counted more times that it should be.
        #[arg(long, default_value_t = false)]
        do_bloom: bool,

        /// DEPRECATED. Bounds the occurrence buffer of the `sort` counter only; the default counter is
        /// now `map`, which buffers nothing, so this has no effect there. Slated for removal with `sort`.
        /// A value of zero disables chunking (`sort` only).
        #[arg(long, default_value_t = 100000)]
        chunk_size: usize,

        /// How to count k-mers. `map` (default) counts into a hash map keyed by canonical hash. `sort`
        /// is the DEPRECATED older path: it buffers every occurrence, sorts it, and run-length counts —
        /// which is strictly more work and more memory than `map` (it keeps the same distinct-k-mer map
        /// *plus* the occurrence buffer), and is slated for removal. Measured at k=31, `map` is
        /// 1.7-2.1x faster and never uses more memory (level where the distinct-k-mer map dominates,
        /// ~20% leaner where it does not), and both produce identical results. Use `sort` only to
        /// reproduce the old behaviour.
        #[arg(long, value_enum, default_value_t = Counter::Map)]
        counter: Counter,

        /// How to extract k-mers when two k values are given. `sequential` (default) makes one pass
        /// over the reads per k. `joint` reads the files once and hashes every k from each batch;
        /// measured, it is a wash (~3% either way), because the hashing — not the I/O — dominates.
        /// Both produce identical results. Ignored when only one k is given.
        #[arg(long, value_enum, default_value_t = MultiKExtraction::Sequential)]
        multik_extraction: MultiKExtraction,

        /// Multi-k: minimum number of reconstructed evidence-k k-mers that must all be present before
        /// a bubble branch is accepted as supported by the reads.
        #[arg(long, default_value_t = 5)]
        multik_min_evidence: usize,

        /// Multi-k: maximum number of distinct evidence-graph nodes a supported branch may span. A
        /// branch confined to one evidence unitig is unambiguous at the larger k; allowing a few more
        /// tolerates the boundaries where the branch meets its flanks.
        #[arg(long, default_value_t = 3)]
        multik_max_nodes: usize,

        /// Multi-k: how many k-mers of each flanking unitig to use as context around a bubble. Must
        /// exceed (k2 - 1 - k1), the minimum needed to span the bubble at all. Larger values buy more
        /// evidence and are what make repeat diagnosis possible, at a cost linear in this value.
        #[arg(long, default_value_t = 200)]
        multik_flank_context: usize,

        /// Multi-k: maximum number of evidence-driven correction rounds.
        #[arg(long, default_value_t = 10)]
        multik_max_rounds: usize,

        /// Multi-k: do NOT resolve collapsed repeats.
        ///
        /// By default, when two k are given, a repeat that the evidence k can span — and whose flanks
        /// the reads pair unambiguously with one branch — is duplicated, so that both genomic copies
        /// survive and contigs run straight through it. It is the only correction here that throws
        /// nothing away, and it measures strictly better: +2.7 kb of real genome recovered, zero
        /// misassemblies, duplication ratio unchanged at 1.000.
        ///
        /// This flag disables it, leaving such bubbles to the coverage heuristic — which, since the two
        /// branches of a collapsed repeat have identical coverage, then picks between them by coin flip
        /// and deletes one real copy.
        #[arg(long, default_value_t = false)]
        no_multik_resolve: bool,

        /// Multi-k: also split SUPERBUBBLES — forks with three or more branches, or branches more than
        /// one unitig long — and not just simple bubbles.
        ///
        /// The same surgery repeat resolution already performs, generalised: the shared repeat around
        /// the fork is duplicated once per path, so every genomic copy survives and contigs run
        /// straight through. Measured read-only, these are 38-97% again as many resolvable loci as the
        /// simple bubbles the assembler acts on today.
        ///
        /// OFF BY DEFAULT while the effect is being measured. The surgery duplicates nodes, so the
        /// number to watch is QUAST's duplication ratio, which simple-bubble resolution holds at 1.000.
        /// `--multik-survey-only` reports how many loci would be split, without splitting any.
        #[arg(long, default_value_t = false)]
        multik_resolve_superbubbles: bool,

        /// Multi-k: survey bubbles and superbubbles against the evidence k, correct nothing, and
        /// leave the multi-k stage.
        ///
        /// Read-only by construction. The contigs of `-k 31,89 --multik-survey-only` are
        /// byte-identical to those of `-k 31`, and that identity is the check that the survey really
        /// does not touch the graph. So a run answers "what is here?" and not "what would correcting
        /// it give?" — to get both the numbers and a corrected assembly, run twice.
        ///
        /// What it adds over the existing per-bubble counters is superbubbles: forks with three or
        /// more branches, branches more than one unitig long, and nested bubbles. None of those are
        /// recognised by the bubble detector the assembler corrects with, so today they simply break
        /// the contig.
        #[arg(long, default_value_t = false)]
        multik_survey_only: bool,

        /// Fraction of the stronger branch's coverage below which the weaker branch of a bubble is
        /// treated as an error and popped.
        ///
        /// This is the ONLY thing that licenses popping. At or above it both branches are taken to be
        /// real — which is what a collapsed repeat looks like, its two copies having equal length and
        /// equal coverage — and the bubble is left exactly as it is, contig break and all. Applies to
        /// superbubbles too: there, every losing path must be under this fraction of the winner.
        ///
        /// Raising it pops more aggressively and risks deleting one copy of a real repeat; lowering it
        /// leaves more forks, and so more contig breaks, but destroys nothing.
        #[arg(long, default_value_t = crate::algorithms::corrector::DEFAULT_POP_RATIO)]
        bubble_pop_ratio: f32,

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
