# sparrowhawk-asm 
Short-read assembler for bacterial genomics based on a de Bruijn graph written in Rust.


<br>
<br>
<br>
<br>


---
## Disclaimer :warning: :construction:
This is a **work in progress** project. This in particular implies:

- Not all the main features we want are yet implemented.
- Code might be messy, and not even documented.
- General documentation on how to install and use the tool might be short or even missing.
- Finding unexpected errors/behaviour or bugs should not be a surprise.
- Some features might be partially hardcoded.

These (and potentially other) items will be progressively fixed before version 1.0.

---


## sparrowhawk?
Sparrowhawk was at one time the Archmage of [Earthsea](https://en.wikipedia.org/wiki/Earthsea).
Also, the [sparrowhawk](https://en.wikipedia.org/wiki/Eurasian_sparrowhawk) (*Accipiter nisus*) is a bird of prey native to Europe (and the island of Gont).

# Description

**Note:** this repository is for the Rust-based genomic assembler. If you are looking for its web implementation, see [sparrowhawk](https://github.com/bacpop/sparrowhawk).

sparrowhawk aims to be a fast short-read assembler for bacterial genomics that can run in a browser. It has been developed taking advantage/inspiration of other Rust-based tools developed by our group (such as [ska.rust](https://github.com/bacpop/ska.rust)), as well as others (such as [Katome](https://github.com/fuine/katome) or [SKESA](https://github.com/ncbi/SKESA)).

Current **main features**:
- Currently only support for Illumina paired short reads.
- Single-k and iterative multi-k assembly, with odd k values in `3..=255`; very small k values require more memory.
- Designed thinking on bacterial genomes (i.e. "small").
- Uses a node-based de Bruijn graph built upon [petgraph](https://docs.rs/petgraph/latest/petgraph).
- Partially parallelised with [rayon](https://docs.rs/rayon/latest/rayon).
- Compilation to WebAssembly targets (`wasm32-unknown-unknown` and the browser integration's WASM64 build) is possible to run the assembler in web projects. For more info, see [sparrowhawk](https://github.com/bacpop/sparrowhawk).

:construction: In-progress future main (not all) features: :construction:
- Partial GPU acceleration.
- Improved error correction and graph collapse logics.


# Installation
Currently the only option is to compile from source.

## Compilation from source
Development has been done only on x86_64 GNU/Linux-based systems, and most surely will probably stay that way (i.e. no other systems have been tested). To compile our project from source as we did, you will need the [rust toolchain](https://www.rust-lang.org/tools/install) installed in your system. To get an in principle working version (up to some degree), always clone a versioned tag. E.g. to get version vX.Y.Z, you could use:

```
git clone --branch vX.Y.Z https://github.com/bacpop/sparrowhawk-asm.git
```

sparrowhawk is designed to compile to run natively as a x86_64 binary, but you can also compile it to the WebAssembly target `wasm32-unknown-unknown`. You can see below how to do it manually (as we did for development). Check out [sparrowhawk](https://github.com/bacpop/sparrowhawk) for an integrated project with Javascript and [wasm-pack](https://github.com/rustwasm/wasm-pack).

### Compilation to x86_64 (default)
Move into the downloaded repository and use `cargo` to build the project. You can add the `--release` argument to include some compiler optimisations.

```
cd sparrowhawk-asm
cargo build --release
```

If using the `--release` flag, this should place your compiled binary inside `target/release`.

### Compilation to wasm32-unknown-unknown
For this, you only need to add the target to your toolchain and set it manually when building: the WebAssembly-specific dependencies and code paths are selected automatically through `cfg(target_family = "wasm")`. You can add the `--release` argument to include some compiler optimisations.

```
cd sparrowhawk-asm
rustup target add wasm32-unknown-unknown
cargo build --release --target wasm32-unknown-unknown
```

If using the `--release` flag, this should place your compiled binary inside `target/wasm32-unknown-unknown/release` as `sparrowhawk_asm.wasm`.


# Usage
Here we will only consider the binary compiled for x86_64, refer to [sparrowhawk](https://github.com/bacpop/sparrowhawk) for an example of usage of the `wasm32-unknown-unknown` compilation target.

Run `./sparrowhawk-asm --help` to see its options.

An example execution could be the following:

```
./sparrowhawk-asm -f ./reads.tsv -k 31 --threads 1 -v --min-count 5 --output-dir ./ --output-prefix prefix
```

This will assemble your reads, with k=31 and using only one thread. The minimum repeats of one particular k-mer to be considered are 5 (which is also the default). The output contigs will be written in the current directory as a fasta file called `prefix_contigs.fasta`, given that we have indicated, using the `--output-prefix` argument the word "prefix" as prefix. The input files in this case are provided through a `reads.tsv` tab-separated file, that contains an identifier for your reads and the two file paths separated by a space, i.e. a file that contains this line

By default, only contigs of at least 500 nucleotides are written to FASTA. The threshold can be changed with `--min-contig-length`; it is inclusive, and applies to the full sequence the contig spells.

```
IDENTIFIER     /path/to/the/read_1.fastq /path/to/the/read_2.fastq
```

Alternatively, you could have run:

```
./sparrowhawk-asm /path/to/the/read_1.fastq /path/to/the/read_2.fastq -k 31 --threads 1 -v --min-count 5 --output-dir ./ --output-prefix prefix
```

The native CLI uses Bloom counting by default to reduce memory use, with possible false-positive k-mer counts.
Pass `--no-bloom` to use exact counts instead. Bloom counting requires an explicit `--min-count` of at least 2;
explicit values 0 or 1 require `--no-bloom`. The old `--do-bloom` flag has been removed: omit it to use the new
default, or add `--no-bloom` to preserve the former non-Bloom default.

A multi-k run keeps its reads in memory after the first k, whole, at four bits a base (a base code and a quality
class), and counts every later k from there. If you prefer to do a faster, single-k execution, you might use a method like [KmerGenie](http://kmergenie.bx.psu.edu/) to select k.

In the same folder as the output FASTA file, the graph before collapsing will be exported in [DOT](https://en.wikipedia.org/wiki/DOT_%28graph_description_language%29), and [GFA](https://gfa-spec.github.io/GFA-spec/) versions 1.1 and 2 as `prefix_graph.dot`, `prefix_graph.gfa`, and `prefix_graph.gfa2` respectively. Two native diagnostic plots of the k-mer frequency spectrum are saved beside them: `prefix_kmerspectrum.png`, linear and capped at three times the empirical peak's height (unbounded without a peak), and `prefix_kmerspectrum_log.png`, with a logarithmic Y axis. Each is also written as SVG. Runs using automatic count selection overlay the fitted error, single-copy, and repeat components, the total mixture, the empirical valley and peak, the fitted single-copy mean and mode, and the error/genome crossover. With Bloom counting, the plots show both the rescaled sketch used for fitting and the raw Bloom count map. These optional files can be avoided with the corresponding arguments.

## Rust library API

Use `assemble()` with `AssemblyOptions` for the complete assembly pipeline, or run
preprocessing, graph assembly, spelling, and FASTA writing separately.

To use this checkout as a dependency, adjust the path below:

```toml
[dependencies]
sparrowhawk-asm = { path = "../sparrowhawk-asm" }
```

See the commented examples:

- [Complete workflow](examples/api_usage.rs): automatic assembly with the default settings.
- [Step-by-step workflow](examples/api_usage_step_by_step.rs): call each stage independently and inspect preprocessing results.

Run either example with your input files:

```sh
cargo run --release --example api_usage -- reads_1.fastq.gz reads_2.fastq.gz > contigs.fasta
cargo run --release --example api_usage_step_by_step -- reads_1.fastq.gz reads_2.fastq.gz > contigs.fasta
```

## Source layout

```text
src/
├── main.rs                 # Native executable entry point.
├── cli.rs                  # CLI options, defaults, and argument validation.
├── lib.rs                  # Public exports, native entry, and browser interface.
├── api.rs                  # Complete assembly API, options, validation, and packed results.
├── pipeline.rs             # Shared single-k and multi-k assembly workflow, for both execution targets.
├── preprocessing/          # Read packing, k-mer counting, and spectrum filtering.
│   ├── mod.rs              # Preprocessing orchestration and empirical spectrum analysis.
│   ├── sketch.rs           # Hash sampling and per-quality-floor k-mer spectra.
│   ├── spectrum_fitter.rs  # Optimiser execution, fit validation, and model selection.
│   ├── math_helpers.rs     # Numerical helpers, probability distributions, and recurrence calculations.
│   ├── models.rs           # Coverage models, their parameters, and mixture evaluation.
│   ├── bloom_filter.rs     # Bloom filtering to reduce memory used for counting.
│   ├── kmer.rs             # K-mer iteration over text and packed sequences.
│   ├── read_store.rs       # Packed read and contig storage for repeated processing.
│   ├── filter.rs           # Quality-floor selection and k-mer count filtering.
│   ├── plotter.rs          # Spectrum plots and diagnostic spectrum tables.
│   └── parsercounter.rs    # Batched parsing, parallel counting, and carried-contig counting.
├── graph_works/            # Graph construction, correction, collapse, and contig spelling.
│   ├── mod.rs              # Graph assembly orchestration and contig representation.
│   ├── indexed_kmers.rs    # Indexed k-mer storage for graph construction and spelling.
│   ├── neighbours.rs       # Neighbour discovery and graph edge calculation.
│   ├── spelling.rs         # Conversion of graph walks into packed contig sequences.
│   ├── collapser.rs        # Collapse corrected graphs into contig walks.
│   ├── corrector.rs        # Correction settings, coverage checks, and dead-end removal.
│   ├── path_correction.rs  # Bubble and erroneous-connection correction.
│   └── shrinker.rs         # Merge compatible graph paths into unitigs.
└── utils/                  # Input/output, encoding, hashing, logging, and timing utilities.
    ├── mod.rs              # Utility module declarations.
    ├── error.rs            # Assembly API errors, re-exported at the crate root.
    ├── logging.rs          # Native and browser logging.
    ├── fastx_wasm.rs       # Browser FASTA/FASTQ parsing and gzip input.
    ├── io_utils.rs         # Native input handling and FASTA/gzip output helpers.
    ├── clock.rs            # Monotonic timing on native and WebAssembly targets.
    ├── reads.rs            # Read records, input descriptions, sources, and browser progress.
    ├── bit_encoding.rs     # Base encoding and packed integer operations.
    ├── nthash.rs           # Rolling DNA hashes and lookup tables.
    └── qual_profile.rs     # Quality-floor discovery and k-mer window classification.
```

```text
examples/
├── api_usage.rs                 # Complete automatic assembly and buffered FASTA output.
└── api_usage_step_by_step.rs    # Separate preprocessing, graph assembly, spelling, and output.
```
