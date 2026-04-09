# sparrowhawk <img src='sparrowhawk_logo.png' align="right" height="250" />
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

**Note:** this repository is for the Rust-based genomic assembler. If you are looking for its web implementation, see [sparrowhawk-web](https://github.com/bacpop/sparrowhawk-web).

sparrowhawk aims to be a fast short-read assembler for bacterial genomics. It has been developed taking advantage/inspiration of other Rust-based tools developed by our group (such as [ska.rust](https://github.com/bacpop/ska.rust)), as well as others (such as [Katome](https://github.com/fuine/katome) or [SKESA](https://github.com/ncbi/SKESA)).

Current **main features**:
- Currently only support for Illumina paired short reads.
- Single k-value, that must be odd and between 3 (not recommended to go below 19/21, due to memory requirements) and 256.
- Designed thinking on bacterial genomes (i.e. "small").
- Uses a node-based de Bruijn graph built upon [petgraph](https://docs.rs/petgraph/latest/petgraph).
- Partially parallelised with [rayon](https://docs.rs/rayon/latest/rayon).
- Compilation to WebAssembly targets (currently only `wasm32-unknown-unknown`) is possible to run the assembler in web projects. For more info, see [sparrowhawk-web](https://github.com/bacpop/sparrowhawk-web).

:construction: In-progress future main (not all) features: :construction:
- Partial GPU acceleration.
- Multi-k support.
- Improved error correction and graph collapse logics.


# Installation
Currently the only option is to compile from source.

## Compilation from source
Development has been done only on x86_64 GNU/Linux-based systems, and most surely will probably stay that way (i.e. no other systems have been tested). To compile our project from source as we did, you will need the [rust toolchain](https://www.rust-lang.org/tools/install) installed in your system. To get an in principle working version (up to some degree), always clone a versioned tag. E.g. to get version vX.Y.Z, you could use:

```
git clone --branch vX.Y.Z https://github.com/bacpop/sparrowhawk.git
```

sparrowhawk is designed to compile to run natively as a x86_64 binary, but you can also compile it to the WebAssembly target `wasm32-unknown-unknown`. You can see below how to do it manually (as we did for development). Check out [sparrowhawk-web](https://github.com/bacpop/sparrowhawk-web) for an integrated project with Javascript and [wasm-pack](https://github.com/rustwasm/wasm-pack).

### Compilation to x86_64 (default)
Move into the downloaded repository and use `cargo` to build the project. You can add the `--release` argument to include some compiler optimisations.

```
cd sparrowhawk
cargo build --release
```

If using the `--release` flag, this should place your compiled binary inside `target/release`.

### Compilation to wasm32-unknown-unknown
For this, you will need to activate the feature `wasm` with the `-F` argument and manually set the target. You can add the `--release` argument to include some compiler optimisations.

```
cd sparrowhawk
cargo build --release -F wasm --target wasm32-unknown-unknown
```

If using the `--release` flag, this should place your compiled binary inside `target/wasm32-unknown-unknown/release` as `sparrowhawk.wasm`.


# Usage
Here we will only consider the binary compiled for x86_64, refer to [sparrowhawk-web](https://github.com/bacpop/sparrowhawk-web) for an example of usage of the `wasm32-unknown-unknown` compilation target.

sparrowhawk can be called later to see the basic options and arguments with

```
./sparrowhawk
```

Currently, only the `build` option is present (apart from `help`), that allows assemblying the genomes. You can check its arguments running

```
./sparrowhawk build --help
```

An example execution could be the following:

```
./sparrowhawk build -f ./reads.tsv -k 31 --threads 1 -v --min-count 5 --output-dir ./ --output-prefix prefix
```

This will assemble your reads, with k=31 and using only one thread. The minimum repeats of one particular k-mer to be considered are 5 (which is also the default). The output contigs will be written in the current directory as a fasta file called `prefix_contigs.fasta`, given that we have indicated, using the `--output-prefix` argument the word "prefix" as prefix. The input files in this case are provided through a `reads.tsv` tab-separated file, that contains an identifier for your reads and the two file paths separated by a space, i.e. a file that contains this line

```
IDENTIFIER     /path/to/the/read_1.fastq /path/to/the/read_2.fastq
```

Alternatively, you could have run:

```
./sparrowhawk build /path/to/the/read_1.fastq /path/to/the/read_2.fastq -k 31 --threads 1 -v --min-count 5 --output-dir ./ --output-prefix prefix
```

In the same folder as the output FASTA file, the graph before collapsing will be exported in [DOT](https://en.wikipedia.org/wiki/DOT_%28graph_description_language%29), and [GFA](https://gfa-spec.github.io/GFA-spec/) versions 1.1 and 2 as `prefix_graph.dot`, `prefix_graph.gfa`, and `prefix_graph.gfa2` respectively. A histogram of the k-mer frequency spectrum will be saved in the same directory as `prefix_kmerspectrum.png`. These optional files can be avoided with the corresponding arguments.
