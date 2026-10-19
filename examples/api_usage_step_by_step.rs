#[cfg(not(target_family = "wasm"))]
use sparrowhawk_asm::{
    graph_works::{spelling::spell_contigs, Assemble, BasicAsm},
    preprocessing::{preprocessing, PassInput, PreprocessingOptions},
    AssemblyOptions, InputFiles, QualOpts, ReadSource,
};

#[cfg(not(target_family = "wasm"))]
use std::{
    error::Error,
    io::{self, Write},
};

// This is a more involved example, that is NOT needed to be done (see the simpler in the other file),
// but allows you to see how the basic pipeline works and if you just want to use part of it, you can
// see how to do it.

#[cfg(not(target_family = "wasm"))]
fn main() -> Result<(), Box<dyn Error>> {
    // Files are represented by paths; records are read lazily through
    // the public ReadSource trait.
    let files = InputFiles::new(std::env::args_os().skip(1))?;

    // Use fixed thresholds so that the individual stages are explicit.
    // The complete API additionally handles automatic quality-floor
    // selection, recounting, and multi-k iteration.
    let k = 31;
    let options = AssemblyOptions {
        k_values: vec![k],
        min_count: Some(2),
        min_qual: Some(20),
        use_bloom: false,
        ..AssemblyOptions::default()
    };

    // Stage 1: extract, count, and filter k-mers from an iterator.
    //
    // u64 holds the 62 packed bits needed for k=31. Larger k values
    // require a sufficiently wide integer type. Quality bytes are
    // PHRED+33, while min_qual is the numerical PHRED score.
    let mut counted = preprocessing::<u64, _>(
        PassInput::Text(files.records()?),
        k,
        &PreprocessingOptions {
            quality: QualOpts {
                min_count: 2,
                min_qual: 20,
            },
            use_bloom: false,
            fit_min_count: false,
        },
    )?;

    // Preprocessing is independently usable. Callers can inspect its
    // k-mers, counts, histogram, and coverage estimate here, or stop
    // without proceeding to graph assembly.
    eprintln!("Retained {} k-mers", counted.kmers.len());
    eprintln!("Minimum count used: {}", counted.used_min_count);

    // Stage 2: build, correct, and collapse the graph. These calls use
    // the existing k-mer result; they do not read or count the files
    // again. Derive correction settings as the complete pipeline does.
    let correction = options.correction_for(k, counted.genomic_peak, counted.used_min_count)?;
    let contigs = BasicAsm::assemble(k, &mut counted.kmers, correction, None)?;

    // Stage 3: spell graph walks into packed sequences. Assembly has
    // consumed graph-construction arrays, but retained the packed
    // k-mer lookup required to recover the bases of each walk.
    let sequences = spell_contigs(&contigs, |hash| counted.kmers.get_packed(hash).copied(), k)?;

    // Stage 4: write FASTA without collecting a second, uncompressed
    // copy of the complete assembly. The length threshold is inclusive.
    let stdout = io::stdout();
    let mut output = io::BufWriter::new(stdout.lock());
    sequences.write_fasta(options.min_contig_length, &mut output)?;
    output.flush()?;
    Ok(())
}

// The executable example uses native files and stdout.
#[cfg(target_family = "wasm")]
fn main() {}
