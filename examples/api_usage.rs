#[cfg(not(target_family = "wasm"))]
use sparrowhawk_asm::{assemble, AssemblyOptions, InputFiles};

#[cfg(not(target_family = "wasm"))]
use std::{
    error::Error,
    io::{self, Write},
};

// This is a simple example to show how sparrowhawk-asm can be used as a dependency

#[cfg(not(target_family = "wasm"))]
fn main() -> Result<(), Box<dyn Error>> {
    // Own the input paths. InputFiles opens the FASTA/FASTQ files when
    // a pass requests records, rather than loading all reads here.
    let reads = InputFiles::new(std::env::args_os().skip(1))?;

    // Defaults choose quality/count thresholds automatically and use
    // the automatic multi-k ladder.
    //
    // Set k_values to vec![31] for single-k, or to vec![21, 31, 41]
    // for an explicit multi-k ladder. See api.rs for all parameters
    let options = AssemblyOptions::default();

    // Run preprocessing, any required recounts, graph assembly, and
    // sequence spelling through the shared pipeline.
    let assembly = assemble(&reads, &options)?;

    // Keep diagnostic information on stderr so stdout contains FASTA.
    eprintln!("Completed k values: {:?}", assembly.completed_k_values());
    if let Some((k, reason)) = assembly.stopped_at() {
        eprintln!("Stopped at k={k}: {reason}");
    }

    // Results stay packed. Writing FASTA decodes a small buffer at a
    // time and applies the configured minimum contig length.
    let stdout = io::stdout();
    let mut output = io::BufWriter::new(stdout.lock());
    assembly.write_fasta(&mut output)?;

    // Propagate errors that a buffered writer might otherwise encounter
    // only when it is dropped.
    output.flush()?;
    Ok(())
}

// This executable demonstrates native input and stdout. The browser
// uses the separately documented AssemblyHelper interface.
#[cfg(target_family = "wasm")]
fn main() {}
