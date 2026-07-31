//! Common helper functions for parsing file input, loading, and setting output
//!
//! The functions are used by a few different subcommands to set correct
//! args to build structs, given the command line input

// use std::error::Error;
use std::fs::File;
use std::io::{stdout, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

// use regex::Regex;

// use super::QualOpts;
use crate::preprocessing::InputFastx;
// use crate::bit_encoding::UInt;

// use crate::cli::{
//     DEFAULT_KMER, DEFAULT_MINCOUNT, DEFAULT_MINQUAL, DEFAULT_STRAND,
// };

/// Set a buffered stream to write to.
///
/// Either a file (if [`Some`]) or stdout otherwise (if [`None`]).
pub fn set_ostream(oprefix: &Option<String>) -> BufWriter<Box<dyn Write>> {
    let out_writer = match oprefix {
        Some(prefix) => {
            let path = Path::new(prefix);
            Box::new(File::create(path).unwrap()) as Box<dyn Write>
        }
        None => Box::new(stdout()) as Box<dyn Write>,
    };
    BufWriter::new(out_writer)
}

/// Obtain a list of input files and names from command line input.
///
/// If `file_list` is provided, its entries will be parsed but currently they will be all assembled together.
/// If `seq_files` are provided use, all input files will be merged.
pub fn get_input_list(
    file_list: &Option<String>,
    seq_files: &Option<Vec<String>>,
) -> Vec<InputFastx> {
    // Read input
    match file_list {
        Some(files) => {
            let mut input_files: Vec<InputFastx> = Vec::new();
            let f = File::open(files).expect("Unable to open file_list");
            let f = BufReader::new(f);
            log::warn!("You have provided a input TSV file. Currently, all input read files will be considered as only one assembly. This might change in the future.");
            for line in f.lines() {
                let line = line.expect("Unable to read line in file_list");
                let fields: Vec<&str> = line.split_whitespace().collect();
                let files: Vec<String> = fields.iter().skip(1).map(|x| x.to_string()).collect();
                input_files.push((fields[0].to_string(), files));
            }
            input_files
        }
        None => {
            vec![(
                "reads".to_owned(),
                seq_files
                    .clone()
                    .expect("Neither input TSV file nor inputs as arguments have been provided"),
            )]
        }
    }
}

/// Iterator for needletail records
#[cfg(not(target_arch = "wasm32"))]
pub struct NeedletailIterator {
    reader: Box<dyn needletail::FastxReader>,
}

#[cfg(not(target_arch = "wasm32"))]
impl NeedletailIterator {
    /// Construct from needletail readers
    pub fn new(
        reader: Box<dyn needletail::FastxReader>,
    ) -> Self {
        Self {
            reader,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Iterator for NeedletailIterator {
    type Item = (Vec<u8>, Option<Vec<u8>>);

    fn next(
        &mut self,
    ) -> Option<(Vec<u8>, Option<Vec<u8>>)> {
        let record = self.reader.next()?.expect("Invalid FASTA/Q record");
        let seq = record.seq();
        let qual = record.qual().map(|qual| qual.to_vec());
        Some((seq.to_vec(), qual))
    }
}
