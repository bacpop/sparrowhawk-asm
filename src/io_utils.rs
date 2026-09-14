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
    let input_files = match file_list {
        Some(files) => {
            let f = File::open(files).expect("Unable to open file_list");
            let f = BufReader::new(f);
            log::warn!("You have provided a input TSV file. Currently, all input read files will be considered as only one assembly. This might change in the future.");

            f.lines()
                .enumerate()
                .map(|(index, line)| {
                    let line = line.expect("Unable to read line in file_list");
                    parse_file_list_line(&line, index + 1)
                })
                .collect()
        }
        None => vec![(
            "reads".to_owned(),
            seq_files
                .clone()
                .expect("Neither input TSV file nor inputs as arguments have been provided"),
        )],
    };

    validate_input_files(input_files)
}

fn parse_file_list_line(line: &str, line_number: usize) -> InputFastx {
    let fields: Vec<&str> = line.split_whitespace().collect();

    assert!(
        fields.len() >= 2,
        "Invalid input file list line {line_number}: expected a sample name followed by at least one read file"
    );

    (
        fields[0].to_owned(),
        fields[1..]
            .iter()
            .map(|field| (*field).to_owned())
            .collect(),
    )
}

fn validate_input_files(input_files: Vec<InputFastx>) -> Vec<InputFastx> {
    assert!(
        !input_files.is_empty(),
        "Input file list contains no entries; provide at least one sample and read file"
    );

    for (name, files) in &input_files {
        assert!(
            !files.is_empty(),
            "Input file list entry {name:?} contains no read files"
        );
    }

    input_files
}

#[cfg(test)]
mod tests {
    use super::{parse_file_list_line, validate_input_files};

    #[test]
    #[should_panic(expected = "Input file list contains no entries")]
    fn empty_input_list_is_rejected() {
        validate_input_files(Vec::new());
    }

    #[test]
    #[should_panic(expected = "contains no read files")]
    fn input_entry_without_read_files_is_rejected() {
        validate_input_files(vec![("sample".to_owned(), Vec::new())]);
    }

    #[test]
    #[should_panic(expected = "Invalid input file list line 3")]
    fn blank_file_list_line_is_rejected_with_line_number() {
        parse_file_list_line("   ", 3);
    }

    #[test]
    fn file_list_line_parses_sample_and_read_files() {
        assert_eq!(
            parse_file_list_line("sample reads_1.fq reads_2.fq", 1),
            (
                "sample".to_owned(),
                vec!["reads_1.fq".to_owned(), "reads_2.fq".to_owned()]
            )
        );
    }
}

/// Iterator for needletail records
#[cfg(not(target_family = "wasm"))]
pub struct NeedletailIterator {
    reader: Box<dyn needletail::FastxReader>,
}

#[cfg(not(target_family = "wasm"))]
impl NeedletailIterator {
    /// Construct from needletail readers
    pub fn new(reader: Box<dyn needletail::FastxReader>) -> Self {
        Self { reader }
    }
}

#[cfg(not(target_family = "wasm"))]
impl Iterator for NeedletailIterator {
    type Item = (Vec<u8>, Option<Vec<u8>>);

    fn next(&mut self) -> Option<(Vec<u8>, Option<Vec<u8>>)> {
        let record = self.reader.next()?.expect("Invalid FASTA/Q record");
        let seq = record.seq();
        let qual = record.qual().map(|qual| qual.to_vec());
        Some((seq.to_vec(), qual))
    }
}
