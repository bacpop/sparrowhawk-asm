//! Native input-file parsing and buffered output, shared FASTA writing, and browser gzip output.

use crate::preprocessing::read_store::PackedSeqs;
use crate::utils::bit_encoding::decode_base;
#[cfg(target_family = "wasm")]
use flate2::{write::GzEncoder, Compression};
use std::fs::File;
use std::io::{stdout, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::utils::reads::InputFastx;

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

/// Bases per FASTA line.
const FASTA_LINE_WIDTH: usize = 80;

/// Write sequences clearing the inclusive length filter, with consecutive names and 80-base lines.
/// Returns the number written; decoding needs only one fixed-size line buffer.
pub(crate) fn write_fasta<W: Write>(
    seqs: &PackedSeqs,
    min_contig_length: usize,
    out: &mut W,
) -> std::io::Result<usize> {
    let mut written = 0;
    let mut line = [0u8; FASTA_LINE_WIDTH];
    for seq in seqs.iter().filter(|s| s.len() >= min_contig_length) {
        writeln!(out, ">{written}")?;
        for start in (0..seq.len()).step_by(FASTA_LINE_WIDTH) {
            let len = (seq.len() - start).min(FASTA_LINE_WIDTH);
            for (i, base) in line[..len].iter_mut().enumerate() {
                *base = decode_base(seq.code(start + i));
            }
            out.write_all(&line[..len])?;
            out.write_all(b"\n")?;
        }
        written += 1;
    }
    Ok(written)
}

/// What `write` produces, gzipped as one member: how the page keeps and offers its downloads.
#[cfg(target_family = "wasm")]
pub(crate) fn gzipped(
    write: impl FnOnce(&mut GzEncoder<Vec<u8>>) -> std::io::Result<()>,
) -> Vec<u8> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    write(&mut encoder)
        .and_then(|()| encoder.finish())
        .expect("writing to memory cannot fail")
}

/// Writes spelled contigs of at least `min_contig_length` bases as FASTA to `outfile`.
pub(crate) fn save_sequences_as_fasta(
    seqs: &PackedSeqs,
    min_contig_length: usize,
    outfile: PathBuf,
) {
    let mut wbuf = set_ostream(&Some(outfile.into_os_string().into_string().unwrap()));
    let written =
        write_fasta(seqs, min_contig_length, &mut wbuf).expect("cannot write the contigs");
    log::info!(
        "Omitted {} contigs shorter than {min_contig_length} nt from FASTA output",
        seqs.len() - written
    );
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn packed(seqs: &[Vec<u8>]) -> PackedSeqs {
        let mut result = PackedSeqs::new(true);
        for seq in seqs {
            result.push_contig(
                &seq.iter()
                    .map(|&b| crate::utils::bit_encoding::encode_base(b))
                    .collect::<Vec<_>>(),
            );
        }
        result.finish();
        result
    }

    #[test]
    fn minimum_contig_length_is_inclusive_and_configurable() {
        // The threshold applies to the sequence written, not to a trimmed body, so it means what it says.
        let sequences = [vec![b'A'; 500], vec![b'C'; 499]];
        let packed = packed(&sequences);
        assert_eq!(write_fasta(&packed, 500, &mut Vec::new()).unwrap(), 1);
        assert_eq!(write_fasta(&packed, 499, &mut Vec::new()).unwrap(), 2);
        assert_eq!(write_fasta(&packed, 501, &mut Vec::new()).unwrap(), 0);
    }

    /// Headers are the contigs' indices, and sequences wrap every `FASTA_LINE_WIDTH` bases.
    #[test]
    fn fasta_wraps_sequences_at_the_line_width() {
        let seqs = vec![
            vec![b'A'; 2 * FASTA_LINE_WIDTH + 5],
            vec![b'C'; FASTA_LINE_WIDTH],
        ];
        let mut out = Vec::new();
        write_fasta(&packed(&seqs), 0, &mut out).unwrap();
        let a = "A".repeat(FASTA_LINE_WIDTH);
        let c = "C".repeat(FASTA_LINE_WIDTH);
        assert_eq!(
            String::from_utf8(out).unwrap(),
            format!(">0\n{a}\n{a}\nAAAAA\n>1\n{c}\n")
        );
    }
}
