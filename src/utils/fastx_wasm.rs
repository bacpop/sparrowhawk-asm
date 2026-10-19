//! FASTA/FASTQ records from a browser `File`, gzipped or not. Taken originally from DATACIN, but improved!

use flate2::read::MultiGzDecoder;
use seq_io::fasta::Reader as FastaReader;
use seq_io::fastq::{Reader as FastqReader, Record as _};
use std::io::{BufRead, BufReader, Read};

use crate::utils::reads::OwnedRecord;

// What a .gz file starts with
const GZ_MAGIC: [u8; 2] = [0x1F, 0x8B];

/// The file's bytes, decompressed when they start with the gzip magic.
type Decoded = Box<dyn Read>;

/// Every record of one file, owned, as `NeedletailIterator` yields them natively.
pub enum FastxRecords {
    /// A FASTQ file.
    Fastq(FastqReader<Decoded>),
    /// A FASTA file: no qualities.
    Fasta(FastaReader<Decoded>),
}

impl FastxRecords {
    /// Open `file`, telling gzip by its magic bytes and FASTA from FASTQ by the first character. An
    /// empty file reads as a FASTQ file with no records.
    pub fn open<R: Read + 'static>(file: R) -> std::io::Result<Self> {
        let mut raw = BufReader::new(file);
        let gzipped = raw.fill_buf()?.starts_with(&GZ_MAGIC);
        let decoded: Decoded = if gzipped {
            Box::new(MultiGzDecoder::new(raw))
        } else {
            Box::new(raw)
        };
        let mut decoded = BufReader::new(decoded);
        let first = decoded.fill_buf()?.first().copied();
        let decoded: Decoded = Box::new(decoded);
        match first {
            Some(b'>') => Ok(Self::Fasta(FastaReader::new(decoded))),
            _ => Ok(Self::Fastq(FastqReader::new(decoded))),
        }
    }
}

impl Iterator for FastxRecords {
    type Item = std::io::Result<OwnedRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Fastq(reader) => reader.next().map(|record| {
                record
                    .map(|record| (record.seq().to_vec(), Some(record.qual().to_vec())))
                    .map_err(|error| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string())
                    })
            }),
            Self::Fasta(reader) => reader.next().map(|record| {
                record
                    .map(|record| (record.full_seq().into_owned(), None))
                    .map_err(|error| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string())
                    })
            }),
        }
    }
}
