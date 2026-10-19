//! Spelling contigs back to nucleotides, and writing them as FASTA.

use core::panic;
#[cfg(target_family = "wasm")]
use flate2::{write::GzEncoder, Compression};
use std::io::Write;

use std::path::PathBuf;

use crate::bit_encoding::{decode_base, UInt};
use crate::graph_works::{spell_path, Contigs};
use crate::io_utils::set_ostream;
use crate::read_store::PackedSeqs;

/// Spells every contig, looking each packed k-mer up with `lookup`. A multi-k step carries all of them
/// to the next k, so none is length-filtered.
pub(crate) fn spell_contigs<IntT>(
    contigs: &Contigs,
    lookup: impl Fn(u64) -> Option<IntT>,
    k: usize,
) -> PackedSeqs
where
    IntT: for<'a> UInt<'a>,
{
    let mut spelled = PackedSeqs::new(true);
    for (i, contig) in contigs.serialized_contigs.iter().enumerate() {
        assert_eq!(contig.len(), 1, "contig {i} was not shrunk to one node");
        let codes = spell_path(&contig[0].abs_ind, &lookup, k)
            .unwrap_or_else(|e| panic!("contig {i} is not a valid walk: {e}"));
        spelled.push_contig(&codes);
    }
    spelled.finish();
    spelled
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

//
//
//
//
// ===================================================================
// ============================================================= TESTS
// ==================================================================
#[cfg(test)]
mod tests {
    use super::*;
    fn packed(seqs: &[Vec<u8>]) -> PackedSeqs {
        let mut result = PackedSeqs::new(true);
        for seq in seqs {
            result.push_contig(
                &seq.iter()
                    .map(|&b| crate::bit_encoding::encode_base(b))
                    .collect::<Vec<_>>(),
            );
        }
        result.finish();
        result
    }

    use crate::graph_works::Contigs;
    use crate::kmer::Kmer;
    use nohash_hasher::NoHashHasher;
    use sparrowhawk_graph::NodeStruct;
    use std::borrow::Cow;
    use std::{collections::HashMap, hash::BuildHasherDefault};

    /// A deterministic ACGT sequence. An LCG keeps it reproducible without pulling in a rng crate.
    fn pseudo_seq(len: usize) -> Vec<u8> {
        const BASES: [u8; 4] = *b"ACGT";
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                BASES[((state >> 33) & 3) as usize]
            })
            .collect()
    }

    /// Canonical hash -> packed k-mer, as preprocessing stores it.
    type PackedDict = HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>;

    /// The (dict, abs_ind) pair preprocessing would produce: canonical hash -> packed k-mer, plus the
    /// ordered hashes along the walk.
    fn dict_and_path(seq: &[u8], k: usize) -> (PackedDict, Vec<u64>) {
        let mut dict = PackedDict::default();
        let mut path: Vec<u64> = Vec::new();

        let mut it = Kmer::<u64>::new(Cow::Borrowed(seq), seq.len(), None, k, 0, true).unwrap();
        let (hc, _, _, km) = it.get_hashes_and_bases_and_kmer();
        dict.insert(hc, km);
        path.push(hc);
        while let Some((hc, _, _, km)) = it.get_next_kmer() {
            dict.insert(hc, km);
            path.push(hc);
        }
        (dict, path)
    }

    fn one_contig(path: Vec<u64>) -> Contigs {
        Contigs::new(vec![vec![NodeStruct {
            counts: 1,
            abs_ind: path,
            innerdir: None,
        }]])
    }

    /// A contig is the whole sequence its k-mer path spells, so it must come back as `S` itself.
    #[test]
    fn contig_preserves_the_full_spelled_path() {
        let k = 31;
        let seq = pseudo_seq(300);
        let n = seq.len() - k + 1; // number of k-mers
        let (dict, path) = dict_and_path(&seq, k);
        assert_eq!(path.len(), n);

        let sequences = spell_contigs(&one_contig(path), |hash| dict.get(&hash).copied(), k);
        let got = sequences.iter().next().unwrap().ascii();
        assert_eq!(got.len(), seq.len(), "contig length");
        assert_eq!(got, seq, "contig sequence");
    }

    /// Both terminal k-mers survive. Guards the end-symmetry the old trim had to be fixed for: an
    /// off-by-one at either end would clip exactly one of these.
    #[test]
    fn contig_preserves_both_terminal_kmers() {
        let k = 31;
        let seq = pseudo_seq(300);
        let (dict, path) = dict_and_path(&seq, k);

        let sequences = spell_contigs(&one_contig(path), |hash| dict.get(&hash).copied(), k);
        let got = sequences.iter().next().unwrap().ascii();
        assert_eq!(&got[..k], &seq[..k], "the first terminal k-mer");
        assert_eq!(&got[got.len() - k..], &seq[seq.len() - k..], "the last one");
    }

    /// A path of fewer than k k-mers used to be dropped whatever the minimum, by a `2*k-1` guard that
    /// outlived the trim it protected. It cost ~104 contigs a run.
    #[test]
    fn a_path_shorter_than_k_kmers_is_still_written() {
        let k = 31;
        let seq = pseudo_seq(k + 4); // 5 k-mers, far fewer than k
        let (dict, path) = dict_and_path(&seq, k);
        assert!(path.len() < k, "fixture must have fewer than k k-mers");

        let sequences = spell_contigs(&one_contig(path), |hash| dict.get(&hash).copied(), k);
        assert_eq!(sequences.len(), 1, "the short path must still be written");
        assert_eq!(sequences.iter().next().unwrap().ascii(), seq);
    }

    #[test]
    fn minimum_contig_length_is_inclusive_and_configurable() {
        // The threshold applies to the sequence written, not to a trimmed body, so it means what it says.
        let sequences = [pseudo_seq(500), pseudo_seq(499)];
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
