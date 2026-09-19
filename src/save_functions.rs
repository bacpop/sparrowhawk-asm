//! Some docs should be here

use core::panic;
use nohash_hasher::NoHashHasher;
use std::{collections::HashMap, hash::BuildHasherDefault};

#[cfg(not(target_family = "wasm"))]
use std::path::PathBuf;

#[cfg(not(target_family = "wasm"))]
use super::io_utils::*;
// use std::process::exit;

use crate::bit_encoding::UInt;
use crate::cli::DEFAULT_MIN_CONTIG_LENGTH_NTS;
use crate::graph_works::spell_path;
use crate::graph_works::Contigs;
#[cfg(target_family = "wasm")]
use crate::logw;

/// Writes the full sequence each contig's k-mer path spells, and hopefully their average
/// counts/coverage in the future.
pub fn write_sequences_and_coverages<IntT>(
    invec: &mut Contigs,
    inmap: &HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    k: usize,
) where
    IntT: for<'a> UInt<'a>,
{
    write_sequences_and_coverages_with_min_contig_length(
        invec,
        inmap,
        k,
        DEFAULT_MIN_CONTIG_LENGTH_NTS,
    );
}

fn write_sequences_and_coverages_with_min_contig_length<IntT>(
    invec: &mut Contigs,
    inmap: &HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    k: usize,
    min_contig_length: usize,
) where
    IntT: for<'a> UInt<'a>,
{
    // TODO: implement coverages somehow...
    invec.contig_sequences = Some(Vec::with_capacity(invec.serialized_contigs.len()));
    let mut omitted = 0;

    for (ipc, contig) in invec.serialized_contigs.iter().enumerate() {
        if contig.len() > 1 {
            panic!("MORE THAN ONE ENTRY!!")
        };

        // A contig is always a walk, so a failure here is an upstream orientation bug, not a bad base.
        let full = spell_path(&contig[0].abs_ind, inmap, k)
            .unwrap_or_else(|e| panic!("contig {ipc} is not a valid walk: {e}"));

        if full.len() >= min_contig_length {
            invec.contig_sequences.as_mut().unwrap().push(full);
        } else {
            omitted += 1;
        }
    }

    #[cfg(not(target_family = "wasm"))]
    log::info!("Omitted {omitted} contigs shorter than {min_contig_length} nt from FASTA output");
    #[cfg(target_family = "wasm")]
    logw(
        format!("Omitted {omitted} contigs shorter than {min_contig_length} nt from FASTA output")
            .as_str(),
        Some("info"),
    );
}

/// Stores all the contigs as a fasta file
#[cfg(not(target_family = "wasm"))]
pub fn save_as_fasta<IntT>(
    ingraph: &mut Contigs,
    inmap: &HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    k: usize,
    outfile: PathBuf,
) where
    IntT: for<'a> UInt<'a>,
{
    // First, we write the sequences and the coverages
    write_sequences_and_coverages(ingraph, inmap, k);

    log::debug!("Starting to save");
    log::debug!("{:?}", outfile);
    // Now, we just write all the contigs. We get our writing buffer with this:
    let mut wbuf = set_ostream(&Some(outfile.into_os_string().into_string().unwrap()));
    // And simply, contig per contig, we write the file

    log::debug!("\tLen.\tMean\tSD\tMedian");
    ingraph.write_fasta(&mut wbuf);
}

/// Stores all contigs at a configurable minimum length as a FASTA file.
#[cfg(not(target_family = "wasm"))]
pub(crate) fn save_as_fasta_with_min_contig_length<IntT>(
    ingraph: &mut Contigs,
    inmap: &HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    k: usize,
    min_contig_length: usize,
    outfile: PathBuf,
) where
    IntT: for<'a> UInt<'a>,
{
    write_sequences_and_coverages_with_min_contig_length(ingraph, inmap, k, min_contig_length);

    log::debug!("Starting to save");
    log::debug!("{:?}", outfile);
    let mut wbuf = set_ostream(&Some(outfile.into_os_string().into_string().unwrap()));
    log::debug!("\tLen.\tMean\tSD\tMedian");
    ingraph.write_fasta(&mut wbuf);
}

/// Writes all the contigs as a fasta file
#[cfg(not(target_family = "wasm"))]
pub fn write_as_fasta<IntT, W>(
    ingraph: &mut Contigs,
    inmap: &HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    k: usize,
    writer: &mut W,
) where
    IntT: for<'a> UInt<'a>,
    W: std::io::Write,
{
    write_sequences_and_coverages(ingraph, inmap, k);
    ingraph.write_fasta(writer);
}

/// Stores all the contigs in fasta format, but exports it as JSON for javascript
#[cfg(target_family = "wasm")]
pub fn save_as_fasta_wasm<IntT>(
    ingraph: &mut Contigs,
    inmap: &HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    k: usize,
) -> String
where
    IntT: for<'a> UInt<'a>,
{
    // First, we write the sequences and the coverages
    logw("Preparing to export contigs...", Some("info"));

    write_sequences_and_coverages(ingraph, inmap, k);

    logw("Saving in FASTA format as a JSON", Some("info"));
    // Now, we just write all the contigs. We get our writing buffer with this:
    let mut out = "".to_string();
    let mut tmpvec: Vec<u8> = Vec::with_capacity(80);
    let mut tmpcounter: usize;

    for i in 0..ingraph.contig_sequences.as_ref().unwrap().len() {
        out += (">".to_owned() + i.to_string().as_str() + "\n").as_str();
        tmpcounter = 0;
        tmpvec.clear();

        for j in &ingraph.contig_sequences.as_ref().unwrap()[i][..] {
            tmpvec.push(*j);
            tmpcounter += 1;
            if tmpcounter >= 80 {
                out += &(String::from_utf8(tmpvec.clone()).unwrap() + "\n");
                tmpcounter = 0;
                tmpvec.clear();
            }
        }

        if !tmpvec.is_empty() {
            out += &(String::from_utf8(tmpvec.clone()).unwrap() + "\n");
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_works::Contigs;
    use crate::kmer::Kmer;
    use sparrowhawk_graph::NodeStruct;
    use std::borrow::Cow;

    /// A deterministic ACGT sequence. An LCG keeps it reproducible without pulling in a rng crate.
    fn pseudo_seq(len: usize) -> Vec<u8> {
        const BASES: [u8; 4] = [b'A', b'C', b'G', b'T'];
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

    /// The (dict, abs_ind) pair preprocessing would produce: canonical hash -> packed k-mer, plus the
    /// ordered hashes along the walk.
    fn dict_and_path(
        seq: &[u8],
        k: usize,
    ) -> (
        HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
        Vec<u64>,
    ) {
        let mut dict: HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>> = HashMap::default();
        let mut path: Vec<u64> = Vec::new();

        let mut it = Kmer::<u64>::new(Cow::Borrowed(seq), seq.len(), None, k, 0, true).unwrap();
        let (hc, _, _, km) = it.get_curr_kmerhash_and_bases_and_kmer();
        dict.insert(hc, km);
        path.push(hc);
        while let Some((hc, _, _, km)) = it.get_next_kmer_and_give_us_things() {
            dict.insert(hc, km);
            path.push(hc);
        }
        (dict, path)
    }

    /// A contig is the whole sequence its k-mer path spells, so it must come back as `S` itself.
    #[test]
    fn contig_preserves_the_full_spelled_path() {
        let k = 31;
        let seq = pseudo_seq(300);
        let n = seq.len() - k + 1; // number of k-mers
        let (dict, path) = dict_and_path(&seq, k);
        assert_eq!(path.len(), n);

        let mut contigs = Contigs::new(vec![vec![NodeStruct {
            counts: 1,
            abs_ind: path,
            innerdir: None,
        }]]);
        write_sequences_and_coverages_with_min_contig_length(&mut contigs, &dict, k, 0);

        let got = &contigs.contig_sequences.as_ref().unwrap()[0];
        assert_eq!(got.len(), seq.len(), "contig length");
        assert_eq!(got, &seq, "contig sequence");
    }

    /// Both terminal k-mers survive. Guards the end-symmetry the old trim had to be fixed for: an
    /// off-by-one at either end would clip exactly one of these.
    #[test]
    fn contig_preserves_both_terminal_kmers() {
        let k = 31;
        let seq = pseudo_seq(300);
        let (dict, path) = dict_and_path(&seq, k);

        let mut contigs = Contigs::new(vec![vec![NodeStruct {
            counts: 1,
            abs_ind: path,
            innerdir: None,
        }]]);
        write_sequences_and_coverages_with_min_contig_length(&mut contigs, &dict, k, 0);

        let got = &contigs.contig_sequences.as_ref().unwrap()[0];
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

        let mut contigs = Contigs::new(vec![vec![NodeStruct {
            counts: 1,
            abs_ind: path,
            innerdir: None,
        }]]);
        write_sequences_and_coverages_with_min_contig_length(&mut contigs, &dict, k, 0);

        let sequences = contigs.contig_sequences.as_ref().unwrap();
        assert_eq!(sequences.len(), 1, "the short path must still be written");
        assert_eq!(sequences[0], seq);
    }

    #[test]
    fn minimum_contig_length_is_inclusive_and_configurable() {
        let k = 31;
        let minimum = 500;

        // The threshold applies to the sequence written, not to a trimmed body, so it means what it says.
        for (sequence_length, expected_contigs) in [(500, 1), (499, 0)] {
            let seq = pseudo_seq(sequence_length);
            let (dict, path) = dict_and_path(&seq, k);
            let mut contigs = Contigs::new(vec![vec![NodeStruct {
                counts: 1,
                abs_ind: path,
                innerdir: None,
            }]]);

            write_sequences_and_coverages_with_min_contig_length(&mut contigs, &dict, k, minimum);

            let sequences = contigs.contig_sequences.as_ref().unwrap();
            assert_eq!(sequences.len(), expected_contigs);
            if expected_contigs == 1 {
                assert_eq!(sequences[0].len(), sequence_length);
            }
        }
    }
}
