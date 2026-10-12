//! Spelling contigs and writing them as FASTA.

use core::panic;

#[cfg(not(target_family = "wasm"))]
use std::path::PathBuf;

#[cfg(not(target_family = "wasm"))]
use super::io_utils::*;

use crate::bit_encoding::UInt;
#[cfg(target_family = "wasm")]
use crate::cli::DEFAULT_MIN_CONTIG_LENGTH_NTS;
use crate::graph_works::{spell_path, Contigs, KmerLookup};
#[cfg(target_family = "wasm")]
use crate::logw;

/// Spells each contig and keeps those of at least `DEFAULT_MIN_CONTIG_LENGTH_NTS` bases.
#[cfg(target_family = "wasm")]
pub fn write_sequences_and_coverages<IntT>(
    invec: &mut Contigs,
    inmap: &impl KmerLookup<IntT>,
    k: usize,
) where
    IntT: for<'a> UInt<'a>,
{
    let sequences = spell_contigs(invec, inmap, k);
    let kept = at_least(&sequences, DEFAULT_MIN_CONTIG_LENGTH_NTS);
    logw(
        &format!(
            "Omitted {} contigs shorter than {DEFAULT_MIN_CONTIG_LENGTH_NTS} nt from FASTA output",
            sequences.len() - kept.len()
        ),
        Some("info"),
    );
    invec.contig_sequences = Some(kept);
}

/// Spells every contig. A multi-k step carries all of them to the next k, so none is length-filtered.
pub(crate) fn spell_contigs<IntT>(
    contigs: &Contigs,
    dict: &impl KmerLookup<IntT>,
    k: usize,
) -> Vec<Vec<u8>>
where
    IntT: for<'a> UInt<'a>,
{
    contigs
        .serialized_contigs
        .iter()
        .enumerate()
        .map(|(ipc, contig)| {
            assert_eq!(contig.len(), 1, "contig {ipc} was not shrunk to one node");
            spell_path(&contig[0].abs_ind, dict, k)
                .unwrap_or_else(|e| panic!("contig {ipc} is not a valid walk: {e}"))
        })
        .collect()
}

/// The contigs of at least `min_contig_length` bases (inclusive), the only ones written to FASTA.
pub(crate) fn at_least(seqs: &[Vec<u8>], min_contig_length: usize) -> Vec<Vec<u8>> {
    seqs.iter()
        .filter(|s| s.len() >= min_contig_length)
        .cloned()
        .collect()
}

/// Writes spelled contigs of at least `min_contig_length` bases as FASTA, as the single-k path does.
#[cfg(not(target_family = "wasm"))]
pub(crate) fn save_sequences_as_fasta(
    seqs: &[Vec<u8>],
    min_contig_length: usize,
    outfile: PathBuf,
) {
    let kept = at_least(seqs, min_contig_length);
    log::info!(
        "Omitted {} contigs shorter than {min_contig_length} nt from FASTA output",
        seqs.len() - kept.len()
    );
    let contigs = Contigs {
        serialized_contigs: Vec::new(),
        contig_sequences: Some(kept),
    };
    let mut wbuf = set_ostream(&Some(outfile.into_os_string().into_string().unwrap()));
    contigs.write_fasta(&mut wbuf);
}

/// Stores all the contigs in fasta format, but exports it as JSON for javascript
#[cfg(target_family = "wasm")]
pub fn save_as_fasta_wasm<IntT>(
    ingraph: &mut Contigs,
    inmap: &impl KmerLookup<IntT>,
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
        let (hc, _, _, km) = it.get_curr_kmerhash_and_bases_and_kmer();
        dict.insert(hc, km);
        path.push(hc);
        while let Some((hc, _, _, km)) = it.get_next_kmer_and_give_us_things() {
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

        let sequences = spell_contigs(&one_contig(path), &dict, k);
        let got = &sequences[0];
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

        let sequences = spell_contigs(&one_contig(path), &dict, k);
        let got = &sequences[0];
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

        let sequences = spell_contigs(&one_contig(path), &dict, k);
        assert_eq!(sequences.len(), 1, "the short path must still be written");
        assert_eq!(sequences[0], seq);
    }

    #[test]
    fn minimum_contig_length_is_inclusive_and_configurable() {
        // The threshold applies to the sequence written, not to a trimmed body, so it means what it says.
        let sequences = [pseudo_seq(500), pseudo_seq(499)];
        assert_eq!(at_least(&sequences, 500), vec![sequences[0].clone()]);
        assert_eq!(at_least(&sequences, 499).len(), 2);
    }
}
