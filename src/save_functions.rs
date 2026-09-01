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
use crate::graph_works::Contigs;
#[cfg(target_family = "wasm")]
use crate::logw;
use crate::graph_works::spell_path;

/// Writes the contig sequences and hopefully their average counts/coverage in the future
///
/// Each contig is trimmed by `k-1` at both ends, keeping only bases two k-mers agree on, as SKESA does.
pub fn write_sequences_and_coverages<IntT>(
    invec: &mut Contigs,
    inmap: &HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    k: usize,
) where
    IntT: for<'a> UInt<'a>,
{
    // TODO: implement coverages somehow...
    invec.contig_sequences = Some(Vec::with_capacity(invec.serialized_contigs.len()));

    for (ipc, contig) in invec.serialized_contigs.iter().enumerate() {
        if contig.len() > 1 {
            panic!("MORE THAN ONE ENTRY!!")
        };

        // A contig is always a walk, so a failure here is an upstream orientation bug, not a bad base.
        let full = spell_path(&contig[0].abs_ind, inmap, k)
            .unwrap_or_else(|e| panic!("contig {ipc} is not a valid walk: {e}"));

        // `full` is n + k - 1 bases for n k-mers; drop k-1 from each end, leaving n - k + 1.
        if full.len() >= 2 * k - 1 {
            let body = &full[k - 1..full.len() - (k - 1)];
            if body.len() > 100 {
                invec
                    .contig_sequences
                    .as_mut()
                    .unwrap()
                    .push(body.to_vec());
            }
        }
    }
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

    /// Spelled one base per k-mer from S[k-1], then trimmed k-1 from the tail, so the contig must be
    /// exactly `S[k-1 .. n]` with `n = |S| - k + 1`.
    #[test]
    fn contig_is_trimmed_by_k_minus_one_at_each_end() {
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
        write_sequences_and_coverages(&mut contigs, &dict, k);

        let got = &contigs.contig_sequences.as_ref().unwrap()[0];
        let want = &seq[k - 1..n];
        assert_eq!(got.len(), want.len(), "contig length");
        assert_eq!(got, want, "contig sequence");

        // The trim is symmetric: k-1 gone from the front, k-1 gone from the back.
        assert_eq!(got.len(), seq.len() - 2 * (k - 1));
    }

    /// The last base must be the final one two k-mers confirm, `S[n-1]`.
    #[test]
    fn contig_keeps_the_final_confirmed_base() {
        let k = 31;
        let seq = pseudo_seq(300);
        let n = seq.len() - k + 1;
        let (dict, path) = dict_and_path(&seq, k);

        let mut contigs = Contigs::new(vec![vec![NodeStruct {
            counts: 1,
            abs_ind: path,
            innerdir: None,
        }]]);
        write_sequences_and_coverages(&mut contigs, &dict, k);

        let got = &contigs.contig_sequences.as_ref().unwrap()[0];
        assert_eq!(
            *got.last().unwrap(),
            seq[n - 1],
            "the last base must be S[n-1]; trimming k rather than k-1 would leave S[n-2]"
        );
    }
}
