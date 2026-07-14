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
use crate::spelling::spell_path;

/// Writes the contig sequences and hopefully their average counts/coverage in the future
///
/// Each contig is trimmed by `k-1` bases at **both** ends. Only the interior of a contig is flanked by
/// a k-mer on either side, so those are the only bases two independent k-mers agree on; the ends are
/// spelled by a single k-mer each and are dropped, as SKESA also does.
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

        // A contig is a walk through the graph, so it always spells. Anything else is a bug in the
        // orientation bookkeeping upstream, and we would rather hear about it than emit a wrong base:
        // this used to count the failures and push a base from the failing orientation regardless.
        let full = spell_path(&contig[0].abs_ind, inmap, k)
            .unwrap_or_else(|e| panic!("contig {ipc} is not a valid walk: {e}"));

        // `full` is the whole walk: n + k - 1 bases, for n k-mers. Keep only the bases with a k-mer on
        // each side, i.e. drop k-1 from each end, leaving n - k + 1. The guard is the old
        // `outseq.len() >= k` (which was in units of k-mers, so n >= k) written in bases.
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

    /// Turn a sequence into the (thedict, abs_ind) pair that `write_sequences_and_coverages` consumes,
    /// exactly as preprocessing would: `thedict` maps canonical hash -> canonical packed k-mer, and
    /// `abs_ind` is the ordered list of canonical hashes along the walk.
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

    /// A contig is spelled one nucleotide per k-mer (the *last* base of each), so it starts at S[k-1].
    /// We then trim k-1 from the tail, so the emitted contig must be exactly `S[k-1 .. n]`, where
    /// `n = |S| - k + 1` is the number of k-mers. Equivalently: k-1 bases dropped from each end.
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

    /// The old code trimmed k from the tail instead of k-1, making every contig one base short.
    /// Pin that down so it cannot regress.
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
