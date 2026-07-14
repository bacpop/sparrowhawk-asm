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
use crate::logw;

/// Writes the contig sequences and hopefully their average counts/coverage in the future
pub fn write_sequences_and_coverages<IntT>(
    invec: &mut Contigs,
    inmap: &HashMap<u64, IntT, BuildHasherDefault<NoHashHasher<u64>>>,
    k: usize,
) where
    IntT: for<'a> UInt<'a>,
{
    // TODO: implement coverages somehow...
    invec.contig_sequences = Some(Vec::with_capacity(invec.serialized_contigs.len()));
    let mut counter = 0;
    for ipc in 0..invec.serialized_contigs.len() {
        //         log::debug!("\nIteration");
        //         let mut outseq : VecDeque<u8>  = VecDeque::new();
        let mut outseq: Vec<u8> = Vec::new();

        //         log::debug!("Initial index: {}", initind);
        // First of all, we decode and set the cov. for the first k nucleotides
        if invec.serialized_contigs[ipc].len() > 1 {
            panic!("MORE THAN ONE ENTRY!!")
        };
        let initkmer = inmap
            .get(&invec.serialized_contigs[ipc][0].abs_ind[0])
            .unwrap();
        let nbitstomove = initkmer.n_bits() as usize - 2 * (k - 1);
        //         log::debug!("nbits: {nbitstomove}");

        let mut prevkmer = *initkmer;
        if invec.serialized_contigs[ipc][0].abs_ind.len() > 1 {
            let tmpkmermoved = *inmap
                .get(&invec.serialized_contigs[ipc][0].abs_ind[1])
                .unwrap()
                >> 2;
            let tmpkmerrevmoved = inmap
                .get(&invec.serialized_contigs[ipc][0].abs_ind[1])
                .unwrap()
                .rev_comp(k)
                >> 2;
            let prevkmermoved = (prevkmer << nbitstomove) >> nbitstomove;
            //             log::debug!("Prev.              {:#066b}", prevkmer            );
            //             log::debug!("Prev. (rev.-comp.) {:#066b}", prevkmer.rev_comp(k));
            //             log::debug!("Post.              {:#066b}", tmpkmermoved);
            //             log::debug!("Post. (rev.-comp.) {:#066b}", tmpkmerrevmoved);
            //             log::debug!("Prev.              {:#066b}", prevkmermoved);

            if tmpkmermoved != prevkmermoved && tmpkmerrevmoved != prevkmermoved {
                // We need to add the first nucleotides from the rev. comp.
                prevkmer = prevkmer.rev_comp(k);
                //                 log::debug!("CHANGED");
            }

            if tmpkmermoved == prevkmermoved && tmpkmerrevmoved == prevkmermoved {
                panic!("HOLI");
            }
        }

        // for inc in 0..k {
        //     outseq.push( prevkmer.get_one_nucleotide(k - 1 - inc));
        // }

        outseq.push(prevkmer.get_one_nucleotide(0));

        // And now, we start the hard work with the remaining nucleotides
        let mut currkmer: IntT;
        let thelen = invec.serialized_contigs[ipc][0].abs_ind.len();
        for i in 1..thelen {
            //             log::debug!("Entry {}/{}", i + 1, thelen);
            currkmer = *inmap
                .get(&invec.serialized_contigs[ipc][0].abs_ind[i])
                .unwrap();

            if ((prevkmer << nbitstomove) >> nbitstomove) == (currkmer >> 2)
                && ((prevkmer << nbitstomove) >> nbitstomove) == (currkmer.rev_comp(k) >> 2)
            {
                panic!("HOLI");
            }

            //             if ((prevkmer.rev_comp(k) << nbitstomove) >> nbitstomove) == (currkmer >> 2) || ((prevkmer.rev_comp(k) << nbitstomove) >> nbitstomove) == currkmer.rev_comp(k) {
            //                 panic!("TEST2");
            //             }

            if ((prevkmer << nbitstomove) >> nbitstomove) != (currkmer >> 2) {
                //                 log::debug!("CAMBIANDO!");
                currkmer = currkmer.rev_comp(k);
                if ((prevkmer << (nbitstomove)) >> nbitstomove) != (currkmer >> 2) {
                    //                     log::debug!("BAD THING");
                    //                     log::debug!("Prev.:              {:#066b}", (prevkmer << (nbitstomove)) >> nbitstomove);
                    //                     log::debug!("Prev. (rev.-comp.): {:#066b}", (prevkmer.rev_comp(k) << (nbitstomove)) >> nbitstomove);
                    //                     log::debug!("Post:               {:#066b}", currkmer.rev_comp(k) >> 2);
                    //                     log::debug!("Post. (rev.-comp.): {:#066b}", currkmer >> 2);
                    //                     log::debug!("Prev. hash: {}",   invec.serialized_contigs[ipc][0].abs_ind[i - 1]);
                    //                     log::debug!("Post. hash: {}\n", invec.serialized_contigs[ipc][0].abs_ind[i]);
                    counter += 1;
                }
            }

            outseq.push(currkmer.get_one_nucleotide(0));
            prevkmer = currkmer;
        }

        // We spell one nucleotide per k-mer, taking the last base of each, so the leading k-1 bases of
        // the contig are already absent. Trim the same k-1 from the tail, so that every base we emit is
        // flanked by a k-mer on both sides. Trimming k here (as we used to) is one base too many.
        if outseq.len() >= k {
            outseq.truncate(outseq.len() - (k - 1));
            if outseq.len() > 100 {
                invec.contig_sequences.as_mut().unwrap().push(outseq);
            }
        }
    }

    logw(
        format!("\nNUMBER OF BAD THINGS: {}\n", counter).as_str(),
        Some("debug"),
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
