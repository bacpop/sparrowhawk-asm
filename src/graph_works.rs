use nohash_hasher::NoHashHasher;
use std::{collections::HashMap, hash::BuildHasherDefault};

#[cfg(not(target_arch = "wasm32"))]
use std::{io::Write, path::PathBuf, time::Instant};

#[cfg(not(target_arch = "wasm32"))]
use super::io_utils::*;

#[cfg(not(target_arch = "wasm32"))]
use needletail::parser::write_fasta;

use crate::algorithms::collapser::Collapsable;
use crate::algorithms::corrector::Correctable;
use crate::algorithms::shrinker::Shrinkable;
use crate::nthash;

use crate::bit_encoding::rc_base;
use crate::logw;
#[cfg(target_arch = "wasm32")]
use crate::post_state;

use sphk_graph::{DbgGraph, EdgeType, HashInfoSimple, SerializedContigs};

/// Get backwards neighbours, i.e. incoming edges to either the canonical or non-canonical hashes
pub fn check_bkg(
    // backwards here mean INCOMING edges, whether from the rev.-comp. strand, or the direct one
    hc: u64,
    hnc: u64,
    k: usize,
    bases: u8,
    thedict: &HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
    maxmindict: &HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
) -> Vec<(u64, EdgeType)> {
    let mut outvec = Vec::new();
    let thecbase = bases & 3;
    let thencbase = (bases >> 2) & 3;
    for i in 0..4 {
        let tmphashc = nthash::swapbits_18_31_42_51_58_63(
            (hc ^ nthash::HASH_LOOKUP[thecbase as usize]
                ^ (nthash::MS_TAB_5LL[(i as usize * 5) + (k % 5)]
                    | nthash::MS_TAB_7L[(i as usize * 7) + (k % 7)]
                    | nthash::MS_TAB_9LC[(i as usize * 9) + (k % 9)]
                    | nthash::MS_TAB_11CR[(i as usize * 11) + (k % 11)]
                    | nthash::MS_TAB_13R[(i as usize * 13) + (k % 13)]
                    | nthash::MS_TAB_19RR[(i as usize * 19) + (k % 19)]))
                .rotate_right(1u32),
        );

        if thedict.contains_key(&tmphashc) {
            outvec.push((tmphashc, EdgeType::MinToMin));
        } else {
            let poth = maxmindict.get(&tmphashc);
            if poth.is_some_and(|x| thedict.contains_key(x)) {
                outvec.push((*poth.unwrap(), EdgeType::MaxToMin));
            }
        }

        let mut tmphashnc = hnc
            ^ (nthash::MS_TAB_5LL[(rc_base(i) as usize * 5) + (k % 5)]
                | nthash::MS_TAB_7L[(rc_base(i) as usize * 7) + (k % 7)]
                | nthash::MS_TAB_9LC[(rc_base(i) as usize * 9) + (k % 9)]
                | nthash::MS_TAB_11CR[(rc_base(i) as usize * 11) + (k % 11)]
                | nthash::MS_TAB_13R[(rc_base(i) as usize * 13) + (k % 13)]
                | nthash::MS_TAB_19RR[(rc_base(i) as usize * 19) + (k % 19)]);
        tmphashnc ^= nthash::RC_HASH_LOOKUP[thencbase as usize];
        tmphashnc = tmphashnc.rotate_right(1_u32);
        tmphashnc = nthash::swapbits_18_31_42_51_58_63(tmphashnc);

        if thedict.contains_key(&tmphashnc) {
            outvec.push((tmphashnc, EdgeType::MinToMax));
        } else {
            let poth = maxmindict.get(&tmphashnc);
            if poth.is_some_and(|x| thedict.contains_key(x)) {
                outvec.push((*poth.unwrap(), EdgeType::MaxToMax));
            }
        }
    }

    outvec
}

/// Get forward neighbours, i.e. outgoing edges from either the canonical or non-canonical hashes
pub fn check_fwd(
    hc: u64,
    hnc: u64,
    k: usize,
    bases: u8,
    thedict: &HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
    maxmindict: &HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
) -> Vec<(u64, EdgeType)> {
    let mut outvec = Vec::new();

    let thecbase = (bases >> 2) & 3;
    let thencbase = bases & 3;
    for i in 0..4 {
        let mut tmphashc = hc.rotate_left(1);
        tmphashc = nthash::swapbits_0_19_32_43_52_59(tmphashc);
        tmphashc ^= nthash::HASH_LOOKUP[i as usize];
        tmphashc ^= nthash::MS_TAB_5LL[(thecbase as usize * 5) + (k % 5)]
            | nthash::MS_TAB_7L[(thecbase as usize * 7) + (k % 7)]
            | nthash::MS_TAB_9LC[(thecbase as usize * 9) + (k % 9)]
            | nthash::MS_TAB_11CR[(thecbase as usize * 11) + (k % 11)]
            | nthash::MS_TAB_13R[(thecbase as usize * 13) + (k % 13)]
            | nthash::MS_TAB_19RR[(thecbase as usize * 19) + (k % 19)];

        if thedict.contains_key(&tmphashc) {
            outvec.push((tmphashc, EdgeType::MinToMin));
        } else {
            let poth = maxmindict.get(&tmphashc);
            if poth.is_some_and(|x| thedict.contains_key(x)) {
                outvec.push((*poth.unwrap(), EdgeType::MinToMax));
            }
        }

        let tmphashnc = nthash::swapbits_0_19_32_43_52_59(hnc.rotate_left(1u32))
            ^ nthash::RC_HASH_LOOKUP[i as usize]
            ^ (nthash::MS_TAB_5LL[(rc_base(thencbase) as usize * 5) + (k % 5)]
                | nthash::MS_TAB_7L[(rc_base(thencbase) as usize * 7) + (k % 7)]
                | nthash::MS_TAB_9LC[(rc_base(thencbase) as usize * 9) + (k % 9)]
                | nthash::MS_TAB_11CR[(rc_base(thencbase) as usize * 11) + (k % 11)]
                | nthash::MS_TAB_13R[(rc_base(thencbase) as usize * 13) + (k % 13)]
                | nthash::MS_TAB_19RR[(rc_base(thencbase) as usize * 19) + (k % 19)]);

        if thedict.contains_key(&tmphashnc) {
            outvec.push((tmphashnc, EdgeType::MaxToMin));
        } else {
            let poth = maxmindict.get(&tmphashnc);
            if poth.is_some_and(|x| thedict.contains_key(x)) {
                outvec.push((*poth.unwrap(), EdgeType::MaxToMax));
            }
        }
    }

    outvec
}

////////////////////////////////////////////////////////////////////////
/// Output from the assembler.
#[derive(Default)]
pub struct Contigs {
    /// Serialized contigs.
    pub serialized_contigs: SerializedContigs,

    /// Sequence of the contigs.
    pub contig_sequences: Option<Vec<Vec<u8>>>,
}

impl Contigs {
    /// Create new `Contigs`.
    pub fn new(serialized: SerializedContigs) -> Contigs {
        Contigs {
            serialized_contigs: serialized,
            contig_sequences: None,
        }
    }

    /// Temporal and historical function to simplify contigs. To be removed in the future
    pub fn shrink(&mut self) {
        for ic in 0..self.serialized_contigs.len() {
            let mut tmpv = self.serialized_contigs[ic][0].abs_ind.clone();
            let contiglen = self.serialized_contigs[ic].len();
            for j in 1..contiglen {
                tmpv.extend(self.serialized_contigs[ic][j].abs_ind.clone());
            }
            self.serialized_contigs[ic].first_mut().unwrap().abs_ind = tmpv;
            self.serialized_contigs[ic].drain(1..);
        }
    }

    /// Save contigs in a file
    #[cfg(not(target_arch = "wasm32"))]
    pub fn write_fasta<W: Write>(&self, f: &mut W) {
        for i in 0..self.contig_sequences.as_ref().unwrap().len() {
            let _ = write_fasta(
                i.to_string().as_bytes(),
                &self.contig_sequences.as_ref().unwrap()[i][..],
                f,
                needletail::parser::LineEnding::Unix,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kmer::Kmer;
    use nohash_hasher::NoHashHasher;
    use std::{borrow::Cow, collections::HashMap, hash::BuildHasherDefault};

    fn empty_thedict(
    ) -> HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>> {
        HashMap::with_hasher(BuildHasherDefault::default())
    }

    fn empty_maxmindict() -> HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>> {
        HashMap::with_hasher(BuildHasherDefault::default())
    }

    #[test]
    fn check_fwd_empty_dict_returns_empty() {
        let dict = empty_thedict();
        let maxmin = empty_maxmindict();
        assert!(check_fwd(0xDEAD_BEEF, 0xCAFE_BABE, 3, 0b0101, &dict, &maxmin).is_empty());
    }

    #[test]
    fn check_bkg_empty_dict_returns_empty() {
        let dict = empty_thedict();
        let maxmin = empty_maxmindict();
        assert!(check_bkg(0xDEAD_BEEF, 0xCAFE_BABE, 3, 0b0101, &dict, &maxmin).is_empty());
    }

    #[test]
    fn check_fwd_finds_consecutive_kmer() {
        let seq = b"ACGTACGTACGT";
        let k = 5;
        let mut it =
            Kmer::<u64>::new(Cow::Borrowed(seq.as_slice()), seq.len(), None, k, 0, true).unwrap();
        let (hc1, hnc1, b1, _) = it.get_curr_kmerhash_and_bases_and_kmer();
        let (hc2, hnc2, b2, _) = it.get_next_kmer_and_give_us_things().unwrap();

        let mut dict = empty_thedict();
        dict.insert(hc2, HashInfoSimple {
            hnc: hnc2,
            b: b2,
            pre: vec![],
            post: vec![],
            counts: 1,
        });
        let mut maxmin = empty_maxmindict();
        maxmin.insert(hnc2, hc2);

        let result = check_fwd(hc1, hnc1, k, b1, &dict, &maxmin);
        assert!(!result.is_empty(), "check_fwd should find the consecutive kmer");
        assert!(result.iter().any(|(h, _)| *h == hc2));
    }

    #[test]
    fn check_bkg_finds_preceding_kmer() {
        let seq = b"ACGTACGTACGT";
        let k = 5;
        let mut it =
            Kmer::<u64>::new(Cow::Borrowed(seq.as_slice()), seq.len(), None, k, 0, true).unwrap();
        let (hc1, hnc1, b1, _) = it.get_curr_kmerhash_and_bases_and_kmer();
        let (hc2, hnc2, b2, _) = it.get_next_kmer_and_give_us_things().unwrap();

        let mut dict = empty_thedict();
        dict.insert(hc1, HashInfoSimple {
            hnc: hnc1,
            b: b1,
            pre: vec![],
            post: vec![],
            counts: 1,
        });
        let mut maxmin = empty_maxmindict();
        maxmin.insert(hnc1, hc1);

        let result = check_bkg(hc2, hnc2, k, b2, &dict, &maxmin);
        assert!(!result.is_empty(), "check_bkg should find the preceding kmer");
        assert!(result.iter().any(|(h, _)| *h == hc1));
    }
}

/// Public API for assemblers.
pub trait Assemble {
    #[cfg(not(target_arch = "wasm32"))]
    /// Assembles given data and writes results into the output file.
    fn assemble(
        k: usize,
        indict: &mut HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
        maxminsize: &mut HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
        timevec: &mut Vec<Instant>,
        path: &mut Option<PathBuf>,
        do_bubble_collapse: bool,
        do_dead_end_removal: bool,
        do_conflictive_links_removal: bool,
    ) -> Contigs;

    #[cfg(target_arch = "wasm32")]
    /// Assembles given data and prepares all info for being later transferred to Javascript.
    fn assemble_wasm(
        k: usize,
        indict: &mut HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
        maxminsize: &mut HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
        do_bubble_collapse: bool,
        do_dead_end_removal: bool,
        do_conflictive_links_removal: bool,
    ) -> (Contigs, String, String, String);
}

///////////////////////////////////////////////////////////
/// Basic standalone assembler.
pub struct BasicAsm {}

impl Assemble for BasicAsm {
    #[cfg(not(target_arch = "wasm32"))]
    fn assemble(
        k: usize,
        indict: &mut HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
        maxmindict: &mut HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
        timevec: &mut Vec<Instant>,
        path: &mut Option<PathBuf>,
        do_bubble_collapse: bool,
        do_dead_end_removal: bool,
        do_conflictive_links_removal: bool,
    ) -> Contigs {
        logw(
            "Constructing graph. Searching for neighbours...",
            Some("info"),
        );
        timevec.push(Instant::now());

        let updates: Vec<(u64, Vec<(u64, EdgeType)>, Vec<(u64, EdgeType)>)> = indict
            .iter()
            .map(|(h, hi)| {
                let pre = check_bkg(*h, hi.hnc, k, hi.b, indict, maxmindict);
                let post = check_fwd(*h, hi.hnc, k, hi.b, indict, maxmindict);
                (*h, pre, post)
            })
            .collect();
        for (h, pre, post) in updates {
            let entry = indict.get_mut(&h).unwrap();
            entry.pre = pre;
            entry.post = post;
        }

        timevec.push(Instant::now());
        logw(
            format!(
                "Neighbours searched for in {} s. Creating graph...",
                timevec
                    .last()
                    .unwrap()
                    .duration_since(*timevec.get(timevec.len().wrapping_sub(2)).unwrap())
                    .as_secs()
            )
            .as_str(),
            Some("info"),
        );

        let mut ptgraph = DbgGraph::from_kmer_map(k, indict);

        timevec.push(Instant::now());
        logw(
            format!(
                "Graph created in {} s.",
                timevec
                    .last()
                    .unwrap()
                    .duration_since(*timevec.get(timevec.len().wrapping_sub(2)).unwrap())
                    .as_secs()
            )
            .as_str(),
            Some("info"),
        );

        logw("Starting graph correction", Some("info"));

        logw("Removing self-loops", Some("info"));
        ptgraph.remove_self_loops();

        if do_conflictive_links_removal {
            logw("Removing conflictive links", Some("info"));
            ptgraph.remove_conflictive_links();
        }

        let mut didanyofusdoanything = true;
        let mut bool1: bool;
        let mut bool2: bool = false;
        let mut bool3: bool = false;
        let mut bool4: bool = false;
        while didanyofusdoanything {
            bool1 = ptgraph.shrink();

            if do_dead_end_removal {
                bool2 = ptgraph.remove_dead_paths();
                bool3 = ptgraph.shrink();
            }

            if do_bubble_collapse {
                bool4 = ptgraph.correct_bubbles();
            }

            didanyofusdoanything = bool1 || bool2 || bool3 || bool4;
        }

        timevec.push(Instant::now());
        logw(
            format!(
                "Graph correction finished in {} s.",
                timevec
                    .last()
                    .unwrap()
                    .duration_since(*timevec.get(timevec.len().wrapping_sub(2)).unwrap())
                    .as_secs()
            )
            .as_str(),
            Some("info"),
        );

        if path.is_some() {
            logw("Saving graph (post-shrink, pre-collapse, w/o one-node contigs) as DOT, GFAv1.1, and GFAv2 files...", Some("info"));
            let pathmutref = path.as_mut().unwrap();
            pathmutref.set_extension("dot");
            let mut wbufdot = set_ostream(&Some(
                pathmutref.clone().into_os_string().into_string().unwrap(),
            ));
            ptgraph.write_to_dot(&mut wbufdot);

            pathmutref.set_extension("gfa");
            let mut wbufgfa = set_ostream(&Some(
                pathmutref.clone().into_os_string().into_string().unwrap(),
            ));
            ptgraph.write_to_gfa(&mut wbufgfa);

            pathmutref.set_extension("gfa2");
            let mut wbufgfa2 = set_ostream(&Some(
                pathmutref.clone().into_os_string().into_string().unwrap(),
            ));
            ptgraph.write_to_gfa2(&mut wbufgfa2);
            logw("Done.", Some("info"));
        }

        timevec.push(Instant::now());
        let serialized_contigs = ptgraph.collapse();
        timevec.push(Instant::now());
        logw(
            format!(
                "Graph collapse finished in {} s.",
                timevec
                    .last()
                    .unwrap()
                    .duration_since(*timevec.get(timevec.len().wrapping_sub(2)).unwrap())
                    .as_secs()
            )
            .as_str(),
            Some("info"),
        );
        logw(
            format!("I created {} contigs", serialized_contigs.len()).as_str(),
            Some("info"),
        );

        let mut contigs = Contigs::new(serialized_contigs);

        // TEMPORAL RESTRICTION, WIP
        contigs.shrink();

        contigs
    }

    #[cfg(target_arch = "wasm32")]
    fn assemble_wasm(
        k: usize,
        indict: &mut HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>>,
        maxmindict: &mut HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
        do_bubble_collapse: bool,
        do_dead_end_removal: bool,
        do_conflictive_links_removal: bool,
    ) -> (Contigs, String, String, String) {
        logw("Starting assembler!", Some("info"));

        post_state("assembly:starting");
        let mut i = 0;
        let mut ialone = 0;
        let mut nedges = 0;

        post_state("assembly:create_graph");
        let updates: Vec<(u64, Vec<(u64, EdgeType)>, Vec<(u64, EdgeType)>)> = indict
            .iter()
            .map(|(h, hi)| {
                let pre = check_bkg(*h, hi.hnc, k, hi.b, indict, maxmindict);
                let post = check_fwd(*h, hi.hnc, k, hi.b, indict, maxmindict);
                (*h, pre, post)
            })
            .collect();
        for (h, pre, post) in updates {
            let entry = indict.get_mut(&h).unwrap();
            nedges += entry.pre.len() + entry.post.len();
            i += 1;
            if entry.pre.is_empty() && entry.post.is_empty() {
                ialone += 1;
            }
            entry.pre = pre;
            entry.post = post;
        }

        logw(
            format!(
                "Prop. of alone kmers: {:.1} %",
                (ialone as f64) / (i as f64) * 100.0
            )
            .as_str(),
            Some("trace"),
        );
        logw(
            format!("Number of edges {}", (nedges as f64) / (2_f64)).as_str(),
            Some("trace"),
        );

        let mut ptgraph = DbgGraph::from_kmer_map(k, indict);

        post_state("assembly:correct_graph");
        logw("Starting graph correction", Some("info"));

        logw("Removing self-loops", Some("info"));
        ptgraph.remove_self_loops();

        if do_conflictive_links_removal {
            logw("Removing conflictive links", Some("info"));
            ptgraph.remove_conflictive_links();
        }

        let mut didanyofusdoanything = true;
        let mut bool1: bool;
        let mut bool2: bool = false;
        let mut bool3: bool = false;
        let mut bool4: bool = false;
        while didanyofusdoanything {
            bool1 = ptgraph.shrink();

            if do_dead_end_removal {
                bool2 = ptgraph.remove_dead_paths();
                bool3 = ptgraph.shrink();
            }

            if do_bubble_collapse {
                bool4 = ptgraph.correct_bubbles();
            }

            didanyofusdoanything = bool1 || bool2 || bool3 || bool4;
        }

        logw("Shrinkage and pruning finished", Some("info"));

        let outdot = ptgraph.get_dot_string();
        let outgfa = ptgraph.get_gfa_string();
        let outgfa2 = ptgraph.get_gfa2_string();

        post_state("assembly:collapse_graph");

        let serialized_contigs = ptgraph.collapse();
        logw(
            format!("I created {} contigs", serialized_contigs.len()).as_str(),
            Some("info"),
        );
        let mut contigs = Contigs::new(serialized_contigs);

        // TEMPORAL RESTRICTION, WIP
        contigs.shrink();

        (contigs, outdot, outgfa, outgfa2)
    }
}
