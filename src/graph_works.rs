use nohash_hasher::NoHashHasher;
use std::{cell::*, collections::HashMap, hash::BuildHasherDefault};

#[cfg(not(feature = "wasm"))]
use std::{io::Write, path::PathBuf, time::Instant};

// use rayon::prelude::*;

#[cfg(not(feature = "wasm"))]
use super::io_utils::*;

#[cfg(not(feature = "wasm"))]
use needletail::parser::write_fasta;

// use std::process::exit;
extern crate petgraph;
use super::HashInfoSimple;

use crate::algorithms::collapser::SerializedContigs;
use crate::graphs::pt_graph::EdgeType;
use crate::graphs::Graph;
use crate::nthash;

use crate::bit_encoding::rc_base;
use crate::logw;
#[cfg(feature = "wasm")]
use crate::post_state;

/// Get backwards neighbours, i.e. incoming edges to either the canonical or non-canonical hashes
pub fn check_bkg(
    // backwards here mean INCOMING edges, whether from the rev.-comp. strand, or the direct one
    hc: u64,
    hnc: u64,
    k: usize,
    bases: u8,
    thedict: &HashMap<u64, RefCell<HashInfoSimple>, BuildHasherDefault<NoHashHasher<u64>>>,
    maxmindict: &HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
) -> Vec<(u64, EdgeType)> {
    let mut outvec = Vec::new();
    // Let's start first with the canonical one
    let thecbase = bases & 3;
    let thencbase = (bases >> 2) & 3;
    for i in 0..4 {
        //ACTG, in that order
        // let tmphashc = (nthash::swapbits033(hc ^ nthash::HASH_LOOKUP[thecbase as usize]
        //                                        ^ (nthash::MS_TAB_31L[(i as usize * 31) + (k % 31)]
        //                                         | nthash::MS_TAB_33R[(i as usize) * 33 + (k % 33)])
        //                 )).rotate_right(1u32);

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
        // let tmphashc = (nthash::swapbits_0_19_32_43_52_59(hc ^ nthash::HASH_LOOKUP[thecbase as usize]
        //                                        ^ (nthash::MS_TAB_5LL[( i as usize * 5)  + (k % 5)]
        //                                         | nthash::MS_TAB_7L[(  i as usize * 7)  + (k % 7)]
        //                                         | nthash::MS_TAB_9LC[( i as usize * 9)  + (k % 9)]
        //                                         | nthash::MS_TAB_11CR[(i as usize * 11) + (k % 11)]
        //                                         | nthash::MS_TAB_13R[( i as usize * 13) + (k % 13)]
        //                                         | nthash::MS_TAB_19RR[(i as usize * 19) + (k % 19)])
        // )).rotate_right(1u32);

        if thedict.contains_key(&tmphashc) {
            outvec.push((tmphashc, EdgeType::MinToMin));
        } else {
            let poth = maxmindict.get(&tmphashc);
            if poth.is_some_and(|x| thedict.contains_key(x)) {
                outvec.push((*poth.unwrap(), EdgeType::MaxToMin));
            }
        }

        // let mut tmphashnc = hnc
        //     ^ (nthash::MS_TAB_31L[(rc_base(i) as usize  * 31) + (k % 31)]
        //     |  nthash::MS_TAB_33R[(rc_base(i) as usize) * 33  + (k % 33)]);
        // tmphashnc ^= nthash::RC_HASH_LOOKUP[thencbase as usize];
        // tmphashnc = tmphashnc.rotate_right(1_u32);
        // tmphashnc = nthash::swapbits3263(tmphashnc);
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
    // Here FORWARD means OUTGOING
    hc: u64,
    hnc: u64,
    k: usize,
    bases: u8,
    thedict: &HashMap<u64, RefCell<HashInfoSimple>, BuildHasherDefault<NoHashHasher<u64>>>,
    maxmindict: &HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
) -> Vec<(u64, EdgeType)> {
    let mut outvec = Vec::new();

    // Let's start first with the canonical one
    let thecbase = (bases >> 2) & 3;
    let thencbase = bases & 3;
    for i in 0..4 {
        //ACTG, in that order
        let mut tmphashc = hc.rotate_left(1);
        // tmphashc =  nthash::swapbits033(tmphashc);
        // tmphashc ^= nthash::HASH_LOOKUP[i as usize];
        // tmphashc ^= nthash::MS_TAB_31L[(thecbase as usize * 31) + (k % 31)]
        //           | nthash::MS_TAB_33R[(thecbase as usize) * 33 + (k % 33)];
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

        // let tmphashnc = (nthash::swapbits3263(hnc)).rotate_left(1u32)
        //     ^ nthash::RC_HASH_LOOKUP[i as usize]
        //     ^ (nthash::MS_TAB_31L[(rc_base(thencbase) as usize * 31) + (k % 31)]
        //      | nthash::MS_TAB_33R[(rc_base(thencbase) as usize) * 33 + (k % 33)]);
        let tmphashnc = nthash::swapbits_0_19_32_43_52_59(hnc.rotate_left(1u32))
            ^ nthash::RC_HASH_LOOKUP[i as usize]
            ^ (nthash::MS_TAB_5LL[(rc_base(thencbase) as usize * 5) + (k % 5)]
                | nthash::MS_TAB_7L[(rc_base(thencbase) as usize * 7) + (k % 7)]
                | nthash::MS_TAB_9LC[(rc_base(thencbase) as usize * 9) + (k % 9)]
                | nthash::MS_TAB_11CR[(rc_base(thencbase) as usize * 11) + (k % 11)]
                | nthash::MS_TAB_13R[(rc_base(thencbase) as usize * 13) + (k % 13)]
                | nthash::MS_TAB_19RR[(rc_base(thencbase) as usize * 19) + (k % 19)]);
        // let tmphashnc = (nthash::swapbits_18_31_42_51_58_63(hnc)).rotate_left(1u32)
        //     ^ nthash::RC_HASH_LOOKUP[i as usize]
        //     ^ (  nthash::MS_TAB_5LL[( rc_base(thencbase) as usize * 5)  + (k % 5)]
        //        | nthash::MS_TAB_7L[(  rc_base(thencbase) as usize * 7)  + (k % 7)]
        //        | nthash::MS_TAB_9LC[( rc_base(thencbase) as usize * 9)  + (k % 9)]
        //        | nthash::MS_TAB_11CR[(rc_base(thencbase) as usize * 11) + (k % 11)]
        //        | nthash::MS_TAB_13R[( rc_base(thencbase) as usize * 13) + (k % 13)]
        //        | nthash::MS_TAB_19RR[(rc_base(thencbase) as usize * 19) + (k % 19)]);

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
        // log::warn!("Contigs are going to be shrunk!");

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
    #[cfg(not(feature = "wasm"))]
    pub fn write_fasta<W: Write>(&self, f: &mut W) {
        //         self.print_coverage_stats();
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

/// Public API for assemblers.
pub trait Assemble {
    #[cfg(not(feature = "wasm"))]
    /// Assembles given data using specified `Graph` and writes results into the output file.
    fn assemble<G: Graph>(
        k: usize,
        indict: &mut HashMap<u64, RefCell<HashInfoSimple>, BuildHasherDefault<NoHashHasher<u64>>>,
        maxminsize: &mut HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
        timevec: &mut Option<&mut Vec<Instant>>,
        path: &mut Option<PathBuf>,
        do_bubble_collapse: bool,
        do_dead_end_removal: bool,
        do_conflictive_links_removal: bool,
    ) -> Contigs;

    #[cfg(feature = "wasm")]
    /// Assembles given data using specified `Graph` and prepares all info for being later transferred to Javascript.
    fn assemble_wasm<G: Graph>(
        k: usize,
        indict: &mut HashMap<u64, RefCell<HashInfoSimple>, BuildHasherDefault<NoHashHasher<u64>>>,
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
    #[cfg(not(feature = "wasm"))]
    fn assemble<G: Graph>(
        k: usize,
        indict: &mut HashMap<u64, RefCell<HashInfoSimple>, BuildHasherDefault<NoHashHasher<u64>>>,
        maxmindict: &mut HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
        timevec: &mut Option<&mut Vec<Instant>>,
        path: &mut Option<PathBuf>,
        do_bubble_collapse: bool,
        do_dead_end_removal: bool,
        do_conflictive_links_removal: bool,
    ) -> Contigs {
        logw(
            "Constructing graph. Searching for neighbours...",
            Some("info"),
        );
        timevec.as_mut().map(|timevec| timevec.push(Instant::now()));

        // FIRST: iterate over all k-mers, check the existance of forwards/backwards neighbours in the dictionary.
        // let mut i = 0;
        // let mut ialone = 0;
        // let mut nedges = 0;
        indict.iter().for_each(|(h, hi)| {
            let mut himutref = hi.borrow_mut();

            himutref.pre = check_bkg(*h, himutref.hnc, k, himutref.b, indict, maxmindict);
            himutref.post = check_fwd(*h, himutref.hnc, k, himutref.b, indict, maxmindict);
            // let tmpnedges = himutref.pre.len() + himutref.post.len();
            // nedges += tmpnedges;
            // i += 1;
            // if tmpnedges == 0 { ialone += 1};
        });

        //         drop(maxmindict);
        // logw(format!("Prop. of alone kmers: {:.1} %", (ialone as f64) / (i as f64) * 100.0).as_str(), Some("info"));
        // logw(format!("Number of edges {}", (nedges as f64) / (2 as f64)).as_str(), Some("info"));

        timevec.as_mut().map(|timevec| {
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
        });

        // indict.iter().for_each(|(h, hi)| {
        //     let himutref = hi.borrow();
        //
        //     // Check first previous neighbours:
        //     for ipre in himutref.pre.iter() {
        //         if !indict.get(&ipre.0).unwrap().borrow().pre.contains(&(*h, ipre.1.rev())) || !indict.get(&ipre.0).unwrap().borrow().post.contains(&(*h, ipre.1)) {
        //             println!("HEY1");
        //         }
        //     }
        //     for ipost in himutref.post.iter() {
        //         if !indict.get(&ipost.0).unwrap().borrow().post.contains(&(*h, ipost.1.rev())) || !indict.get(&ipost.0).unwrap().borrow().pre.contains(&(*h, ipost.1)) {
        //             println!("HEY2");
        //         }
        //     }
        // });

        let mut ptgraph = G::create_from_map::<G>(k, indict);

        timevec.as_mut().map(|timevec| {
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
            )
        });

        // log::info!("Saving graph (pre-shrink w/o one-node contigs) as DOT file...");
        // if path_.is_some() {
        //     let mut wbuf = set_ostream(&Some(path_.unwrap().clone().replace(".dot", "_preshrink.dot")));
        //     ptgraph.write_to_dot(&mut wbuf);
        // }
        // log::info!("Done.");

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
            // let bool1 = false;

            if do_dead_end_removal {
                bool2 = ptgraph.remove_dead_paths();
                // let bool2 = false;

                bool3 = ptgraph.shrink();
                // let bool3 = false;
            }

            if do_bubble_collapse {
                bool4 = ptgraph.correct_bubbles();
                // let bool4 = false;
            }

            //             println!("{} {}", bools, boolr);
            didanyofusdoanything = bool1 || bool2 || bool3 || bool4;
            // break;
        }

        timevec.as_mut().map(|timevec| {
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
        });

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

        timevec.as_mut().map(|timevec| timevec.push(Instant::now()));
        let serialized_contigs = ptgraph.collapse();
        timevec.as_mut().map(|timevec| {
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
        });

        let mut contigs = Contigs::new(serialized_contigs);

        // TEMPORAL RESTRICTION, WIP
        contigs.shrink();

        contigs
    }

    #[cfg(feature = "wasm")]
    fn assemble_wasm<G: Graph>(
        k: usize,
        indict: &mut HashMap<u64, RefCell<HashInfoSimple>, BuildHasherDefault<NoHashHasher<u64>>>,
        maxmindict: &mut HashMap<u64, u64, BuildHasherDefault<NoHashHasher<u64>>>,
        do_bubble_collapse: bool,
        do_dead_end_removal: bool,
        do_conflictive_links_removal: bool,
    ) -> (Contigs, String, String, String) {
        logw("Starting assembler!", Some("info"));

        post_state("assembly:starting");
        // FIRST: iterate over all k-mers, check the existance of forwards/backwards neighbours in the dictionary.
        let mut i = 0;
        let mut ialone = 0;
        let mut nedges = 0;

        // TODO: explore parallelisation?

        post_state("assembly:create_graph");
        indict.iter().for_each(|(h, hi)| {
            let mut himutref = hi.borrow_mut();

            himutref.pre = check_bkg(*h, himutref.hnc, k, himutref.b, indict, maxmindict);
            himutref.post = check_fwd(*h, himutref.hnc, k, himutref.b, indict, maxmindict);
            let tmpnedges = himutref.pre.len() + himutref.post.len();
            nedges += tmpnedges;
            i += 1;
            if tmpnedges == 0 {
                ialone += 1
            };
        });

        //         drop(maxmindict);
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

        // log::info!("Neighbours searched for in {} s", timevec.last().unwrap().duration_since(*timevec.get(timevec.len().wrapping_sub(2)).unwrap()).as_secs());

        // indict.iter().for_each(|(h, hi)| {
        //     let himutref = hi.borrow();
        //
        //     // Check first previous neighbours:
        //     for ipre in himutref.pre.iter() {
        //         if !indict.get(&ipre.0).unwrap().borrow().pre.contains(&(*h, ipre.1.rev())) || !indict.get(&ipre.0).unwrap().borrow().post.contains(&(*h, ipre.1)) {
        //             println!("HEY");
        //         }
        //     }
        //     for ipost in himutref.post.iter() {
        //         if !indict.get(&ipost.0).unwrap().borrow().post.contains(&(*h, ipost.1.rev())) || !indict.get(&ipost.0).unwrap().borrow().pre.contains(&(*h, ipost.1)) {
        //             println!("HEY");
        //         }
        //     }
        // });

        let mut ptgraph = G::create_from_map::<G>(k, indict);

        // log::info!("Saving graph (pre-shrink w/o one-node contigs) as DOT file...");
        // if path_.is_some() {
        //     let mut wbuf = set_ostream(&Some(path_.unwrap().clone().replace(".dot", "_preshrink.dot")));
        //     ptgraph.write_to_dot(&mut wbuf);
        // }
        // log::info!("Done.");

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
            // let bool1 = false;

            if do_dead_end_removal {
                bool2 = ptgraph.remove_dead_paths();
                // let bool2 = false;

                bool3 = ptgraph.shrink();
                // let bool3 = false;
            }

            if do_bubble_collapse {
                bool4 = ptgraph.correct_bubbles();
                // let bool4 = false;
            }

            //             println!("{} {}", bools, boolr);
            didanyofusdoanything = bool1 || bool2 || bool3 || bool4;
            // break;
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
