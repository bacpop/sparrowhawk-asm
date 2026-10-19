//! Consumer tests: these imports deliberately use only the crate's public API.
#![cfg(not(target_family = "wasm"))]

use sparrowhawk_asm::{
    assemble, assemble_with_graph,
    graph_works::{
        spelling::{spell_contigs, spell_path, SpellError},
        Assemble, BasicAsm, Contigs,
    },
    preprocessing::{preprocessing, read_store::ReadStore, PassInput, PreprocessingOptions},
    Assembly, AssemblyError, AssemblyOptions, InputFiles, IteratorSource, NodeStruct, OwnedRecord,
    QualOpts, ReadSource,
};
use std::{
    cell::Cell,
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "sparrowhawk-api-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn files() -> InputFiles {
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    InputFiles::new([
        data.join("ecoli_window_1.fastq.gz"),
        data.join("ecoli_window_2.fastq.gz"),
    ])
    .unwrap()
}
fn fixed(ks: &[usize]) -> AssemblyOptions {
    AssemblyOptions {
        k_values: ks.to_vec(),
        min_count: Some(2),
        min_qual: Some(20),
        use_bloom: false,
        ..AssemblyOptions::default()
    }
}
fn fasta(assembly: &Assembly) -> Vec<u8> {
    let mut out = Vec::new();
    assert_eq!(assembly.write_fasta(&mut out).unwrap(), assembly.len());
    out
}
fn normalise(mut seqs: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    for seq in &mut seqs {
        let rc: Vec<_> = seq
            .iter()
            .rev()
            .map(|base| match base {
                b'A' => b'T',
                b'C' => b'G',
                b'G' => b'C',
                b'T' => b'A',
                _ => unreachable!(),
            })
            .collect();
        if rc < *seq {
            *seq = rc;
        }
    }
    seqs.sort();
    seqs
}
fn sequences(assembly: &Assembly) -> Vec<Vec<u8>> {
    normalise(assembly.contigs().map(|seq| seq.ascii()).collect())
}
fn parse_fasta(bytes: &[u8]) -> Vec<Vec<u8>> {
    let mut seqs: Vec<Vec<u8>> = Vec::new();
    for line in bytes.split(|&b| b == b'\n').filter(|line| !line.is_empty()) {
        if line[0] == b'>' {
            seqs.push(Vec::new());
        } else {
            seqs.last_mut().unwrap().extend_from_slice(line);
        }
    }
    normalise(seqs)
}
fn matches_cli(ks: &[usize]) {
    let scratch = Scratch::new();
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_sparrowhawk-asm"));
    cmd.arg(data.join("ecoli_window_1.fastq.gz"))
        .arg(data.join("ecoli_window_2.fastq.gz"))
        .args([
            "--threads",
            "2",
            "--no-histo",
            "--no-graphs",
            "--no-bloom",
            "--min-count",
            "2",
            "--min-qual",
            "20",
            "-k",
        ]);
    cmd.arg(
        ks.iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(","),
    );
    let out = cmd.arg("--output-dir").arg(&scratch.0).output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let assembly = assemble(&files(), &fixed(ks)).unwrap();
    assert!(!assembly.is_empty());
    assert_eq!(
        sequences(&assembly),
        parse_fasta(&fs::read(scratch.0.join("sphk_contigs.fasta")).unwrap())
    );
    assert_eq!(assembly.completed_k_values(), ks);
    assert_eq!(assembly.stopped_at(), None);
}
#[test]
fn single_k_matches_cli() {
    matches_cli(&[31]);
}
#[test]
fn multik_matches_cli() {
    matches_cli(&[21, 31, 41]);
}

#[test]
fn file_and_iterator_sources_match() {
    let records = files()
        .records()
        .unwrap()
        .collect::<io::Result<Vec<_>>>()
        .unwrap();
    let source = IteratorSource::new(|| Ok(records.clone().into_iter().map(Ok)));
    let options = fixed(&[21, 31]);
    assert_eq!(
        sequences(&assemble(&files(), &options).unwrap()),
        sequences(&assemble(&source, &options).unwrap())
    );
}
#[test]
fn components_match_complete_single_k() {
    let options = fixed(&[31]);
    let mut counted = preprocessing::<u64, _>(
        PassInput::Text(files().records().unwrap()),
        31,
        &PreprocessingOptions {
            quality: QualOpts {
                min_count: 2,
                min_qual: 20,
            },
            use_bloom: false,
            fit_min_count: false,
        },
    )
    .unwrap();
    assert!(!counted.kmers.is_empty());
    let entries: Vec<_> = counted
        .kmers
        .iter()
        .map(|(hash, count, packed)| (hash, count, *packed))
        .collect();
    assert_eq!(entries.len(), counted.kmers.len());
    for &(hash, count, packed) in entries.iter().take(16) {
        assert_eq!(counted.kmers.count(hash), Some(count));
        assert_eq!(counted.kmers.get_packed(hash), Some(&packed));
    }
    let correction = options
        .correction_for(31, counted.genomic_peak, counted.used_min_count)
        .unwrap();
    // Invalid graph options must leave the graph-construction arrays intact.
    assert!(matches!(
        BasicAsm::assemble(33, &mut counted.kmers, correction, None),
        Err(AssemblyError::InvalidOptions(_))
    ));
    assert_eq!(counted.kmers.len(), entries.len());
    let contigs = BasicAsm::assemble(31, &mut counted.kmers, correction, None).unwrap();
    assert!(counted.kmers.is_empty());
    assert_eq!(counted.kmers.iter().count(), 0);
    assert_eq!(counted.kmers.count(entries[0].0), None);
    assert_eq!(counted.kmers.get_packed(entries[0].0), Some(&entries[0].2));
    assert!(matches!(
        BasicAsm::assemble(31, &mut counted.kmers, correction, None),
        Err(AssemblyError::InvalidOptions(_))
    ));
    let packed =
        spell_contigs(&contigs, |hash| counted.kmers.get_packed(hash).copied(), 31).unwrap();
    let mut manual = Vec::new();
    packed
        .write_fasta(options.min_contig_length, &mut manual)
        .unwrap();
    assert_eq!(
        parse_fasta(&manual),
        sequences(&assemble(&files(), &options).unwrap())
    );
}

#[test]
fn invalid_options_are_rejected() {
    let opens = Cell::new(0);
    let source = IteratorSource::new(|| {
        opens.set(opens.get() + 1);
        Ok(std::iter::empty::<io::Result<OwnedRecord>>())
    });
    for ks in [vec![0], vec![32], vec![257], vec![31, 21], vec![31, 31]] {
        assert!(matches!(
            assemble(&source, &fixed(&ks)),
            Err(AssemblyError::InvalidOptions(_))
        ));
    }
    for options in [
        AssemblyOptions {
            min_count: Some(1),
            ..AssemblyOptions::default()
        },
        AssemblyOptions {
            pop_ratio: f32::NAN,
            ..fixed(&[31])
        },
        AssemblyOptions {
            peak_ratio: Some(1.0),
            ..fixed(&[31])
        },
        AssemblyOptions {
            dead_end_length_kmult: f32::INFINITY,
            ..fixed(&[31])
        },
        AssemblyOptions {
            dead_end_coverage_kmult: Some(-1.0),
            ..fixed(&[31])
        },
        AssemblyOptions {
            dead_end_coverage_ratio: 0.0,
            ..fixed(&[31])
        },
        AssemblyOptions {
            ec_ratio: 1.0,
            ..fixed(&[31])
        },
        AssemblyOptions {
            two_bit_reads: true,
            ..fixed(&[31])
        },
        AssemblyOptions {
            two_bit_reads: true,
            ..fixed(&[21, 31])
        },
    ] {
        assert!(matches!(
            assemble(&source, &options),
            Err(AssemblyError::InvalidOptions(_))
        ));
    }
    assert_eq!(opens.get(), 0);
    assert!(InputFiles::new(Vec::<PathBuf>::new()).is_err());
}
#[test]
fn undersized_packed_type_is_rejected() {
    let input = || PassInput::Text(std::iter::empty::<io::Result<OwnedRecord>>());
    for k in [0, 2, 33, 255, usize::MAX] {
        assert!(matches!(
            preprocessing::<u64, _>(input(), k, &PreprocessingOptions::default()),
            Err(AssemblyError::InvalidOptions(_))
        ));
    }
}
#[test]
fn malformed_and_truncated_input_returns_error() {
    let scratch = Scratch::new();
    for (i, data) in [
        b"not a FASTX file".as_slice(),
        b"@read\nACGTACGT\n+\n!\n",
        b"@read\nACGT\n+\n",
    ]
    .iter()
    .enumerate()
    {
        let path = scratch.0.join(format!("bad{i}.fq"));
        fs::write(&path, data).unwrap();
        let source = InputFiles::new([&path]).unwrap();
        assert!(matches!(
            assemble(&source, &fixed(&[3])),
            Err(AssemblyError::Input(_))
        ));
        let mut records = source.records().unwrap();
        assert!(records.next().unwrap().is_err());
        assert!(
            records.next().is_none(),
            "a file source must terminate after its error"
        );
    }
    let compressed =
        fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/ecoli_window_1.fastq.gz"))
            .unwrap();
    let path = scratch.0.join("truncated.fq.gz");
    fs::write(&path, &compressed[..compressed.len() / 2]).unwrap();
    assert!(matches!(
        assemble(&InputFiles::new([path]).unwrap(), &fixed(&[31])),
        Err(AssemblyError::Input(_))
    ));
    let path = scratch.0.join("absent.fq");
    let source = InputFiles::new([path.clone()]).unwrap();
    match source.records().unwrap().next().unwrap().unwrap_err() {
        error if error.kind() == io::ErrorKind::NotFound => {
            assert!(error.to_string().contains("absent.fq"))
        }
        error => panic!("unexpected error: {error}"),
    }
    // A failure opening a subsequent file cannot silently end a successful first file.
    let good = scratch.0.join("good.fa");
    fs::write(&good, b">read\nACGTACGTACGT\n").unwrap();
    let source = InputFiles::new([good, path]).unwrap();
    let mut records = source.records().unwrap();
    assert!(records.next().unwrap().is_ok());
    assert!(records.next().unwrap().is_err());
    assert!(records.next().is_none());
}
#[test]
fn input_error_after_completed_batch_is_propagated() {
    let source = IteratorSource::new(|| {
        Ok((0..9000).map(|i| {
            if i == 8999 {
                Err(io::Error::other("injected after a full batch"))
            } else {
                Ok((b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT".to_vec(), None))
            }
        }))
    });
    match assemble(&source, &fixed(&[31])) {
        Err(AssemblyError::Input(error)) => {
            assert!(error.to_string().contains("after a full batch"))
        }
        _ => panic!("input failure must propagate after a completed batch"),
    }
}
#[test]
fn quality_length_mismatch_is_rejected() {
    let source = IteratorSource::new(|| {
        Ok(std::iter::once(Ok((
            b"ACGTACGT".to_vec(),
            Some(vec![b'I'; 7]),
        ))))
    });
    // Also checks the quality peek, before counting.
    for min_qual in [None, Some(20)] {
        assert!(matches!(
            assemble(
                &source,
                &AssemblyOptions {
                    min_qual,
                    ..fixed(&[3])
                }
            ),
            Err(AssemblyError::Input(_))
        ));
    }
}
#[test]
fn two_bit_input_errors_are_recoverable() {
    for floors in [vec![], vec![20, 10], vec![0, 0], vec![0, 1, 2, 3]] {
        assert!(ReadStore::new(&floors, false).is_err());
    }
    assert!(ReadStore::new(&[20], true).is_err());
    let mut store = ReadStore::new(&[0], true).unwrap();
    store.push(b"ACGT", None).unwrap();
    assert!(store.push(b"ACNT", None).is_err());
    assert_eq!(store.seqs().len(), 1);
    assert_eq!(store.average_read_len(), Some(4.0));
    store.push(b"TGCA", None).unwrap();
    store.finish();
    assert!(store.push(b"ACGT", None).is_err());
    assert_eq!(store.seqs().len(), 2);
    let source = IteratorSource::new(|| Ok(std::iter::once(Ok((b"ACGTN".to_vec(), None)))));
    assert!(matches!(
        assemble(
            &source,
            &AssemblyOptions {
                min_qual: Some(0),
                two_bit_reads: true,
                ..fixed(&[3, 5])
            }
        ),
        Err(AssemblyError::Input(_))
    ));
    let store = ReadStore::new(&[0], false).unwrap();
    assert!(matches!(
        preprocessing::<u64, _>(
            PassInput::stored(&store),
            3,
            &PreprocessingOptions {
                quality: QualOpts {
                    min_count: 2,
                    min_qual: 20
                },
                use_bloom: false,
                fit_min_count: false
            }
        ),
        Err(AssemblyError::InvalidOptions(_))
    ));
}
struct BrokenWriter;
impl Write for BrokenWriter {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        Err(io::Error::other("writer failed"))
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
#[test]
fn fasta_filter_matches_contig_iterator() {
    let mut options = fixed(&[31]);
    options.min_contig_length = 0;
    let unfiltered = assemble(&files(), &options).unwrap();
    let longest = unfiltered.contigs().map(|seq| seq.len()).max().unwrap();
    for threshold in [longest, longest + 1] {
        let assembly = assemble(
            &files(),
            &AssemblyOptions {
                min_contig_length: threshold,
                ..options.clone()
            },
        )
        .unwrap();
        let bytes = fasta(&assembly);
        assert_eq!(parse_fasta(&bytes), sequences(&assembly));
        assert_eq!(assembly.is_empty(), threshold > longest);
        if threshold == longest {
            assert!(assembly.write_fasta(&mut BrokenWriter).is_err());
        }
    }
}
#[test]
fn graph_callback_error_is_propagated() {
    let mut calls = 0;
    let mut callback = |_: &sparrowhawk_asm::DbgGraph| {
        calls += 1;
        Err(io::Error::other("graph failed"))
    };
    assert!(matches!(
        assemble_with_graph(&files(), &fixed(&[21, 31]), &mut callback),
        Err(AssemblyError::GraphOutput(_))
    ));
    assert_eq!(calls, 1);
}
#[test]
fn repeated_calls_work_without_global_initialisation() {
    let options = fixed(&[31]);
    assert_eq!(
        fasta(&assemble(&files(), &options).unwrap()),
        fasta(&assemble(&files(), &options).unwrap())
    );
}
#[test]
fn assembly_works_in_caller_rayon_pool() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let assembly = pool.install(|| assemble(&files(), &fixed(&[31]))).unwrap();
    assert!(!assembly.is_empty());
}
#[test]
fn invalid_spelling_is_rejected() {
    assert!(matches!(
        spell_path::<u64>(&[1], |_| Some(0), 33),
        Err(SpellError::InvalidK { .. })
    ));
    assert_eq!(spell_path::<u64>(&[], |_| None, 31), Err(SpellError::Empty));
    assert_eq!(
        spell_path::<u64>(&[7], |_| None, 31),
        Err(SpellError::UnknownKmer { index: 0, hash: 7 })
    );
    assert_eq!(
        spell_path::<u64>(&[1, 2], |h| Some(if h == 1 { 0 } else { 0b010101 }), 3),
        Err(SpellError::NotAWalk { index: 1 })
    );
    let node = NodeStruct {
        counts: 1,
        abs_ind: vec![7],
        innerdir: None,
    };
    let contigs = Contigs::new(vec![vec![node.clone(), node]]);
    assert!(matches!(
        spell_contigs::<u64>(&contigs, |_| Some(0), 31),
        Err(SpellError::UncollapsedContig { index: 0 })
    ));
}
