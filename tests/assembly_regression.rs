//! End-to-end regression: assemble simulated reads of a 20 kb E. coli window with the CLI's default
//! settings, and check the contigs against the window's reference. See `tests/data/README.txt`.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The repeat-free window inside `ecoli_window.fasta`, which adds this much flank on each side.
const FLANK: usize = 1_000;
const WINDOW: usize = 20_000;

/// The md5 of the expected contigs' FASTA, as `tests/data/README.txt` records it.
const EXPECTED_MD5: &str = "6ec5c2b95ed2080067608ae058d12a77";

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

/// Runs the CLI on the fixture's reads into `out`, with two threads, no histogram, and `args`.
fn assemble(out: &Path, args: &[&str]) {
    let data = data_dir();
    let run = Command::new(env!("CARGO_BIN_EXE_sparrowhawk-asm"))
        .arg(data.join("ecoli_window_1.fastq.gz"))
        .arg(data.join("ecoli_window_2.fastq.gz"))
        .args(["--threads", "2", "--no-histo"])
        .args(args)
        .arg("--output-dir")
        .arg(out)
        .output()
        .expect("run sparrowhawk-asm");
    assert!(
        run.status.success(),
        "sparrowhawk-asm failed ({}); output kept in {}:\n{}",
        run.status,
        out.display(),
        String::from_utf8_lossy(&run.stderr)
    );
}

fn read_fasta(path: &Path) -> Vec<Vec<u8>> {
    let text = std::fs::read_to_string(path).expect("read FASTA");
    let mut seqs: Vec<Vec<u8>> = Vec::new();
    for line in text.lines() {
        if line.starts_with('>') {
            seqs.push(Vec::new());
        } else if let Some(seq) = seqs.last_mut() {
            seq.extend_from_slice(line.trim().as_bytes());
        }
    }
    seqs
}

fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .rev()
        .map(|base| match base {
            b'A' => b'T',
            b'C' => b'G',
            b'G' => b'C',
            b'T' => b'A',
            other => *other,
        })
        .collect()
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// MD5 (RFC 1321) as lowercase hex: enough to check the README's digest without a dependency.
fn md5_hex(data: &[u8]) -> String {
    const SHIFTS: [[u32; 4]; 4] = [
        [7, 12, 17, 22],
        [5, 9, 14, 20],
        [4, 11, 16, 23],
        [6, 10, 15, 21],
    ];
    // RFC 1321's floor(2^32 * |sin(i + 1)|), fixed here so the digest does not
    // depend on the platform's floating-point sine implementation.
    const CONSTANTS: [u32; 64] = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
        0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
        0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
        0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
        0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
        0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
        0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
        0xeb86d391,
    ];
    let mut state: [u32; 4] = [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
    let mut message = data.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&(data.len() as u64).wrapping_mul(8).to_le_bytes());
    for block in message.as_chunks::<64>().0 {
        let word_bytes = block.as_chunks::<4>().0;
        let words: [u32; 16] = std::array::from_fn(|i| u32::from_le_bytes(word_bytes[i]));
        let [mut a, mut b, mut c, mut d] = state;
        for (i, constant) in CONSTANTS.into_iter().enumerate() {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let rotated = a
                .wrapping_add(f)
                .wrapping_add(constant)
                .wrapping_add(words[g])
                .rotate_left(SHIFTS[i / 16][i % 4]);
            (a, b, c, d) = (d, b.wrapping_add(rotated), b, c);
        }
        for (word, add) in state.iter_mut().zip([a, b, c, d]) {
            *word = word.wrapping_add(add);
        }
    }
    state
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[test]
fn md5_matches_the_rfc_1321_test_suite() {
    for (input, want) in [
        ("", "d41d8cd98f00b204e9800998ecf8427e"),
        ("a", "0cc175b9c0f1b6a831c399e269772661"),
        ("abc", "900150983cd24fb0d6963f7d28e17f72"),
        ("message digest", "f96b697d7cb7938d525a2f31aaf161d0"),
        (
            "abcdefghijklmnopqrstuvwxyz",
            "c3fcd3d76192e4007dfb496cca67e13b",
        ),
        (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
            "d174ab98d277d9f5a5611c2c9f419d9f",
        ),
        (
            "12345678901234567890123456789012345678901234567890123456789012345678901234567890",
            "57edf4a22be3c955ac49da2e2107b67a",
        ),
    ] {
        assert_eq!(md5_hex(input.as_bytes()), want, "{input:?}");
    }
}

/// Every contig must be an exact piece of the reference, on either strand, and together they must
/// cover the window. Today it assembles into a single 20,629 bp contig, whose FASTA has the md5 in
/// `tests/data/README.txt`.
#[test]
fn a_simulated_window_assembles_into_exact_contigs() {
    let out = std::env::temp_dir().join(format!("sparrowhawk-regression-{}", std::process::id()));
    assemble(&out, &["--no-graphs", "--output-prefix", "regression"]);

    let reference = read_fasta(&data_dir().join("ecoli_window.fasta")).remove(0);
    let fasta = out.join("regression_contigs.fasta");
    let digest = md5_hex(&std::fs::read(&fasta).expect("read the contigs"));
    let contigs = read_fasta(&fasta);
    std::fs::remove_dir_all(&out).ok();

    let mut covered = vec![false; reference.len()];
    for (i, contig) in contigs.iter().enumerate() {
        let start = find(&reference, contig)
            .or_else(|| find(&reference, &reverse_complement(contig)))
            .unwrap_or_else(|| {
                panic!(
                    "contig {i} ({} bp) is not an exact piece of the reference",
                    contig.len()
                )
            });
        covered[start..start + contig.len()].fill(true);
    }
    let window_covered = covered[FLANK..FLANK + WINDOW]
        .iter()
        .filter(|&&c| c)
        .count();
    assert!(
        window_covered as f64 >= 0.99 * WINDOW as f64,
        "contigs cover {window_covered} of the window's {WINDOW} bp"
    );
    assert_eq!(
        contigs.iter().map(Vec::len).collect::<Vec<_>>(),
        [20_629],
        "contig lengths"
    );
    assert_eq!(digest, EXPECTED_MD5, "md5 of the contigs' FASTA");
}

/// A dot in the output prefix is part of the name, not an extension to replace: the graph files
/// of `--output-prefix s.v1` are `s.v1_graph.dot`, `.gfa` and `.gfa2`. A single k keeps it quick.
#[test]
fn a_simulated_window_keeps_a_dotted_prefix_in_its_graph_names() {
    let out = std::env::temp_dir().join(format!("sparrowhawk-graphs-{}", std::process::id()));
    assemble(&out, &["-k", "31", "--output-prefix", "s.v1"]);
    let sizes: Vec<_> = ["dot", "gfa", "gfa2"]
        .map(|extension| {
            let path = out.join(format!("s.v1_graph.{extension}"));
            (path.metadata().map(|m| m.len()).unwrap_or(0), path)
        })
        .into();
    let listing: Vec<_> = std::fs::read_dir(&out)
        .map(|dir| dir.flatten().map(|entry| entry.file_name()).collect())
        .unwrap_or_default();
    std::fs::remove_dir_all(&out).ok();
    for (size, path) in sizes {
        assert!(
            size > 0,
            "{} is missing or empty; the output directory held {listing:?}",
            path.display()
        );
    }
}
