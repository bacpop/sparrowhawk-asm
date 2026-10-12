//! End-to-end regression: assemble simulated reads of a 20 kb E. coli window with the CLI's default
//! settings, and check the contigs against the window's reference. See `tests/data/README.txt`.

use std::path::Path;
use std::process::Command;

/// The repeat-free window inside `ecoli_window.fasta`, which adds this much flank on each side.
const FLANK: usize = 1_000;
const WINDOW: usize = 20_000;

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

/// Every contig must be an exact piece of the reference, on either strand, and together they must
/// cover the window in one long contig: today it assembles into a single 20,629 bp contig.
#[test]
fn a_simulated_window_assembles_into_exact_contigs() {
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let out = std::env::temp_dir().join(format!("sparrowhawk-regression-{}", std::process::id()));

    let run = Command::new(env!("CARGO_BIN_EXE_sparrowhawk-asm"))
        .arg("build")
        .arg(data.join("ecoli_window_1.fastq.gz"))
        .arg(data.join("ecoli_window_2.fastq.gz"))
        .args(["--threads", "2", "--no-graphs", "--no-histo"])
        .args(["--output-prefix", "regression", "--output-dir"])
        .arg(&out)
        .output()
        .expect("run sparrowhawk-asm");
    assert!(
        run.status.success(),
        "sparrowhawk-asm failed ({}); output kept in {}:\n{}",
        run.status,
        out.display(),
        String::from_utf8_lossy(&run.stderr)
    );

    let reference = read_fasta(&data.join("ecoli_window.fasta")).remove(0);
    let contigs = read_fasta(&out.join("regression_contigs.fasta"));
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
    let longest = contigs.iter().map(Vec::len).max().unwrap_or(0);
    assert!(
        window_covered as f64 >= 0.99 * WINDOW as f64,
        "contigs cover {window_covered} of the window's {WINDOW} bp"
    );
    assert!(
        longest >= 19_000,
        "the longest contig is {longest} bp, under 19,000"
    );
}
