//! The candidate base-quality floors for a library, and which of them each k-mer window clears.

/// Reads peeked for the quality alphabet. Only distinct values are wanted, never a spectrum, so an
/// RTA3 library shows all 4-5 of its bins long before this.
pub const PEEK_READS: usize = 5_000;
/// The ladder never holds more than the A/B/C of the workflow.
pub const MAX_GROUPS: usize = 3;
/// A "loose" floor at or below this is not a filter: on every RTA3 alphabet the bin below B is 2.
pub const MIN_LOOSE_FLOOR: u8 = 10;
/// Tag for a window that clears no floor at all (fewer than k bases, or an N inside it).
pub const NONE: u8 = u8::MAX;

/// Distinct PHRED values in the head of `path`, ascending. Empty for FASTA, which has no qualities.
pub fn peek_alphabet(path: &str, max_reads: usize) -> Vec<u8> {
    let mut seen = [false; 64];
    let Ok(mut reader) = needletail::parse_fastx_file(path) else {
        return Vec::new();
    };
    let mut n = 0;
    while let Some(Ok(record)) = reader.next() {
        let Some(qual) = record.qual() else {
            return Vec::new();
        };
        for &q in qual {
            let phred = q.saturating_sub(33) as usize;
            if phred < seen.len() {
                seen[phred] = true;
            }
        }
        n += 1;
        if n >= max_reads {
            break;
        }
    }
    (0..seen.len())
        .filter(|p| seen[*p])
        .map(|p| p as u8)
        .collect()
}

/// The A/B/C ladder, ascending in looseness, so a group index is directly "how loose".
pub fn floors_from(bins: &[u8], default_floor: u8) -> Vec<u8> {
    // A is not the default itself but the bin the default selects: on 2/11/25/37 a default of 20 keeps
    // Q25 and above, so tagging at 25 and tagging at 20 are the same partition.
    let strict = *bins
        .iter()
        .find(|b| **b >= default_floor)
        .unwrap_or(&default_floor);
    match bins.iter().rev().find(|b| **b < strict && **b > MIN_LOOSE_FLOOR) {
        Some(&loose) => vec![0, loose, strict],
        None => vec![0, strict],
    }
}

/// For each position, the strictest floor whose k-window ending there is clear of Ns and low qualities,
/// as an index into `floors`; [`NONE`] where no floor is cleared. Indexed by the window's **last** base,
/// matching `Kmer::end_index`.
pub fn window_groups(seq: &[u8], qual: &[u8], k: usize, floors: &[u8], out: &mut Vec<u8>) {
    out.clear();
    out.resize(seq.len(), NONE);
    // One run counter per floor, so nothing is allocated per window. The reset condition is deliberately
    // the pair of tests the k-mer iterator itself restarts on, `valid_base` and `valid_qual`.
    let mut run = [0usize; MAX_GROUPS];
    for i in 0..seq.len().min(qual.len()) {
        let phred = qual[i].saturating_sub(33);
        let valid = matches!(
            seq[i],
            b'A' | b'C' | b'G' | b'T' | b'a' | b'c' | b'g' | b't'
        );
        let mut best = NONE;
        for (g, floor) in floors.iter().enumerate() {
            run[g] = if valid && phred >= *floor { run[g] + 1 } else { 0 };
            if run[g] >= k {
                best = g as u8;
            }
        }
        out[i] = best;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kmer::Kmer;
    use std::borrow::Cow;

    /// K-mers the real iterator emits from this read at this floor.
    fn iterator_count(seq: &[u8], qual: &[u8], k: usize, floor: u8) -> usize {
        match Kmer::<u64>::new(Cow::Borrowed(seq), seq.len(), Some(qual), k, floor, true) {
            None => 0,
            Some(mut it) => {
                let mut n = 1;
                while it.get_next_hash_and_bases().is_some() {
                    n += 1;
                }
                n
            }
        }
    }

    /// The invariant the whole tagging scheme rests on: a window classed at or above floor `g` is
    /// exactly a k-mer the iterator would emit at that floor. Catches an off-by-one in either.
    #[test]
    fn window_groups_agree_with_the_iterator() {
        let seq = b"ACGTACGTAACGGTTACGATCGATTACGGCATCAGGTACAGGTTACAGGATCAGGTACA";
        // A quality string spanning all three bins, with a dip and a recovery.
        let qual: Vec<u8> = (0..seq.len())
            .map(|i| 33 + match i % 9 {
                0 | 1 => 2u8,
                2 | 3 | 4 => 11,
                _ => 25,
            })
            .collect();
        let floors = vec![0u8, 11, 25];
        let mut groups = Vec::new();
        for k in [3usize, 5, 7] {
            window_groups(seq, &qual, k, &floors, &mut groups);
            for (g, &floor) in floors.iter().enumerate() {
                let tagged = groups
                    .iter()
                    .filter(|&&t| t != NONE && (t as usize) >= g)
                    .count();
                assert_eq!(
                    tagged,
                    iterator_count(seq, &qual, k, floor),
                    "k={k} floor={floor} (group {g})"
                );
            }
        }
    }

    #[test]
    fn floors_from_caps_the_loose_floor() {
        assert_eq!(floors_from(&[2, 11, 25, 37], 20), vec![0, 11, 25]);
        assert_eq!(floors_from(&[2, 12, 24, 40], 20), vec![0, 12, 24]);
        assert_eq!(floors_from(&[14, 21, 27, 32, 36], 20), vec![0, 14, 21]);
        // 2 is below MIN_LOOSE_FLOOR, so there is no usable B and the ladder is A -> C.
        assert_eq!(floors_from(&[2, 40], 20), vec![0, 40]);
        // No bin reaches the default, so A admits nothing and the strict pass comes back empty. B is
        // still the top bin, which is what the ladder then loosens to.
        assert_eq!(floors_from(&[2, 11], 20), vec![0, 11, 20]);
        // No alphabet at all (FASTA): nothing to loosen to.
        assert_eq!(floors_from(&[], 20), vec![0, 20]);
    }

    /// An N breaks every run, so no window spanning it clears any floor, not even 0.
    #[test]
    fn an_n_breaks_every_group() {
        let seq = b"ACGTACGTNACGTACGT";
        let qual = vec![33 + 37u8; seq.len()];
        let floors = vec![0u8, 11, 25];
        let k = 5;
        let mut groups = Vec::new();
        window_groups(seq, &qual, k, &floors, &mut groups);
        let n_at = 8;
        for i in n_at..(n_at + k).min(groups.len()) {
            assert_eq!(groups[i], NONE, "window ending at {i} spans the N");
        }
        assert_eq!(groups[n_at + k], 2, "the first clean window after the N");
    }
}
