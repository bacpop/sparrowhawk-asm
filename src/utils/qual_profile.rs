//! The candidate base-quality floors for a library, and which of them each k-mer window clears.

use crate::utils::bit_encoding::valid_base;

/// Reads peeked for the quality alphabet. Only distinct values are wanted, never a spectrum, so a
/// library shows all 4-5 of its bins long before this.
pub const PEEK_READS: usize = 5_000;
/// The ladder never holds more than the A/B/C of the workflow.
pub const MAX_GROUPS: usize = 3;
/// A "loose" floor at or below this is not a filter, after checking many reads it seems lower than ten implies no quality whatsoever.
pub const MIN_LOOSE_FLOOR: u8 = 10;
/// A loose rung must sit at least this far below the strict one, or it admits the same bases under a
/// different name. Binned alphabets clear this easily; near-continuous ones do not.
pub const MIN_RUNG_GAP: u8 = 5;
/// Tag for a window that clears no floor at all (fewer than k bases, or an N inside it).
pub const NONE: u8 = u8::MAX;

/// Distinct PHRED values on called bases (A, C, G, T) in the first `max_reads` of `records`, ascending.
/// Ns carry Q2 but break every window at every floor, so they say nothing about the ladder.
/// Empty for FASTA, which has no qualities.
pub fn peek_alphabet<I: Iterator<Item = crate::utils::reads::OwnedRecord>>(
    records: I,
    max_reads: usize,
) -> Vec<u8> {
    let mut seen = [false; 64];
    for (seq, qual) in records.take(max_reads) {
        let Some(qual) = qual else {
            return Vec::new();
        };
        for (&base, &q) in seq.iter().zip(&qual) {
            if !valid_base(base) {
                continue;
            }
            let phred = q.saturating_sub(33) as usize;
            if phred < seen.len() {
                seen[phred] = true;
            }
        }
    }
    (0..seen.len())
        .filter(|p| seen[*p])
        .map(|p| p as u8)
        .collect()
}

/// The A/B/C ladder, ascending in looseness, so a group index is directly "how loose". The loose rung
/// merges into floor 0 when no called base sits below it; one quality value, or none below the strict
/// floor, leaves floor 0 alone.
pub fn floors_from(bins: &[u8], default_floor: u8) -> Vec<u8> {
    // A is not the default itself but the bin the default selects: on 2/11/25/37 a default of 20 keeps
    // Q25 and above, so tagging at 25 and tagging at 20 are the same partition.
    let strict = *bins
        .iter()
        .find(|b| **b >= default_floor)
        .unwrap_or(&default_floor);
    // One value carries no information (SRA Lite's constant Q30; FASTA has none), and with no call
    // below the strict floor every floor keeps the same windows.
    if bins.len() <= 1 || bins[0] >= strict {
        return vec![0];
    }
    match bins
        .iter()
        .rev()
        .find(|b| b.saturating_add(MIN_RUNG_GAP) <= strict && **b > MIN_LOOSE_FLOOR)
    {
        // Floor 0 differs from the loose rung only if some called base sits below it; otherwise the
        // two are one floor, and the loose rung merges into 0.
        Some(&loose) if bins[0] < loose => vec![0, loose, strict],
        _ => vec![0, strict],
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
        let valid = valid_base(seq[i]);
        let mut best = NONE;
        for (g, floor) in floors.iter().enumerate() {
            run[g] = if valid && phred >= *floor {
                run[g] + 1
            } else {
                0
            };
            if run[g] >= k {
                best = g as u8;
            }
        }
        out[i] = best;
    }
}

/// Stored four-bit equivalent of `window_groups`, using the active ladder's highest class.
pub fn window_classes(
    seq: crate::preprocessing::read_store::PackedSeq<'_>,
    k: usize,
    top: u8,
    out: &mut Vec<u8>,
) {
    out.clear();
    out.resize(seq.len(), NONE);
    let mut run = [0usize; MAX_GROUPS];
    for (i, slot) in out.iter_mut().enumerate() {
        let class = seq.nibble(i) >> 2;
        let mut best = NONE;
        for (g, count) in run.iter_mut().enumerate().take(usize::from(top) + 1) {
            *count =
                if class != crate::preprocessing::read_store::UNUSABLE && usize::from(class) >= g {
                    *count + 1
                } else {
                    0
                };
            if *count >= k {
                best = g as u8;
            }
        }
        *slot = best;
    }
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
    use crate::preprocessing::kmer::Kmer;
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
        // The R must break windows for both, as an N does.
        let seq = b"ACGTACGTAACGGTTACGATCGATTACGRCATCAGGTACAGGTTACAGGATCAGGTACA";
        // A quality string spanning all three bins, with a dip and a recovery.
        let qual: Vec<u8> = (0..seq.len())
            .map(|i| {
                33 + match i % 9 {
                    0 | 1 => 2u8,
                    2..=4 => 11,
                    _ => 25,
                }
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
        // Gaps of 14, 12 and 7 all clear MIN_RUNG_GAP, so each keeps its middle rung.
        assert_eq!(floors_from(&[2, 11, 25, 37], 20), vec![0, 11, 25]);
        assert_eq!(floors_from(&[2, 12, 24, 40], 20), vec![0, 12, 24]);
        // No called base below 14, so floor 0 and the 14 rung admit the same bases and merge.
        assert_eq!(floors_from(&[14, 21, 27, 32, 36], 20), vec![0, 21]);
        assert_eq!(floors_from(&[12, 20, 37], 20), vec![0, 20]);
        assert_eq!(floors_from(&[11, 25, 37], 20), vec![0, 25]);
        assert_eq!(floors_from(&[3, 7, 16, 23, 28], 20), vec![0, 16, 23]);
        // Exactly MIN_RUNG_GAP is enough; one less is not. A default of 22 so that the bin under test
        // is not itself picked as the strict floor.
        assert_eq!(floors_from(&[2, 20, 25], 22), vec![0, 20, 25]);
        assert_eq!(floors_from(&[2, 21, 25], 22), vec![0, 25]);
        // 2 is below MIN_LOOSE_FLOOR, so there is no usable B and the ladder is A -> C.
        assert_eq!(floors_from(&[2, 40], 20), vec![0, 40]);
        // No bin reaches the default, so A admits nothing and the strict pass comes back empty. B is
        // still the top bin, which is what the ladder then loosens to.
        assert_eq!(floors_from(&[2, 11], 20), vec![0, 11, 20]);
        // No alphabet at all (FASTA), or a single value: qualities say nothing, so only floor 0.
        assert_eq!(floors_from(&[], 20), vec![0]);
        assert_eq!(floors_from(&[30], 20), vec![0]);
        assert_eq!(floors_from(&[2], 20), vec![0]);
        // Every call clears the strict floor, so it keeps the same windows as floor 0.
        assert_eq!(floors_from(&[30, 37], 20), vec![0]);
        assert_eq!(floors_from(&[2, 11, 25, 37], 0), vec![0]);
        // A near-continuous alphabet: the bin below the strict floor is one PHRED unit away, so it is
        // the same filter under another name and the ladder must skip it.
        assert_eq!(floors_from(&[2, 19, 20], 20), vec![0, 20]);
    }

    /// Ns carry Q2 on Illumina, but they are not calls, so the peek leaves them out of the alphabet.
    #[test]
    fn the_peek_ignores_the_quality_of_ns() {
        let records = vec![
            (b"ACGTNNACGT".to_vec(), Some(b"-.-.##-.-.".to_vec())),
            (b"NACG".to_vec(), Some(b"#---".to_vec())),
        ];
        assert_eq!(peek_alphabet(records.into_iter(), 10), vec![12, 13]);
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
        for (i, &group) in groups.iter().enumerate().take(n_at + k).skip(n_at) {
            assert_eq!(group, NONE, "window ending at {i} spans the N");
        }
        assert_eq!(groups[n_at + k], 2, "the first clean window after the N");
    }
}
