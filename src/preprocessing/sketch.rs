//! Hash-space sampling and per-quality-floor k-mer spectrum sketches.

use super::{add_to_histogram, MAXSIZEHISTO};
use crate::utils::qual_profile::{MAX_GROUPS, NONE};
use nohash_hasher::NoHashHasher;
use std::{collections::HashMap, hash::BuildHasherDefault};

/// Sampled hashes held before the threshold halves
const SKETCH_BUDGET: usize = 500_000;

/// The canonical k-mer hash never returns a value below this, so the sample is taken from the interval
/// above it: halving from zero would put the threshold in a range the hash cannot reach, and the whole
/// table would be retained away. Measured over 8M k-mers at k = 21, 31, 41 and 71.
const HASH_FLOOR: u64 = 1 << 52;

/// Per-floor k-mer spectra from a fixed-size sample of hash space. Subsampling hash space does not move
/// lambda, as a k-mer's hash does not depend on how often it occurs.
pub(super) struct SpectrumSketch {
    /// Exclusive hash cutoff, also read by the counter before sketch insertion.
    pub(super) threshold: u64,
    counts: HashMap<u64, [u32; MAX_GROUPS], BuildHasherDefault<NoHashHasher<u64>>>,
    /// Set once the threshold can fall no further without emptying the table.
    saturated: bool,
    /// Times the threshold has halved, for the diagnostic line only.
    shrinks: u32,
}

impl SpectrumSketch {
    /// Allocated once at full size: `shrink` frees entries but not the table, so the high-water mark is
    /// reached anyway, and pre-allocating avoids holding the old and new tables during a resize.
    pub(super) fn new() -> Self {
        Self {
            threshold: u64::MAX,
            counts: HashMap::with_capacity_and_hasher(SKETCH_BUDGET, BuildHasherDefault::default()),
            saturated: false,
            shrinks: 0,
        }
    }

    /// One compare rejects most occurrences once the threshold has fallen, so this stays cheap in the
    /// hot loop.
    #[inline]
    pub(super) fn observe(&mut self, hash: u64, group: u8) {
        if hash >= self.threshold || group == NONE {
            return;
        }
        self.counts.entry(hash).or_insert([0; MAX_GROUPS])[group as usize] += 1;
        if !self.saturated && self.counts.len() > SKETCH_BUDGET {
            self.shrink();
        }
    }

    /// Halves the interval above [`HASH_FLOOR`], not the threshold itself, so the threshold converges
    /// on the floor from above and the table always keeps roughly half. Survivors keep the counts they
    /// already had, which is the whole reason the spectrum survives the subsampling.
    pub(super) fn shrink(&mut self) {
        let (previous, was_populated) = (self.threshold, !self.counts.is_empty());
        self.threshold = HASH_FLOOR + (self.threshold - HASH_FLOOR) / 2;
        let t = self.threshold;
        self.counts.retain(|h, _| *h < t);
        self.shrinks += 1;
        // The floor is a property of the hash, measured rather than derived, so a k whose reachable
        // range starts higher would empty the table here. Stop instead, and keep the sample we have.
        if was_populated && self.counts.is_empty() {
            self.threshold = previous;
            self.saturated = true;
        }
    }

    /// Fraction of k-mers retained, so a sampled spectrum can be rescaled to the whole library. The
    /// hash kept is `min(forward, reverse)`, which is not uniform: `P(min < t) = 1 - (1 - t)^2`, about
    /// twice `t` at the thresholds this reaches. Abundance does not enter it, so the shape is unbiased.
    ///
    /// `t` is measured against the whole of `u64`, not the interval above [`HASH_FLOOR`] that
    /// [`Self::shrink`] halves. That looks inconsistent but is what the library measures: against
    /// `check_sketch_against_table` this predicts 27 484 where 27 291 were sampled, while normalising
    /// by the interval is out by half. The hash is denser just above its floor than uniform.
    pub(super) fn fraction(&self) -> f64 {
        let p = self.threshold as f64 / u64::MAX as f64;
        1.0 - (1.0 - p) * (1.0 - p)
    }

    /// One spectrum per floor. A k-mer tagged `g` survives floors `0..=g`, so its count at floor `f` is
    /// the total of the groups at or above `f` — hence the reverse accumulation.
    pub(super) fn spectra(&self, n: usize) -> Vec<Vec<u32>> {
        let mut out = vec![vec![0u32; MAXSIZEHISTO]; n];
        for counts in self.counts.values() {
            let mut acc = 0u32;
            for g in (0..n).rev() {
                acc += counts[g];
                if acc > 0 {
                    add_to_histogram(&mut out[g], acc);
                }
            }
        }
        out
    }

    /// [`Self::spectra`] scaled back to the whole library, since the hole guard budgets in absolute
    /// k-mers. Counts on the x-axis do not move.
    pub(super) fn rescaled_spectra(&self, n: usize) -> Vec<Vec<u32>> {
        let scale = 1.0 / self.fraction();
        let mut out = self.spectra(n);
        for bin in out.iter_mut().flatten() {
            *bin = ((*bin as f64) * scale).min(u32::MAX as f64) as u32;
        }
        out
    }
}

/// The sketch's own spectrum at the floor in force, rescaled, against the table built beside it. Free on
/// every run and the strongest check available that the sampling is unbiased.
pub(super) fn check_sketch_against_table(histovec: &[u32], sketch: &SpectrumSketch, keep: usize) {
    let sampled = &sketch.spectra(keep + 1)[keep];
    let (table, sample): (u64, u64) = (
        histovec.iter().map(|&n| n as u64).sum(),
        sampled.iter().map(|&n| n as u64).sum(),
    );
    let expected = table as f64 * sketch.fraction();
    // The table size is reported alongside the strict-group total: a sketch that has emptied itself
    // used to be indistinguishable here from a library with no strict-floor k-mers.
    log::info!(
        "Sketch: {sample} distinct k-mers sampled from {table} at a hash fraction of {:.3e} \
             (expected {expected:.0}; table holds {} after {} halvings{})",
        sketch.fraction(),
        sketch.counts.len(),
        sketch.shrinks,
        if sketch.saturated { ", saturated" } else { "" }
    );
}

/// The sketch's strict-floor spectrum, scaled back to the whole library. The hole guard budgets in
/// absolute k-mers, so bin heights matter to it; the counts on the x-axis are scale-invariant.
pub(super) fn rescaled_strict_spectrum(sketch: &SpectrumSketch, keep: u8) -> Vec<u32> {
    sketch
        .rescaled_spectra(keep as usize + 1)
        .pop()
        .unwrap_or_default()
}

#[cfg(test)]
pub(super) mod test_support {
    use super::*;

    /// Put a spectrum into one group of a sketch, one hash per distinct k-mer.
    pub(in crate::preprocessing) fn sketch_with(
        group: usize,
        spectrum: &[(u32, usize)],
    ) -> SpectrumSketch {
        let mut s = SpectrumSketch::new();
        let mut h = 1u64;
        for &(count, n) in spectrum {
            for _ in 0..n {
                let mut c = [0u32; MAX_GROUPS];
                c[group] = count;
                s.counts.insert(h, c);
                h += 1;
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::sketch_with;
    use super::*;
    use crate::preprocessing::{
        test_support::{bimodal, xorshift},
        SpectrumEstimate,
    };

    /// The floor the sketch's subsampling rests on. If a future hash change lowers it, `HASH_FLOOR`
    /// is wrong and the sample would be silently truncated rather than emptied.
    #[test]
    fn the_canonical_hash_never_falls_below_the_floor() {
        use crate::utils::nthash::NtHashIterator;
        const BASES: [u8; 4] = *b"ACGT";
        let mut st = 0x2545_F491_4F6C_DD1Du64;
        for k in [21usize, 31, 41, 71] {
            let mut lowest = u64::MAX;
            for _ in 0..20_000 {
                let seq: Vec<u8> = (0..k)
                    .map(|_| BASES[(xorshift(&mut st) % 4) as usize])
                    .collect();
                lowest = lowest.min(
                    NtHashIterator::new(
                        &seq.iter()
                            .map(|&b| crate::utils::bit_encoding::encode_base(b))
                            .collect::<Vec<_>>(),
                        k,
                        true,
                    )
                    .canonical_hash(),
                );
            }
            assert!(
                lowest >= HASH_FLOOR,
                "k={k}: saw {lowest}, below HASH_FLOOR {HASH_FLOOR}"
            );
        }
    }

    /// The bug this guards: halving the threshold itself walks it below the hash's floor, so `retain`
    /// drops every entry and, both gates being `hash < threshold`, nothing is ever admitted again.
    #[test]
    fn shrinking_halves_the_sample_and_keeps_survivors_exact() {
        let mut st = 0x9E37_79B9_7F4A_7C15u64;
        let mut sketch = SpectrumSketch::new();
        // One halving above the floor: the old code stepped from here to exactly `HASH_FLOOR`, which
        // no hash is below, so its `retain` took the whole table with it.
        sketch.threshold = 2 * HASH_FLOOR;
        for _ in 0..40_000 {
            // Drawn from the support the real hash has, which `the_canonical_hash_never_falls_below
            // _the_floor` pins down: uniform above the floor, never under it.
            let hash = HASH_FLOOR + xorshift(&mut st) % HASH_FLOOR;
            sketch.observe(hash, (hash % MAX_GROUPS as u64) as u8);
        }
        let before = sketch.counts.len();
        assert!(before > 1_000, "fixture admitted only {before}");

        sketch.shrink();
        assert!(
            !sketch.counts.is_empty(),
            "shrink erased a populated table: {before} -> 0, threshold now {}",
            sketch.threshold
        );
        let kept = sketch.counts.len() as f64 / before as f64;
        assert!(
            (0.3..0.7).contains(&kept),
            "kept {kept:.2} of the table, expected about half"
        );
        assert!(
            sketch.fraction() > 0.0,
            "a zero fraction divides by zero when rescaling"
        );
        // Survivors keep the counts they had, which is why the spectrum survives the subsampling.
        let mut exact = SpectrumSketch::new();
        let (low, high) = (HASH_FLOOR + 1, u64::MAX - 1);
        for _ in 0..5 {
            exact.observe(low, 0);
        }
        for _ in 0..9 {
            exact.observe(high, 0);
        }
        exact.shrink();
        assert_eq!(exact.counts[&low][0], 5, "a survivor must keep its count");
        assert!(
            !exact.counts.contains_key(&high),
            "a hash above the threshold must go"
        );
    }

    /// The core invariant: a sampled spectrum has the same shape as the full one, because subsampling
    /// hash space does not touch the counts of the k-mers it keeps.
    #[test]
    fn sketch_matches_a_full_count() {
        let mut full = vec![0u32; MAXSIZEHISTO];
        let mut sketch = SpectrumSketch::new();
        // Canonical hashes, i.e. min of a forward and a reverse value, so the test exercises the same
        // non-uniform distribution `fraction` has to correct for.
        for i in 0..60_000u64 {
            let fwd = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let rev = (i ^ 0x5DEE_CE66_D1B7_1234).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            let hash = fwd.min(rev);
            let count = 10 + (i % 40) as u32;
            add_to_histogram(&mut full, count);
            for _ in 0..count {
                sketch.observe(hash, 0);
            }
        }
        sketch.shrink();
        sketch.shrink();
        let sampled = &sketch.spectra(1)[0];
        let (ft, st) = (full.iter().sum::<u32>(), sampled.iter().sum::<u32>());
        assert!(st > 0, "the sample is empty");
        assert_eq!(ft, 60_000);
        let rescaled = st as f64 / sketch.fraction();
        assert!(
            (rescaled / ft as f64 - 1.0).abs() < 0.1,
            "rescaled {rescaled:.0} against {ft}"
        );
        // Bin by bin, the sample is the full spectrum scaled down: empty where it is empty, and each of
        // the 1,500-k-mer bins at counts 10..=49 within 15 % (about five standard errors) of its share.
        for (i, (&f, &s)) in full.iter().zip(sampled.iter()).enumerate() {
            if f == 0 {
                assert_eq!(s, 0, "count {} is empty in the full spectrum", i + 1);
            } else {
                let ratio = s as f64 / (f as f64 * sketch.fraction());
                assert!(
                    (ratio - 1.0).abs() < 0.15,
                    "count {}: sampled {s} of {f}, ratio {ratio:.3}",
                    i + 1
                );
            }
        }
    }

    /// Rescaling moves bin heights, never the counts they sit at: the hole guard reads the heights,
    /// while everything the valley estimator gates on is a ratio.
    #[test]
    fn rescaled_strict_spectrum_preserves_counts() {
        let mut sketch = sketch_with(0, &bimodal(40));
        // `sketch_with` inserts past the threshold, so halve it twice for a fraction worth scaling by.
        sketch.shrink();
        sketch.shrink();
        let raw = sketch.spectra(1).pop().unwrap();
        let scaled = rescaled_strict_spectrum(&sketch, 0);

        let (r, s) = (
            SpectrumEstimate::by_valley(&raw),
            SpectrumEstimate::by_valley(&scaled),
        );
        assert_eq!(r.genomic_peak, s.genomic_peak, "the peak moved");
        assert_eq!(r.valley, s.valley, "the valley moved");
        assert_eq!(r.min_count, s.min_count, "the cutoff moved");

        let scale = 1.0 / sketch.fraction();
        assert!(scale > 1.5, "the fraction did not fall: scale {scale}");
        for (i, (&a, &b)) in raw.iter().zip(scaled.iter()).enumerate() {
            assert_eq!(b, (a as f64 * scale).min(u32::MAX as f64) as u32, "bin {i}");
        }
        assert!(scaled[39] > raw[39], "heights must actually scale");
    }

    /// A k-mer moves between count bins as the floor drops; it does not appear in two bins at once.
    #[test]
    fn a_kmer_moves_between_count_bins_when_the_floor_drops() {
        let mut s = SpectrumSketch::new();
        s.counts.insert(1, [10, 0, 40]);
        let spectra = s.spectra(3);
        assert_eq!(spectra[2][39], 1, "count 40 at the strict floor");
        assert_eq!(spectra[2].iter().sum::<u32>(), 1);
        assert_eq!(spectra[0][49], 1, "count 50 with no filter");
        assert_eq!(spectra[0].iter().sum::<u32>(), 1);
    }
}
