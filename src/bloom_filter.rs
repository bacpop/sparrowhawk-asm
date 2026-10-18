//! A Bloom filter for counting k-mers from reads
//!
//! This is to support a minimum count from FASTQ files, for basic error
//! correction, while keeping memory use low and predictable. The counter keeps
//! one [`BloomBits`] per count-map shard.

use std::borrow::BorrowMut;

/// Default bloom filter width (expected number of k-mers)
///
/// 2^27 =~ 130M
const BLOOM_WIDTH: usize = 1 << 27;
/// Number of bits to use in each bloom block (~1% FPR)
const BITS_PER_ENTRY: usize = 12;

/// The blocked Bloom filter's bit array on its own, so the parallel counter can shard it: one array
/// per count-map shard keeps every buffer single-writer, needing no atomics. Based on
/// <https://github.com/lemire/Code-used-on-Daniel-Lemire-s-blog/blob/master/2021/10/02/wordbasedbloom.cpp>, then taken from ska.rust in origin
#[derive(Debug, Clone)]
pub struct BloomBits {
    /// Size of the bloom filter, in 64-bit words
    buf_size: u64,
    /// Buffer for the bloom filter, i.e. the actual filter.
    buffer: Vec<u64>,
}

impl BloomBits {
    /// Cheap modulo
    /// https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    #[inline(always)]
    fn reduce(key: u64, range: u64) -> u64 {
        (((key as u128) * (range as u128)) >> 64) as u64
    }

    /// Like splitmix64 but simpler and faster
    #[inline(always)]
    fn cheap_mix(key: u64) -> u64 {
        (key ^ (key >> 31)).wrapping_mul(0x85D0_59AA_3331_21CF)
    }

    /// Extract five bits from word as fingerprint
    #[inline(always)]
    fn fingerprint(key: u64) -> u64 {
        1 << (key & 63)
            | 1 << ((key >> 6) & 63)
            | 1 << ((key >> 12) & 63)
            | 1 << ((key >> 18) & 63)
            | 1 << ((key >> 24) & 63)
    }

    /// Generate a location in the buffer from the hash
    #[inline(always)]
    fn location(key: u64, range: u64) -> usize {
        Self::reduce(Self::cheap_mix(key), range) as usize
    }

    /// Words the unsharded filter uses. A sharded caller divides this, so bits per key are preserved.
    /// `BITS_PER_ENTRY` is already in bits; the former `/ 8.0` sized the filter 8x too small, I think
    /// the bug is already in the Lemire blog, but perhaps I'm wrong?
    pub fn default_words() -> u64 {
        f64::round(BLOOM_WIDTH as f64 * (BITS_PER_ENTRY as f64) / (u64::BITS as f64)) as u64
    }

    /// Sized but not allocated; [`Self::init`] does that, so FASTA input never pays for the buffer.
    /// Floored at one word, because `location` reduces into this range and would divide by zero.
    pub fn with_words(words: u64) -> Self {
        Self {
            buf_size: words.max(1),
            buffer: Vec::new(),
        }
    }

    /// Allocates the buffer, if it has not been allocated already.
    pub fn init(&mut self) {
        if self.buffer.is_empty() {
            self.buffer.resize(self.buf_size as usize, 0);
        }
    }

    /// Number of set bits and allocated bits in the Bloom buffer.
    pub(crate) fn occupancy(&self) -> (u64, u64) {
        let set_bits = self
            .buffer
            .iter()
            .map(|word| u64::from(word.count_ones()))
            .sum();
        let capacity_bits = self.buffer.len() as u64 * u64::BITS as u64;
        (set_bits, capacity_bits)
    }

    /// Set this key's fingerprint, reporting whether it was already present.
    #[inline]
    pub fn add_and_check(&mut self, key: u64) -> bool {
        let f_print = Self::fingerprint(key);
        let buf_val = self.buffer[Self::location(key, self.buf_size)].borrow_mut();
        if *buf_val & f_print == f_print {
            true
        } else {
            *buf_val |= f_print;
            false
        }
    }
}

/// A zero-word filter would index an empty buffer, so the default is the full-size one.
impl Default for BloomBits {
    fn default() -> Self {
        Self::with_words(Self::default_words())
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

    #[test]
    fn bloom_occupancy_counts_set_bits_without_changing_the_filter() {
        let mut bits = BloomBits::with_words(2);
        bits.init();
        assert_eq!(bits.occupancy(), (0, 128));

        assert!(!bits.add_and_check(0x1234_5678_9ABC_DEF0));
        let after_first_insert = bits.occupancy();
        assert!(after_first_insert.0 > 0);
        assert!(after_first_insert.0 <= 5);
        assert_eq!(after_first_insert.1, 128);

        assert!(bits.add_and_check(0x1234_5678_9ABC_DEF0));
        assert_eq!(bits.occupancy(), after_first_insert);
    }
}
