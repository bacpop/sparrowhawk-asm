//! A Bloom filter for counting k-mers from reads
//!
//! This is to support a minimum count from FASTQ files, for basic error
//! correction, while keeping memory use low and predictable. Use the
//! [`KmerFilter`] struct.

use std::borrow::BorrowMut;
use std::cmp::Ordering;

use nohash_hasher::NoHashHasher;
use std::{collections::HashMap, hash::BuildHasherDefault};

/// Default bloom filter width (expected number of k-mers)
///
/// 2^27 =~ 130M
const BLOOM_WIDTH: usize = 1 << 27;
/// Number of bits to use in each bloom block (~1% FPR)
const BITS_PER_ENTRY: usize = 12;

/// A filter which counts input k-mers, returns whether they have passed a count threshold.
///
/// Uses a blocked bloom filter as a first pass to remove singletons.
/// Code for blocked bloom filter based on:
/// <https://github.com/lemire/Code-used-on-Daniel-Lemire-s-blog/blob/master/2021/10/02/wordbasedbloom.cpp>
///
/// This has the advantage of using less memory than a larger countmin filter,
/// being a bit faster (bloom is ~3x faster than countmin, but having count also
/// allows entry to dictionary to be only checked once for each passing k-mer)
///
/// Once passed through the bloom filter, a HashMap is used for counts >=2.
/// This filter therefore has no false-negatives and negligible false-negatives
#[derive(Debug, Clone, Default)]
pub struct KmerFilter {
    /// Size of the bloom filter
    buf_size: u64,
    /// Buffer for the bloom filter
    buffer: Vec<u64>,
    /// Table of counts
    counts: HashMap<u64, (u16, u64, u8), BuildHasherDefault<NoHashHasher<u64>>>,
    /// Minimum count to pass filter
    min_count: u16,
}

impl KmerFilter {
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

    /// Check if in the bloom filter, add if not. Returns whether passed filter
    fn bloom_add_and_check(&mut self, key: u64) -> bool {
        let f_print = Self::fingerprint(key);
        let buf_val = self.buffer[Self::location(key, self.buf_size)].borrow_mut();
        if *buf_val & f_print == f_print {
            true
        } else {
            *buf_val |= f_print;
            false
        }
    }

    /// Creates a new filter with given threshold
    ///
    /// Note:
    /// - Maximum count is [`u16::MAX`] i.e. 65535
    /// - Must call [`KmerFilter::init()`] before using.
    pub fn new(min_count: u16) -> Self {
        let buf_size =
            f64::round(BLOOM_WIDTH as f64 * (BITS_PER_ENTRY as f64 / 8.0) / (u64::BITS as f64))
                as u64;
        Self {
            buf_size,
            buffer: Vec::new(),
            counts: HashMap::with_hasher(BuildHasherDefault::default()),
            min_count,
        }
    }

    /// Initialises table so it is ready for use.
    ///
    /// Allocates memory for bloom filter (which can be avoided with FASTA input)
    pub fn init(&mut self) {
        if self.buffer.is_empty() {
            self.buffer.resize(self.buf_size as usize, 0);
        }
    }

    /// Add an observation of a k-mer and middle base to the filter, and return if it passed
    /// minimum count filtering criterion.
    pub fn filter(&mut self, kmer_hash: u64, kmer_nc_hash: u64, bases: u8) -> Ordering {
        // This is possible because of the k-mer size restriction, the top two
        // bit are always zero
        match self.min_count {
            // No filtering
            0 | 1 => Ordering::Equal,
            // Just the bloom filter
            2 => {
                if self.bloom_add_and_check(kmer_hash) {
                    Ordering::Equal
                } else {
                    Ordering::Less
                }
            }
            // Bloom filter then hash table
            _ => {
                if self.bloom_add_and_check(kmer_hash) {
                    let mut count: u16 = 2;
                    self.counts
                        .entry(kmer_hash)
                        .and_modify(|tuple| {
                            count = tuple.0.saturating_add(1);
                            tuple.0 = count;
                        })
                        .or_insert((count, kmer_nc_hash, bases));
                    self.min_count.cmp(&count)
                } else {
                    Ordering::Less
                }
            }
        }
    }

    /// Get method to retrieve the count map
    pub fn get_counts_map(
        &mut self,
    ) -> &mut HashMap<u64, (u16, u64, u8), BuildHasherDefault<NoHashHasher<u64>>> {
        &mut self.counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_filter(min_count: u16) -> KmerFilter {
        let mut f = KmerFilter::new(min_count);
        f.init();
        f
    }

    #[test]
    fn min_count_zero_always_equal() {
        let mut f = make_filter(0);
        assert_eq!(f.filter(42, 99, 0), Ordering::Equal);
        assert_eq!(f.filter(42, 99, 0), Ordering::Equal);
    }

    #[test]
    fn min_count_one_always_equal() {
        let mut f = make_filter(1);
        assert_eq!(f.filter(42, 99, 0), Ordering::Equal);
        assert_eq!(f.filter(42, 99, 0), Ordering::Equal);
    }

    #[test]
    fn min_count_two_reaches_threshold() {
        let mut f = make_filter(2);
        // First pass: not yet in bloom filter
        assert_eq!(f.filter(1234567890, 9876543210, 0b01), Ordering::Less);
        // Second pass: bloom filter has it → threshold reached
        assert_eq!(f.filter(1234567890, 9876543210, 0b01), Ordering::Equal);
    }

    #[test]
    fn min_count_three_reaches_threshold_on_third_call() {
        let mut f = make_filter(3);
        let h = 0xDEADBEEF_CAFEBABEu64;
        assert_eq!(f.filter(h, 0, 0), Ordering::Less); // singleton (not in bloom)
        assert_eq!(f.filter(h, 0, 0), Ordering::Greater); // in bloom, count=2 < 3
        assert_eq!(f.filter(h, 0, 0), Ordering::Equal); // count=3 == min_count
    }

    #[test]
    fn independent_counters() {
        let mut f = make_filter(2);
        let h1 = 0x0001_0000_0000_0001u64;
        let h2 = 0x0002_0000_0000_0002u64;
        // h1 first seen: Less
        assert_eq!(f.filter(h1, 0, 0), Ordering::Less);
        // h2 first seen: Less (independent)
        assert_eq!(f.filter(h2, 0, 0), Ordering::Less);
        // h1 second time: Equal
        assert_eq!(f.filter(h1, 0, 0), Ordering::Equal);
        // h2 second time: Equal (independent)
        assert_eq!(f.filter(h2, 0, 0), Ordering::Equal);
    }

    #[test]
    fn get_counts_map_has_entry_after_threshold() {
        let mut f = make_filter(3);
        let h = 0x1234_5678_9ABC_DEF0u64;
        let nc = 0xFEDC_BA98_7654_3210u64;
        let bases = 0b1001u8;
        f.filter(h, nc, bases); // singleton
        f.filter(h, nc, bases); // count=2 inserted into map
        f.filter(h, nc, bases); // count=3 reached
        let map = f.get_counts_map();
        assert!(map.contains_key(&h));
        let (count, stored_nc, stored_bases) = map[&h];
        assert_eq!(count, 3);
        assert_eq!(stored_nc, nc);
        assert_eq!(stored_bases, bases);
    }
}
