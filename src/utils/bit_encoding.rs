//! Encode and decode DNA bases into bit representation.
//!
//! The main functions are those to encode and decode bases, and get their
//! reverse complement. Each bases uses two bits, and are packed into 64-bit
//! integers.
//!
//! Easy encoding from ASCII are the 3rd and 2nd bits (works with both
//! upper and lowercase)
//!
//! ```none
//! This encodes as A: 00; C: 01; T: 10; G: 11
//!
//! EOR w/ 10          10     10     10     10
//! gives rc           10     11     00     01
//! ```
//! [`valid_base`] accepts exactly A, C, G and T, in either case.

use bnum::BUint;
use num_traits::{PrimInt, Unsigned};

/// Table from bits 0-3 to ASCII (use [`decode_base()`] not this table).
const LETTER_CODE: [u8; 4] = *b"ACTG";

/// Encode an ASCII char to bits 0-3.
#[inline(always)]
pub fn encode_base(base: u8) -> u8 {
    (base >> 1) & 0x3
}

/// Decode bits 0-3 to ASCII.
#[inline(always)]
pub fn decode_base(bitbase: u8) -> u8 {
    LETTER_CODE[bitbase as usize]
}

/// Reverse complement an encoded base.
#[inline(always)]
pub fn rc_base(base: u8) -> u8 {
    base ^ 2
}

/// A, C, G or T, in either case, with ASCII input: anything else (N, other IUPAC codes, gaps) breaks a
/// k-mer. Lower-casing by OR-ing 0x20 maps exactly these eight bytes onto the four letters.
#[inline(always)]
pub fn valid_base(base: u8) -> bool {
    matches!(base | 0x20, b'a' | b'c' | b'g' | b't')
}

/// Trait to support both `u64`, `u128`, and larger integer format representation of split k-mers
pub trait UInt<'a>:
    PrimInt
    + Unsigned
    + std::fmt::Display
    + std::fmt::Debug
    + std::fmt::Binary
    + std::marker::Send
    + std::marker::Sync
    + std::hash::Hash
    + std::ops::Shl<usize, Output = Self>
    + std::ops::BitAnd
    + std::ops::ShrAssign<i32>
    + std::ops::ShlAssign<i32>
    + std::ops::BitOrAssign
    + std::borrow::BorrowMut<Self>
{
    /// Reverse complement of an encoded and packed split k-mer (64-bits).
    fn rev_comp(self, k_size: usize) -> Self;
    /// Convert to u8 as primitive
    fn as_u8(self) -> u8;
    /// Set to zero
    fn zero_init() -> Self;
    /// Convert from u8, encoded bases 0-3
    fn from_encoded_base(encoded_base: u8) -> Self;
    /// Number of bits in the representation
    fn n_bits(self) -> u32;
    /// Get one two-bit base code, indexed from the least significant end
    fn get_one_code(self, i: usize) -> u8;
}

impl<'a> UInt<'a> for u64 {
    #[inline(always)]
    #[rustfmt::skip]
    fn rev_comp(mut self, k_size: usize) -> Self {
        // Reverse the base order by swapping adjacent groups of 2, 4, 8, ... bits.
        // Binary patterns below are written most significant bit first.
        // 0x3 = 0011: repeating it selects alternating 2-bit groups.
        self = (self >> 2 & 0x3333333333333333)  | (self & 0x3333333333333333) << 2;
        // 0x0F = 0000 1111: repeat this pattern for alternating 4-bit groups.
        self = (self >> 4 & 0x0F0F0F0F0F0F0F0F)  | (self & 0x0F0F0F0F0F0F0F0F) << 4;
        // 0x00FF = 00000000 11111111: alternating groups of 8 zero and 8 one bits.
        self = (self >> 8 & 0x00FF00FF00FF00FF)  | (self & 0x00FF00FF00FF00FF) << 8;
        // 0x0000FFFF = 00000000 00000000 11111111 11111111: 16 zero, then 16 one bits.
        self = (self >> 16 & 0x0000FFFF0000FFFF) | (self & 0x0000FFFF0000FFFF) << 16;
        // 0x00000000FFFFFFFF = 00000000 00000000 00000000 00000000
        //                      11111111 11111111 11111111 11111111: 32 zero, then 32 one bits.
        self = (self >> 32 & 0x00000000FFFFFFFF) | (self & 0x00000000FFFFFFFF) << 32;
        // 0xAA = 1010 1010, so 0xAA... repeats 10 for every two-bit base.
        // XOR flips each base's upper bit: A <-> T and C <-> G.
        self ^= 0xAAAAAAAAAAAAAAAA;

        // Move the k-mer into the lowest 2 * k_size bits.
        self >> (2 * (32 - k_size))
    }

    #[inline(always)]
    fn as_u8(self) -> u8 {
        self as u8
    }

    #[inline(always)]
    fn zero_init() -> Self {
        0
    }

    #[inline(always)]
    fn from_encoded_base(encoded_base: u8) -> Self {
        encoded_base as Self
    }

    #[inline(always)]
    fn n_bits(self) -> u32 {
        Self::BITS
    }

    #[inline(always)]
    fn get_one_code(self, i: usize) -> u8 {
        ((self >> (i * 2)) & 0x3) as u8
    }
}

impl<'a> UInt<'a> for u128 {
    #[inline(always)]
    #[rustfmt::skip]
    fn rev_comp(mut self, k_size: usize) -> u128 {
        // Reverse the base order by swapping adjacent groups of 2, 4, 8, ... bits.
        // Binary patterns below are written most significant bit first.
        // 0x3 = 0011: repeating it selects alternating 2-bit groups.
        self = (self >> 2  & 0x33333333333333333333333333333333)
            |  (self       & 0x33333333333333333333333333333333) << 2;
        // 0x0F = 0000 1111: repeat this pattern for alternating 4-bit groups.
        self = (self >> 4  & 0x0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F)
            |  (self       & 0x0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F) << 4;
        // 0x00FF = 00000000 11111111: alternating groups of 8 zero and 8 one bits.
        self = (self >> 8  & 0x00FF00FF00FF00FF00FF00FF00FF00FF)
            |  (self       & 0x00FF00FF00FF00FF00FF00FF00FF00FF) << 8;
        // 0x0000FFFF = 00000000 00000000 11111111 11111111: 16 zero, then 16 one bits.
        self = (self >> 16 & 0x0000FFFF0000FFFF0000FFFF0000FFFF)
            |  (self       & 0x0000FFFF0000FFFF0000FFFF0000FFFF) << 16;
        // 0x00000000FFFFFFFF = 00000000 00000000 00000000 00000000
        //                      11111111 11111111 11111111 11111111: 32 zero, then 32 one bits.
        self = (self >> 32 & 0x00000000FFFFFFFF00000000FFFFFFFF)
            |  (self       & 0x00000000FFFFFFFF00000000FFFFFFFF) << 32;
        // 0xFFFFFFFFFFFFFFFF is 64 one bits; 0x0000000000000000 is 64 zero bits.
        // The mask is 64 zero, then 64 one bits, repeated across the integer.
        self = (self >> 64 & 0x0000000000000000FFFFFFFFFFFFFFFF)
            |  (self       & 0x0000000000000000FFFFFFFFFFFFFFFF) << 64;
        // 0xAA = 1010 1010, so 0xAA... repeats 10 for every two-bit base.
        // XOR flips each base's upper bit: A <-> T and C <-> G.
        self ^= 0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA;

        // Move the k-mer into the lowest 2 * k_size bits.
        self >> (2 * (64 - k_size))
    }

    #[inline(always)]
    fn as_u8(self) -> u8 {
        self as u8
    }

    #[inline(always)]
    fn zero_init() -> Self {
        0
    }

    #[inline(always)]
    fn from_encoded_base(encoded_base: u8) -> Self {
        encoded_base as Self
    }

    #[inline(always)]
    fn n_bits(self) -> u32 {
        Self::BITS
    }

    #[inline(always)]
    fn get_one_code(self, i: usize) -> u8 {
        ((self >> (i * 2)) & 0x3) as u8
    }
}

// ADDING NEW UINTS TO INCREASE K-MER LENGTH UP TO 256/255.

/// 256 bits: packed k-mers for k from 65 to 128.
pub type U256 = BUint<4>;
/// 512 bits: packed k-mers for k from 129 to 255.
pub type U512 = BUint<8>;

impl<'a> UInt<'a> for U256 {
    #[inline(always)]
    #[rustfmt::skip]
    fn rev_comp(mut self, k_size: usize) -> U256 {
        // Reverse the base order by swapping adjacent groups of 2, 4, 8, ... bits.
        // Binary patterns below are written most significant bit first.

        // from_digits lists u64 limbs least significant first (little-endian).

        // 0x3 = 0011: repeating it selects alternating 2-bit groups.
        self = (self >> 2  & U256::from_digits([0x3333333333333333u64, 0x3333333333333333u64, 0x3333333333333333u64, 0x3333333333333333u64]))
            | ( self       & U256::from_digits([0x3333333333333333u64, 0x3333333333333333u64, 0x3333333333333333u64, 0x3333333333333333u64])) << 2;
        // 0x0F = 0000 1111: repeat this pattern for alternating 4-bit groups.
        self = (self >> 4  & U256::from_digits([0x0F0F0F0F0F0F0F0Fu64, 0x0F0F0F0F0F0F0F0Fu64, 0x0F0F0F0F0F0F0F0Fu64, 0x0F0F0F0F0F0F0F0Fu64]))
            | ( self       & U256::from_digits([0x0F0F0F0F0F0F0F0Fu64, 0x0F0F0F0F0F0F0F0Fu64, 0x0F0F0F0F0F0F0F0Fu64, 0x0F0F0F0F0F0F0F0Fu64])) << 4;
        // 0x00FF = 00000000 11111111: alternating groups of 8 zero and 8 one bits.
        self = (self >> 8  & U256::from_digits([0x00FF00FF00FF00FFu64, 0x00FF00FF00FF00FFu64, 0x00FF00FF00FF00FFu64, 0x00FF00FF00FF00FFu64]))
            | ( self       & U256::from_digits([0x00FF00FF00FF00FFu64, 0x00FF00FF00FF00FFu64, 0x00FF00FF00FF00FFu64, 0x00FF00FF00FF00FFu64])) << 8;
        // 0x0000FFFF = 00000000 00000000 11111111 11111111: 16 zero, then 16 one bits.
        self = (self >> 16 & U256::from_digits([0x0000FFFF0000FFFFu64, 0x0000FFFF0000FFFFu64, 0x0000FFFF0000FFFFu64, 0x0000FFFF0000FFFFu64]))
            | ( self       & U256::from_digits([0x0000FFFF0000FFFFu64, 0x0000FFFF0000FFFFu64, 0x0000FFFF0000FFFFu64, 0x0000FFFF0000FFFFu64])) << 16;
        // 0x00000000FFFFFFFF = 00000000 00000000 00000000 00000000
        //                      11111111 11111111 11111111 11111111: 32 zero, then 32 one bits.
        self = (self >> 32 & U256::from_digits([0x00000000FFFFFFFFu64, 0x00000000FFFFFFFFu64, 0x00000000FFFFFFFFu64, 0x00000000FFFFFFFFu64]))
            | ( self       & U256::from_digits([0x00000000FFFFFFFFu64, 0x00000000FFFFFFFFu64, 0x00000000FFFFFFFFu64, 0x00000000FFFFFFFFu64])) << 32;

        // 0xFFFFFFFFFFFFFFFF is 64 one bits; 0x0000000000000000 is 64 zero bits.
        // The mask is 64 zero, then 64 one bits, repeated across the integer.
        self = (self >> 64 & U256::from_digits([0xFFFFFFFFFFFFFFFFu64, 0x0000000000000000u64, 0xFFFFFFFFFFFFFFFFu64, 0x0000000000000000u64]))
            | ( self       & U256::from_digits([0xFFFFFFFFFFFFFFFFu64, 0x0000000000000000u64, 0xFFFFFFFFFFFFFFFFu64, 0x0000000000000000u64])) << 64;
        // Two all-one limbs, then two all-zero limbs (least significant first):
        // the binary mask repeats 128 zero bits followed by 128 one bits.
        self = (self >> 128& U256::from_digits([0xFFFFFFFFFFFFFFFFu64, 0xFFFFFFFFFFFFFFFFu64, 0x0000000000000000u64, 0x0000000000000000u64]))
            | ( self       & U256::from_digits([0xFFFFFFFFFFFFFFFFu64, 0xFFFFFFFFFFFFFFFFu64, 0x0000000000000000u64, 0x0000000000000000u64])) << 128;

        // 0xAA = 1010 1010, so 0xAA... repeats 10 for every two-bit base.
        // XOR flips each base's upper bit: A <-> T and C <-> G.
        self ^= U256::from_digits([0xAAAAAAAAAAAAAAAAu64, 0xAAAAAAAAAAAAAAAAu64, 0xAAAAAAAAAAAAAAAAu64, 0xAAAAAAAAAAAAAAAAu64]);

        // Move the k-mer into the lowest 2 * k_size bits.
        self >> (2 * (128 - k_size))
    }

    #[inline(always)]
    fn as_u8(self) -> u8 {
        self.digits()[0] as u8
    }

    #[inline(always)]
    fn zero_init() -> Self {
        U256::ZERO
    }

    #[inline(always)]
    fn from_encoded_base(encoded_base: u8) -> Self {
        U256::from_digit(encoded_base as u64)
    }

    #[inline(always)]
    fn n_bits(self) -> u32 {
        Self::BITS
    }

    #[inline(always)]
    fn get_one_code(self, i: usize) -> u8 {
        ((self >> (i * 2)) & U256::from_digit(0x3)).as_u8()
    }
}

impl<'a> UInt<'a> for U512 {
    #[inline(always)]
    #[rustfmt::skip]
    fn rev_comp(mut self, k_size: usize) -> U512 {
        // Reverse the base order by swapping adjacent groups of 2, 4, 8, ... bits.
        // Binary patterns below are written most significant bit first.

        // from_digits lists u64 limbs least significant first (little-endian).

        // 0x3 = 0011: repeating it selects alternating 2-bit groups.
        self = (self >> 2  & U512::from_digits([0x3333333333333333u64, 0x3333333333333333u64, 0x3333333333333333u64, 0x3333333333333333u64,
                                                0x3333333333333333u64, 0x3333333333333333u64, 0x3333333333333333u64, 0x3333333333333333u64]))
            | ( self       & U512::from_digits([0x3333333333333333u64, 0x3333333333333333u64, 0x3333333333333333u64, 0x3333333333333333u64,
                                                0x3333333333333333u64, 0x3333333333333333u64, 0x3333333333333333u64, 0x3333333333333333u64])) << 2;
        // 0x0F = 0000 1111: repeat this pattern for alternating 4-bit groups.
        self = (self >> 4  & U512::from_digits([0x0F0F0F0F0F0F0F0Fu64, 0x0F0F0F0F0F0F0F0Fu64, 0x0F0F0F0F0F0F0F0Fu64, 0x0F0F0F0F0F0F0F0Fu64,
                                                0x0F0F0F0F0F0F0F0Fu64, 0x0F0F0F0F0F0F0F0Fu64, 0x0F0F0F0F0F0F0F0Fu64, 0x0F0F0F0F0F0F0F0Fu64]))
            | ( self       & U512::from_digits([0x0F0F0F0F0F0F0F0Fu64, 0x0F0F0F0F0F0F0F0Fu64, 0x0F0F0F0F0F0F0F0Fu64, 0x0F0F0F0F0F0F0F0Fu64,
                                                0x0F0F0F0F0F0F0F0Fu64, 0x0F0F0F0F0F0F0F0Fu64, 0x0F0F0F0F0F0F0F0Fu64, 0x0F0F0F0F0F0F0F0Fu64])) << 4;
        // 0x00FF = 00000000 11111111: alternating groups of 8 zero and 8 one bits.
        self = (self >> 8  & U512::from_digits([0x00FF00FF00FF00FFu64, 0x00FF00FF00FF00FFu64, 0x00FF00FF00FF00FFu64, 0x00FF00FF00FF00FFu64,
                                                0x00FF00FF00FF00FFu64, 0x00FF00FF00FF00FFu64, 0x00FF00FF00FF00FFu64, 0x00FF00FF00FF00FFu64]))
            | ( self       & U512::from_digits([0x00FF00FF00FF00FFu64, 0x00FF00FF00FF00FFu64, 0x00FF00FF00FF00FFu64, 0x00FF00FF00FF00FFu64,
                                                0x00FF00FF00FF00FFu64, 0x00FF00FF00FF00FFu64, 0x00FF00FF00FF00FFu64, 0x00FF00FF00FF00FFu64])) << 8;
        // 0x0000FFFF = 00000000 00000000 11111111 11111111: 16 zero, then 16 one bits.
        self = (self >> 16 & U512::from_digits([0x0000FFFF0000FFFFu64, 0x0000FFFF0000FFFFu64, 0x0000FFFF0000FFFFu64, 0x0000FFFF0000FFFFu64,
                                                0x0000FFFF0000FFFFu64, 0x0000FFFF0000FFFFu64, 0x0000FFFF0000FFFFu64, 0x0000FFFF0000FFFFu64]))
            | ( self       & U512::from_digits([0x0000FFFF0000FFFFu64, 0x0000FFFF0000FFFFu64, 0x0000FFFF0000FFFFu64, 0x0000FFFF0000FFFFu64,
                                                0x0000FFFF0000FFFFu64, 0x0000FFFF0000FFFFu64, 0x0000FFFF0000FFFFu64, 0x0000FFFF0000FFFFu64])) << 16;
        // 0x00000000FFFFFFFF = 00000000 00000000 00000000 00000000
        //                      11111111 11111111 11111111 11111111: 32 zero, then 32 one bits.
        self = (self >> 32 & U512::from_digits([0x00000000FFFFFFFFu64, 0x00000000FFFFFFFFu64, 0x00000000FFFFFFFFu64, 0x00000000FFFFFFFFu64,
                                                0x00000000FFFFFFFFu64, 0x00000000FFFFFFFFu64, 0x00000000FFFFFFFFu64, 0x00000000FFFFFFFFu64]))
            | ( self       & U512::from_digits([0x00000000FFFFFFFFu64, 0x00000000FFFFFFFFu64, 0x00000000FFFFFFFFu64, 0x00000000FFFFFFFFu64,
                                                0x00000000FFFFFFFFu64, 0x00000000FFFFFFFFu64, 0x00000000FFFFFFFFu64, 0x00000000FFFFFFFFu64])) << 32;

        // 0xFFFFFFFFFFFFFFFF is 64 one bits; 0x0000000000000000 is 64 zero bits.
        // The mask is 64 zero, then 64 one bits, repeated across the integer.
        self = (self >> 64 & U512::from_digits([0xFFFFFFFFFFFFFFFFu64, 0x0000000000000000u64, 0xFFFFFFFFFFFFFFFFu64, 0x0000000000000000u64,
                                                0xFFFFFFFFFFFFFFFFu64, 0x0000000000000000u64, 0xFFFFFFFFFFFFFFFFu64, 0x0000000000000000u64]))
            | ( self       & U512::from_digits([0xFFFFFFFFFFFFFFFFu64, 0x0000000000000000u64, 0xFFFFFFFFFFFFFFFFu64, 0x0000000000000000u64,
                                                0xFFFFFFFFFFFFFFFFu64, 0x0000000000000000u64, 0xFFFFFFFFFFFFFFFFu64, 0x0000000000000000u64])) << 64;
        // Two all-one limbs, then two all-zero limbs (least significant first):
        // the binary mask repeats 128 zero bits followed by 128 one bits.
        self = (self >> 128& U512::from_digits([0xFFFFFFFFFFFFFFFFu64, 0xFFFFFFFFFFFFFFFFu64, 0x0000000000000000u64, 0x0000000000000000u64,
                                                0xFFFFFFFFFFFFFFFFu64, 0xFFFFFFFFFFFFFFFFu64, 0x0000000000000000u64, 0x0000000000000000u64]))
            | ( self       & U512::from_digits([0xFFFFFFFFFFFFFFFFu64, 0xFFFFFFFFFFFFFFFFu64, 0x0000000000000000u64, 0x0000000000000000u64,
                                                0xFFFFFFFFFFFFFFFFu64, 0xFFFFFFFFFFFFFFFFu64, 0x0000000000000000u64, 0x0000000000000000u64])) << 128;
        // Four all-one limbs, then four all-zero limbs (least significant first):
        // the binary mask is 256 zero bits followed by 256 one bits.
        self = (self >> 256& U512::from_digits([0xFFFFFFFFFFFFFFFFu64, 0xFFFFFFFFFFFFFFFFu64, 0xFFFFFFFFFFFFFFFFu64, 0xFFFFFFFFFFFFFFFFu64,
                                                0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64]))
            | ( self       & U512::from_digits([0xFFFFFFFFFFFFFFFFu64, 0xFFFFFFFFFFFFFFFFu64, 0xFFFFFFFFFFFFFFFFu64, 0xFFFFFFFFFFFFFFFFu64,
                                                0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64])) << 256;

        // 0xAA = 1010 1010, so 0xAA... repeats 10 for every two-bit base.
        // XOR flips each base's upper bit: A <-> T and C <-> G.
        self ^= U512::from_digits([0xAAAAAAAAAAAAAAAAu64, 0xAAAAAAAAAAAAAAAAu64, 0xAAAAAAAAAAAAAAAAu64, 0xAAAAAAAAAAAAAAAAu64,
                                   0xAAAAAAAAAAAAAAAAu64, 0xAAAAAAAAAAAAAAAAu64, 0xAAAAAAAAAAAAAAAAu64, 0xAAAAAAAAAAAAAAAAu64]);

        // Move the k-mer into the lowest 2 * k_size bits.
        self >> (2 * (256 - k_size))
    }

    #[inline(always)]
    fn as_u8(self) -> u8 {
        self.digits()[0] as u8
    }

    #[inline(always)]
    fn zero_init() -> Self {
        U512::ZERO
    }

    #[inline(always)]
    fn from_encoded_base(encoded_base: u8) -> Self {
        U512::from_digit(encoded_base as u64)
    }

    #[inline(always)]
    fn n_bits(self) -> u32 {
        Self::BITS
    }

    #[inline(always)]
    fn get_one_code(self, i: usize) -> u8 {
        ((self >> (i * 2)) & U512::from_digit(0x3)).as_u8()
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

    use pretty_assertions::assert_eq;

    /// The 2-bit code (A=0, C=1, T=2, G=3, either case) round-trips and complements A<->T and C<->G; only
    /// A, C, G and T are valid.
    #[test]
    fn base_codes() {
        // (base, code, complement's code)
        for (base, code, complement) in [(b'A', 0, 2), (b'C', 1, 3), (b'T', 2, 0), (b'G', 3, 1)] {
            assert_eq!(encode_base(base), code);
            assert_eq!(encode_base(base.to_ascii_lowercase()), code);
            assert_eq!(decode_base(code), base);
            assert_eq!(rc_base(code), complement);
            assert!(valid_base(base));
            assert!(valid_base(base.to_ascii_lowercase()));
        }
        for &base in b"NnRYKMSWBDHVrykmswbdhvUu-." {
            assert!(!valid_base(base), "{}", base as char);
        }
    }

    /// Every width reverse-complements a k-mer as base-by-base complementing in reverse does, which
    /// checks each width's masks against the others'. A hand-worked pair and palindrome pin the
    /// packing itself, so the generic check cannot agree with a wrong reference.
    #[test]
    fn rev_comp_agrees_across_widths() {
        // Packing is most significant first: the first base sits in the highest two bits used.
        // ACG = 00 01 11 = 7, and its reverse complement CGT = 01 11 10 = 30.
        let acg: u64 = 0b00_01_11;
        let cgt: u64 = 0b01_11_10;
        assert_eq!(acg.rev_comp(3), cgt);
        assert_eq!(cgt.rev_comp(3), acg);
        // ACGT = 00 01 11 10 = 30 is its own reverse complement.
        let acgt: u64 = 0b00_01_11_10;
        assert_eq!(acgt.rev_comp(4), acgt);
        assert_eq!((acgt as u128).rev_comp(4), acgt as u128);
        assert_eq!(U256::from_digit(acgt).rev_comp(4), U256::from_digit(acgt));
        assert_eq!(U512::from_digit(acgt).rev_comp(4), U512::from_digit(acgt));

        fn check<IntT: for<'a> UInt<'a>>(k: usize) {
            let codes: Vec<u8> = (0..k).map(|i| ((i * 7 + i / 3) % 4) as u8).collect();
            let pack = |codes: &mut dyn Iterator<Item = u8>| {
                codes.fold(IntT::zero_init(), |kmer, code| {
                    (kmer << 2) | IntT::from_encoded_base(code)
                })
            };
            let kmer = pack(&mut codes.iter().copied());
            let reverse = pack(&mut codes.iter().rev().map(|&code| rc_base(code)));
            assert_eq!(kmer.rev_comp(k), reverse, "k={k}, {} bits", kmer.n_bits());
            assert_eq!(kmer.rev_comp(k).rev_comp(k), kmer);
        }
        for k in [3, 21, 31] {
            check::<u64>(k);
        }
        for k in [3, 31, 33, 63] {
            check::<u128>(k);
        }
        for k in [3, 63, 65, 101, 127] {
            check::<U256>(k);
        }
        for k in [3, 127, 129, 201, 255] {
            check::<U512>(k);
        }
    }
}
