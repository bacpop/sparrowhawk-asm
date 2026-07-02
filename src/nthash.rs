//! ntHash is a hash function optimised for a DNA alphabet `{A, C, T, G}`.
//!
//! It works particularly well as a rolling hash, e.g. for k-mers in an input
//! sequence.
//!
//! This implementation based on ntHash 2

use super::bit_encoding::{encode_base, rc_base};

/// Values to look up and multiply for direct k-mers
pub const HASH_LOOKUP: [u64; 4] = [
    0x3c8b_fbb3_95c6_0474, // A
    0x3193_c185_62a0_2b4c, // C
    0x2955_49f5_4be2_4456, // T
    0x2032_3ed0_8257_2324, // G
];

/// Values to look up and multiply for reverse-complement k-mers
pub const RC_HASH_LOOKUP: [u64; 4] = [
    0x2955_49f5_4be2_4456,
    0x2032_3ed0_8257_2324,
    0x3c8b_fbb3_95c6_0474,
    0x3193_c185_62a0_2b4c,
];

/// This is A33R, C33R, T33R, G33R so can be indexed by nucleotide * 33
pub const MS_TAB_33R: [u64; 132] = [
    0x195c60474,
    0x12b8c08e9,
    0x571811d3,
    0xae3023a6,
    0x15c60474c,
    0xb8c08e99,
    0x171811d32,
    0xe3023a65,
    0x1c60474ca,
    0x18c08e995,
    0x11811d32b,
    0x3023a657,
    0x60474cae,
    0xc08e995c,
    0x1811d32b8,
    0x1023a6571,
    0x474cae3,
    0x8e995c6,
    0x11d32b8c,
    0x23a65718,
    0x474cae30,
    0x8e995c60,
    0x11d32b8c0,
    0x3a657181,
    0x74cae302,
    0xe995c604,
    0x1d32b8c08,
    0x1a6571811,
    0x14cae3023,
    0x995c6047,
    0x132b8c08e,
    0x6571811d,
    0xcae3023a,
    0x162a02b4c,
    0xc5405699,
    0x18a80ad32,
    0x115015a65,
    0x2a02b4cb,
    0x54056996,
    0xa80ad32c,
    0x15015a658,
    0xa02b4cb1,
    0x140569962,
    0x80ad32c5,
    0x1015a658a,
    0x2b4cb15,
    0x569962a,
    0xad32c54,
    0x15a658a8,
    0x2b4cb150,
    0x569962a0,
    0xad32c540,
    0x15a658a80,
    0xb4cb1501,
    0x169962a02,
    0xd32c5405,
    0x1a658a80a,
    0x14cb15015,
    0x9962a02b,
    0x132c54056,
    0x658a80ad,
    0xcb15015a,
    0x1962a02b4,
    0x12c540569,
    0x58a80ad3,
    0xb15015a6,
    0x14be24456,
    0x97c488ad,
    0x12f89115a,
    0x5f1222b5,
    0xbe24456a,
    0x17c488ad4,
    0xf89115a9,
    0x1f1222b52,
    0x1e24456a5,
    0x1c488ad4b,
    0x189115a97,
    0x11222b52f,
    0x24456a5f,
    0x488ad4be,
    0x9115a97c,
    0x1222b52f8,
    0x4456a5f1,
    0x88ad4be2,
    0x1115a97c4,
    0x22b52f89,
    0x456a5f12,
    0x8ad4be24,
    0x115a97c48,
    0x2b52f891,
    0x56a5f122,
    0xad4be244,
    0x15a97c488,
    0xb52f8911,
    0x16a5f1222,
    0xd4be2445,
    0x1a97c488a,
    0x152f89115,
    0xa5f1222b,
    0x82572324,
    0x104ae4648,
    0x95c8c91,
    0x12b91922,
    0x25723244,
    0x4ae46488,
    0x95c8c910,
    0x12b919220,
    0x57232441,
    0xae464882,
    0x15c8c9104,
    0xb9192209,
    0x172324412,
    0xe4648825,
    0x1c8c9104a,
    0x191922095,
    0x12324412b,
    0x46488257,
    0x8c9104ae,
    0x11922095c,
    0x324412b9,
    0x64882572,
    0xc9104ae4,
    0x1922095c8,
    0x124412b91,
    0x48825723,
    0x9104ae46,
    0x122095c8c,
    0x4412b919,
    0x88257232,
    0x1104ae464,
    0x2095c8c9,
    0x412b9192,
];

/// This is A31R, C31R, T31R, G31R so can be indexed by nucleotide * 31
pub const MS_TAB_31L: [u64; 124] = [
    0x3c8bfbb200000000,
    0x7917f76400000000,
    0xf22feec800000000,
    0xe45fdd9200000000,
    0xc8bfbb2600000000,
    0x917f764e00000000,
    0x22feec9e00000000,
    0x45fdd93c00000000,
    0x8bfbb27800000000,
    0x17f764f200000000,
    0x2feec9e400000000,
    0x5fdd93c800000000,
    0xbfbb279000000000,
    0x7f764f2200000000,
    0xfeec9e4400000000,
    0xfdd93c8a00000000,
    0xfbb2791600000000,
    0xf764f22e00000000,
    0xeec9e45e00000000,
    0xdd93c8be00000000,
    0xbb27917e00000000,
    0x764f22fe00000000,
    0xec9e45fc00000000,
    0xd93c8bfa00000000,
    0xb27917f600000000,
    0x64f22fee00000000,
    0xc9e45fdc00000000,
    0x93c8bfba00000000,
    0x27917f7600000000,
    0x4f22feec00000000,
    0x9e45fdd800000000,
    0x3193c18400000000,
    0x6327830800000000,
    0xc64f061000000000,
    0x8c9e0c2200000000,
    0x193c184600000000,
    0x3278308c00000000,
    0x64f0611800000000,
    0xc9e0c23000000000,
    0x93c1846200000000,
    0x278308c600000000,
    0x4f06118c00000000,
    0x9e0c231800000000,
    0x3c18463200000000,
    0x78308c6400000000,
    0xf06118c800000000,
    0xe0c2319200000000,
    0xc184632600000000,
    0x8308c64e00000000,
    0x6118c9e00000000,
    0xc23193c00000000,
    0x1846327800000000,
    0x308c64f000000000,
    0x6118c9e000000000,
    0xc23193c000000000,
    0x8463278200000000,
    0x8c64f0600000000,
    0x118c9e0c00000000,
    0x23193c1800000000,
    0x4632783000000000,
    0x8c64f06000000000,
    0x18c9e0c200000000,
    0x295549f400000000,
    0x52aa93e800000000,
    0xa55527d000000000,
    0x4aaa4fa200000000,
    0x95549f4400000000,
    0x2aa93e8a00000000,
    0x55527d1400000000,
    0xaaa4fa2800000000,
    0x5549f45200000000,
    0xaa93e8a400000000,
    0x5527d14a00000000,
    0xaa4fa29400000000,
    0x549f452a00000000,
    0xa93e8a5400000000,
    0x527d14aa00000000,
    0xa4fa295400000000,
    0x49f452aa00000000,
    0x93e8a55400000000,
    0x27d14aaa00000000,
    0x4fa2955400000000,
    0x9f452aa800000000,
    0x3e8a555200000000,
    0x7d14aaa400000000,
    0xfa29554800000000,
    0xf452aa9200000000,
    0xe8a5552600000000,
    0xd14aaa4e00000000,
    0xa295549e00000000,
    0x452aa93e00000000,
    0x8a55527c00000000,
    0x14aaa4fa00000000,
    0x20323ed000000000,
    0x40647da000000000,
    0x80c8fb4000000000,
    0x191f68200000000,
    0x323ed0400000000,
    0x647da0800000000,
    0xc8fb41000000000,
    0x191f682000000000,
    0x323ed04000000000,
    0x647da08000000000,
    0xc8fb410000000000,
    0x91f6820200000000,
    0x23ed040600000000,
    0x47da080c00000000,
    0x8fb4101800000000,
    0x1f68203200000000,
    0x3ed0406400000000,
    0x7da080c800000000,
    0xfb41019000000000,
    0xf682032200000000,
    0xed04064600000000,
    0xda080c8e00000000,
    0xb410191e00000000,
    0x6820323e00000000,
    0xd040647c00000000,
    0xa080c8fa00000000,
    0x410191f600000000,
    0x820323ec00000000,
    0x40647da00000000,
    0x80c8fb400000000,
    0x10191f6800000000,
];

/// This is the look-up table for the first split, 5LL
pub const MS_TAB_5LL: [u64; 20] = [
    0x3800000000000000,
    0x7000000000000000,
    0xe000000000000000,
    0xc800000000000000,
    0x9800000000000000,
    0x3000000000000000,
    0x6000000000000000,
    0xc000000000000000,
    0x8800000000000000,
    0x1800000000000000,
    0x2800000000000000,
    0x5000000000000000,
    0xa000000000000000,
    0x4800000000000000,
    0x9000000000000000,
    0x2000000000000000,
    0x4000000000000000,
    0x8000000000000000,
    0x0800000000000000,
    0x1000000000000000,
];

/// This is the LUT for the second split, 7L
pub const MS_TAB_7L: [u64; 28] = [
    0x0480000000000000,
    0x0110000000000000,
    0x0220000000000000,
    0x0440000000000000,
    0x0090000000000000,
    0x0120000000000000,
    0x0240000000000000,
    0x0190000000000000,
    0x0320000000000000,
    0x0640000000000000,
    0x0490000000000000,
    0x0130000000000000,
    0x0260000000000000,
    0x04c0000000000000,
    0x0150000000000000,
    0x02a0000000000000,
    0x0540000000000000,
    0x0290000000000000,
    0x0520000000000000,
    0x0250000000000000,
    0x04a0000000000000,
    0x0030000000000000,
    0x0060000000000000,
    0x00c0000000000000,
    0x0180000000000000,
    0x0300000000000000,
    0x0600000000000000,
    0x0410000000000000,
];

/// This is the LUT for the third split, 9LC
pub const MS_TAB_9LC: [u64; 36] = [
    0x000bf80000000000,
    0x0007f80000000000,
    0x000ff00000000000,
    0x000fe80000000000,
    0x000fd80000000000,
    0x000fb80000000000,
    0x000f780000000000,
    0x000ef80000000000,
    0x000df80000000000,
    0x0003c00000000000,
    0x0007800000000000,
    0x000f000000000000,
    0x000e080000000000,
    0x000c180000000000,
    0x0008380000000000,
    0x0000780000000000,
    0x0000f00000000000,
    0x0001e00000000000,
    0x0005480000000000,
    0x000a900000000000,
    0x0005280000000000,
    0x000a500000000000,
    0x0004a80000000000,
    0x0009500000000000,
    0x0002a80000000000,
    0x0005500000000000,
    0x000aa00000000000,
    0x0002380000000000,
    0x0004700000000000,
    0x0008e00000000000,
    0x0001c80000000000,
    0x0003900000000000,
    0x0007200000000000,
    0x000e400000000000,
    0x000c880000000000,
    0x0009180000000000,
];

/// This is the LUT for the fourth split, 11CR
pub const MS_TAB_11CR: [u64; 44] = [
    0x03b300000000,
    0x076600000000,
    0x06cd00000000,
    0x059b00000000,
    0x033700000000,
    0x066e00000000,
    0x04dd00000000,
    0x01bb00000000,
    0x037600000000,
    0x06ec00000000,
    0x05d900000000,
    0x018500000000,
    0x030a00000000,
    0x061400000000,
    0x042900000000,
    0x005300000000,
    0x00a600000000,
    0x014c00000000,
    0x029800000000,
    0x053000000000,
    0x026100000000,
    0x04c200000000,
    0x01f500000000,
    0x03ea00000000,
    0x07d400000000,
    0x07a900000000,
    0x075300000000,
    0x06a700000000,
    0x054f00000000,
    0x029f00000000,
    0x053e00000000,
    0x027d00000000,
    0x04fa00000000,
    0x06d000000000,
    0x05a100000000,
    0x034300000000,
    0x068600000000,
    0x050d00000000,
    0x021b00000000,
    0x043600000000,
    0x006d00000000,
    0x00da00000000,
    0x01b400000000,
    0x036800000000,
];

/// This is the LUT for the fifth split, 13R
pub const MS_TAB_13R: [u64; 52] = [
    0x000095c00000,
    0x00002b880000,
    0x000057100000,
    0x0000ae200000,
    0x00005c480000,
    0x0000b8900000,
    0x000071280000,
    0x0000e2500000,
    0x0000c4a80000,
    0x000089580000,
    0x000012b80000,
    0x000025700000,
    0x00004ae00000,
    0x000062a00000,
    0x0000c5400000,
    0x00008a880000,
    0x000015180000,
    0x00002a300000,
    0x000054600000,
    0x0000a8c00000,
    0x000051880000,
    0x0000a3100000,
    0x000046280000,
    0x00008c500000,
    0x000018a80000,
    0x000031500000,
    0x00004be00000,
    0x000097c00000,
    0x00002f880000,
    0x00005f100000,
    0x0000be200000,
    0x00007c480000,
    0x0000f8900000,
    0x0000f1280000,
    0x0000e2580000,
    0x0000c4b80000,
    0x000089780000,
    0x000012f80000,
    0x000025f00000,
    0x000082500000,
    0x000004a80000,
    0x000009500000,
    0x000012a00000,
    0x000025400000,
    0x00004a800000,
    0x000095000000,
    0x00002a080000,
    0x000054100000,
    0x0000a8200000,
    0x000050480000,
    0x0000a0900000,
    0x000041280000,
];

/// This is the LUT for the sixth split, 19RR
pub const MS_TAB_19RR: [u64; 76] = [
    0x00060474, 0x000408e9, 0x000011d3, 0x000023a6, 0x0000474c, 0x00008e98, 0x00011d30, 0x00023a60,
    0x000474c0, 0x0000e981, 0x0001d302, 0x0003a604, 0x00074c08, 0x00069811, 0x00053023, 0x00026047,
    0x0004c08e, 0x0001811d, 0x0003023a, 0x00002b4c, 0x00005698, 0x0000ad30, 0x00015a60, 0x0002b4c0,
    0x00056980, 0x0002d301, 0x0005a602, 0x00034c05, 0x0006980a, 0x00053015, 0x0002602b, 0x0004c056,
    0x000180ad, 0x0003015a, 0x000602b4, 0x00040569, 0x00000ad3, 0x000015a6, 0x00024456, 0x000488ac,
    0x00011159, 0x000222b2, 0x00044564, 0x00008ac9, 0x00011592, 0x00022b24, 0x00045648, 0x0000ac91,
    0x00015922, 0x0002b244, 0x00056488, 0x0002c911, 0x00059222, 0x00032445, 0x0006488a, 0x00049115,
    0x0001222b, 0x00072324, 0x00064649, 0x00048c93, 0x00011927, 0x0002324e, 0x0004649c, 0x0000c939,
    0x00019272, 0x000324e4, 0x000649c8, 0x00049391, 0x00012723, 0x00024e46, 0x00049c8c, 0x00013919,
    0x00027232, 0x0004e464, 0x0001c8c9, 0x00039192,
];

/// Stores forward and (optionally) reverse complement hashes of k-mers in a nucleotide sequence
#[derive(Debug)]
pub struct NtHashIterator {
    k: usize,
    fh: u64,
    rh: Option<u64>,
}

/// Function that exchanges the bits in the 0 and 33 positions
#[inline(always)]
pub fn swapbits033(v: u64) -> u64 {
    let x = (v ^ (v >> 33)) & 1;
    v ^ (x | (x << 33))
}

/// Function that cycles to the RIGHT the bits in the 0, 19, 32, 43, 52, and 59 positions
#[inline(always)]
pub fn swapbits_0_19_32_43_52_59(v: u64) -> u64 {
    let x = (v ^ (v >> 19)) & 1; // The 19th bit XOR the 0th  bit
    let y = ((v >> 32) ^ (v >> 19)) & 1; // The 32nd bit XOR the 19th bit
    let z = ((v >> 43) ^ (v >> 32)) & 1; // The 32nd bit XOR the 43rd bit
    let t = ((v >> 52) ^ (v >> 43)) & 1; // The 43rd bit XOR the 52nd bit
    let u = ((v >> 59) ^ (v >> 52)) & 1; // The 59th bit XOR the 52nd bit
    let w = (v ^ (v >> 59)) & 1; // The 59th bit XOR the 0th  bit

    v ^ (x | (y << 19) | (z << 32) | (t << 43) | (u << 52) | (w << 59))
}

/// Function that exchanges the bits in the 32 and 63 positions
#[inline(always)]
pub fn swapbits3263(v: u64) -> u64 {
    let x = ((v >> 32) ^ (v >> 63)) & 1;
    v ^ ((x << 32) | (x << 63))
}

/// Function that cycles to the LEFT the bits in the 18, 31, 42, 51, 58, and 63 positions
#[inline(always)]
pub fn swapbits_18_31_42_51_58_63(v: u64) -> u64 {
    let x = ((v >> 63) ^ (v >> 58)) & 1; // 63rd XOR 58th bit
    let y = ((v >> 58) ^ (v >> 51)) & 1; // 58th XOR 51st bit
    let z = ((v >> 51) ^ (v >> 42)) & 1; // 51st XOR 42nd bit
    let t = ((v >> 42) ^ (v >> 31)) & 1; // 42nd XOR 31st bit
    let u = ((v >> 31) ^ (v >> 18)) & 1; // 31st XOR 18th bit
    let w = ((v >> 18) ^ (v >> 63)) & 1; // 18th XOR 63rd bit

    v ^ ((x << 63) | (y << 58) | (z << 51) | (t << 42) | (u << 31) | (w << 18))
}

impl NtHashIterator {
    /// Creates a new iterator over a sequence with a given k-mer size
    pub fn new(seq: &[u8], k: usize, rc: bool) -> NtHashIterator {
        let mut fh: u64 = 0;
        // for (_, v) in seq[0..k].iter().enumerate() {
        for v in seq[0..k].iter() {
            fh = fh.rotate_left(1_u32); // This, WITH swapbits033 (next line), is the "srol" function
                                        // fh = swapbits033(fh);
            fh = swapbits_0_19_32_43_52_59(fh);
            fh ^= HASH_LOOKUP[encode_base(*v) as usize];
        }

        let rh = if rc {
            let mut h: u64 = 0;
            // for (_, v) in seq[0..k].iter().rev().enumerate() {
            for v in seq[0..k].iter().rev() {
                h = h.rotate_left(1_u32);
                // h =  swapbits033(h);
                h = swapbits_0_19_32_43_52_59(h);
                h ^= RC_HASH_LOOKUP[encode_base(*v) as usize];
            }
            Some(h)
        } else {
            None
        };

        Self { k, fh, rh }
    }

    /// Move to the next k-mer by adding a new base, removing a base from the end, efficiently updating the hash.
    pub fn roll_fwd(&mut self, old_base: u8, new_base: u8) {
        self.fh = self.fh.rotate_left(1);
        // self.fh = swapbits033(self.fh);
        self.fh = swapbits_0_19_32_43_52_59(self.fh);
        self.fh ^= HASH_LOOKUP[new_base as usize];
        // self.fh ^=   MS_TAB_31L[(old_base as usize * 31) + (self.k % 31)]
        //            | MS_TAB_33R[(old_base as usize) * 33 + (self.k % 33)];
        self.fh ^= MS_TAB_5LL[(old_base as usize * 5) + (self.k % 5)]
            | MS_TAB_7L[(old_base as usize * 7) + (self.k % 7)]
            | MS_TAB_9LC[(old_base as usize * 9) + (self.k % 9)]
            | MS_TAB_11CR[(old_base as usize * 11) + (self.k % 11)]
            | MS_TAB_13R[(old_base as usize * 13) + (self.k % 13)]
            | MS_TAB_19RR[(old_base as usize * 19) + (self.k % 19)];

        if let Some(rev) = self.rh {
            let mut h = rev
                // ^ (MS_TAB_31L[(rc_base(new_base) as usize * 31) + (self.k % 31)]
                //  | MS_TAB_33R[(rc_base(new_base) as usize) * 33 + (self.k % 33)]);
                ^ (  MS_TAB_5LL[( rc_base(new_base) as usize * 5)  + (self.k % 5)]
                   | MS_TAB_7L[(  rc_base(new_base) as usize * 7)  + (self.k % 7)]
                   | MS_TAB_9LC[( rc_base(new_base) as usize * 9)  + (self.k % 9)]
                   | MS_TAB_11CR[(rc_base(new_base) as usize * 11) + (self.k % 11)]
                   | MS_TAB_13R[( rc_base(new_base) as usize * 13) + (self.k % 13)]
                   | MS_TAB_19RR[(rc_base(new_base) as usize * 19) + (self.k % 19)]);
            h ^= RC_HASH_LOOKUP[old_base as usize];
            h = h.rotate_right(1_u32);
            // h = swapbits3263(h);
            h = swapbits_18_31_42_51_58_63(h);
            self.rh = Some(h);
        };
    }

    /// Retrieve the current hash (minimum of forward and reverse complement hashes)
    pub fn curr_hash(&self) -> u64 {
        if let Some(rev) = self.rh {
            u64::min(self.fh, rev)
        } else {
            self.fh
        }
    }

    /// Retrieve the current hash (minimum of forward and reverse complement hashes)
    pub fn curr_hash_and_whether_it_is_the_inverse(&self) -> (u64, u64, bool) {
        let rev = self.rh.unwrap();
        (
            u64::min(self.fh, rev),
            u64::max(self.fh, rev),
            rev < self.fh,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bit_encoding::encode_base;

    #[test]
    fn swapbits033_known_value() {
        // Bit 0 set, bit 33 unset → should swap → only bit 33 set
        assert_eq!(swapbits033(1), 1u64 << 33);
        // Bit 33 set, bit 0 unset → should swap → only bit 0 set
        assert_eq!(swapbits033(1u64 << 33), 1);
    }

    #[test]
    fn swapbits033_roundtrip() {
        for v in [0u64, 1, 1u64 << 33, 0xDEADBEEF, u64::MAX] {
            assert_eq!(swapbits033(swapbits033(v)), v);
        }
    }

    #[test]
    fn swapbits3263_known_value() {
        // Bit 32 set, bit 63 unset → swap → only bit 63 set
        assert_eq!(swapbits3263(1u64 << 32), 1u64 << 63);
        assert_eq!(swapbits3263(1u64 << 63), 1u64 << 32);
    }

    #[test]
    fn swapbits3263_roundtrip() {
        for v in [0u64, 1u64 << 32, 1u64 << 63, 0xDEADBEEF_CAFEBABEu64] {
            assert_eq!(swapbits3263(swapbits3263(v)), v);
        }
    }

    #[test]
    fn nthash_determinism() {
        let seq = b"ACGTACGT";
        let k = 4;
        let h1 = NtHashIterator::new(&seq[0..k], k, true).curr_hash();
        let h2 = NtHashIterator::new(&seq[0..k], k, true).curr_hash();
        assert_eq!(h1, h2);
    }

    #[test]
    fn rolling_equals_fresh() {
        // Roll from seq[0..k] by one position; should equal fresh iterator on seq[1..k+1]
        let seq = b"ACGTACGT";
        let k = 4;
        let mut it = NtHashIterator::new(&seq[0..k], k, true);
        // roll_fwd takes encoded bases, not ASCII
        it.roll_fwd(encode_base(seq[0]), encode_base(seq[k]));
        let rolled = it.curr_hash();
        let fresh = NtHashIterator::new(&seq[1..k + 1], k, true).curr_hash();
        assert_eq!(rolled, fresh);
    }

    #[test]
    fn rolling_all_windows() {
        // All rolling windows must equal fresh iterators on the same window
        let seq = b"ACGTACGT";
        let k = 4;
        let mut it = NtHashIterator::new(&seq[0..k], k, true);
        for i in 1..=(seq.len() - k) {
            it.roll_fwd(encode_base(seq[i - 1]), encode_base(seq[i + k - 1]));
            let fresh = NtHashIterator::new(&seq[i..i + k], k, true).curr_hash();
            assert_eq!(it.curr_hash(), fresh, "window {i} mismatch");
        }
    }

    #[test]
    fn canonical_hash_is_minimum() {
        let seq = b"ACGTACGT";
        let k = 4;
        let it = NtHashIterator::new(&seq[0..k], k, true);
        let (canonical, nc, _is_rc) = it.curr_hash_and_whether_it_is_the_inverse();
        assert!(canonical <= nc);
        assert_eq!(it.curr_hash(), canonical);
    }

    #[test]
    fn is_rc_flag_consistent() {
        let seq = b"ACGTACGT";
        let k = 4;
        let it = NtHashIterator::new(&seq[0..k], k, true);
        let (canonical, _nc, is_rc) = it.curr_hash_and_whether_it_is_the_inverse();
        // is_rc == true means the rc hash is smaller (i.e. canonical == rc hash)
        if is_rc {
            assert_eq!(canonical, it.rh.unwrap());
        } else {
            assert_eq!(canonical, it.fh);
        }
    }
}
