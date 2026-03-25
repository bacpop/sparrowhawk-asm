// Phase 2 — extract valid k-mers per read and compute ntHash.
//
// One GPU thread per read.  Uses the same sliding-window quality filter as
// count_kmers.wgsl, but also computes the rolling ntHash for each valid
// window and writes a GpuKmerEntry to the pre-allocated output buffer.
//
// Bindings:
//   0 — seq_buf         packed 4 ASCII bases per u32 (little-endian)
//   1 — qual_buf        packed 4 ASCII quality bytes per u32
//   2 — read_offsets    read_offsets[i]  = byte position of read i in seq_buf
//   3 — read_lengths    read_lengths[i]  = number of bases in read i
//   4 — kmer_offsets    exclusive prefix sum of Phase-1 kmer_count; output
//                       for read i starts at kmer_offsets[i]
//   5 — output          pre-allocated GpuKmerEntry array (sized by prefix sum)
//   6 — params          Params uniform

// ── ntHash lookup tables ─────────────────────────────────────────────────────
//
// All 64-bit table values are split as vec2u(lo, hi) where
//   lo = bits[31:0],  hi = bits[63:32].
//
// For MS_TAB_5LL / _7L / _9LC / _11CR the lower 32 bits are always zero, so
// only the hi word is stored (as a plain u32 array).
// For MS_TAB_13R / _19RR the upper 32 bits are always zero, so only the lo
// word is stored.
//
// Indexing: TAB<N>[base * N + k % N]  where base ∈ {0=A,1=C,2=T,3=G}.

// HASH_LOOKUP[base]  — forward hash seed (A=0, C=1, T=2, G=3)
const HASH_LO: array<u32, 4> = array(0x95c60474u, 0x62a02b4cu, 0x4be24456u, 0x82572324u);
const HASH_HI: array<u32, 4> = array(0x3c8bfbb3u, 0x3193c185u, 0x295549f5u, 0x20323ed0u);

// RC_HASH_LOOKUP[base]  — RC hash seed = HASH_LOOKUP[T,G,A,C]
const RC_HASH_LO: array<u32, 4> = array(0x4be24456u, 0x82572324u, 0x95c60474u, 0x62a02b4cu);
const RC_HASH_HI: array<u32, 4> = array(0x295549f5u, 0x20323ed0u, 0x3c8bfbb3u, 0x3193c185u);

// MS_TAB_5LL — upper-32 only, 20 entries, index = base*5 + k%5
const TAB5_HI: array<u32, 20> = array(
    0x38000000u, 0x70000000u, 0xe0000000u, 0xc8000000u, 0x98000000u,
    0x30000000u, 0x60000000u, 0xc0000000u, 0x88000000u, 0x18000000u,
    0x28000000u, 0x50000000u, 0xa0000000u, 0x48000000u, 0x90000000u,
    0x20000000u, 0x40000000u, 0x80000000u, 0x08000000u, 0x10000000u,
);

// MS_TAB_7L — upper-32 only, 28 entries, index = base*7 + k%7
const TAB7_HI: array<u32, 28> = array(
    0x04800000u, 0x01100000u, 0x02200000u, 0x04400000u, 0x00900000u,
    0x01200000u, 0x02400000u, 0x01900000u, 0x03200000u, 0x06400000u,
    0x04900000u, 0x01300000u, 0x02600000u, 0x04c00000u, 0x01500000u,
    0x02a00000u, 0x05400000u, 0x02900000u, 0x05200000u, 0x02500000u,
    0x04a00000u, 0x00300000u, 0x00600000u, 0x00c00000u, 0x01800000u,
    0x03000000u, 0x06000000u, 0x04100000u,
);

// MS_TAB_9LC — upper-32 only, 36 entries, index = base*9 + k%9
const TAB9_HI: array<u32, 36> = array(
    0x000bf800u, 0x0007f800u, 0x000ff000u, 0x000fe800u, 0x000fd800u,
    0x000fb800u, 0x000f7800u, 0x000ef800u, 0x000df800u, 0x0003c000u,
    0x00078000u, 0x000f0000u, 0x000e0800u, 0x000c1800u, 0x00083800u,
    0x00007800u, 0x0000f000u, 0x0001e000u, 0x00054800u, 0x000a9000u,
    0x00052800u, 0x000a5000u, 0x0004a800u, 0x00095000u, 0x0002a800u,
    0x00055000u, 0x000aa000u, 0x00023800u, 0x00047000u, 0x0008e000u,
    0x0001c800u, 0x00039000u, 0x00072000u, 0x000e4000u, 0x000c8800u,
    0x00091800u,
);

// MS_TAB_11CR — upper-32 only, 44 entries, index = base*11 + k%11
const TAB11_HI: array<u32, 44> = array(
    0x000003b3u, 0x00000766u, 0x000006cdu, 0x0000059bu, 0x00000337u,
    0x0000066eu, 0x000004ddu, 0x000001bbu, 0x00000376u, 0x000006ecu,
    0x000005d9u, 0x00000185u, 0x0000030au, 0x00000614u, 0x00000429u,
    0x00000053u, 0x000000a6u, 0x0000014cu, 0x00000298u, 0x00000530u,
    0x00000261u, 0x000004c2u, 0x000001f5u, 0x000003eau, 0x000007d4u,
    0x000007a9u, 0x00000753u, 0x000006a7u, 0x0000054fu, 0x0000029fu,
    0x0000053eu, 0x0000027du, 0x000004fau, 0x000006d0u, 0x000005a1u,
    0x00000343u, 0x00000686u, 0x0000050du, 0x0000021bu, 0x00000436u,
    0x0000006du, 0x000000dau, 0x000001b4u, 0x00000368u,
);

// MS_TAB_13R — lower-32 only, 52 entries, index = base*13 + k%13
const TAB13_LO: array<u32, 52> = array(
    0x95c00000u, 0x2b880000u, 0x57100000u, 0xae200000u, 0x5c480000u,
    0xb8900000u, 0x71280000u, 0xe2500000u, 0xc4a80000u, 0x89580000u,
    0x12b80000u, 0x25700000u, 0x4ae00000u, 0x62a00000u, 0xc5400000u,
    0x8a880000u, 0x15180000u, 0x2a300000u, 0x54600000u, 0xa8c00000u,
    0x51880000u, 0xa3100000u, 0x46280000u, 0x8c500000u, 0x18a80000u,
    0x31500000u, 0x4be00000u, 0x97c00000u, 0x2f880000u, 0x5f100000u,
    0xbe200000u, 0x7c480000u, 0xf8900000u, 0xf1280000u, 0xe2580000u,
    0xc4b80000u, 0x89780000u, 0x12f80000u, 0x25f00000u, 0x82500000u,
    0x04a80000u, 0x09500000u, 0x12a00000u, 0x25400000u, 0x4a800000u,
    0x95000000u, 0x2a080000u, 0x54100000u, 0xa8200000u, 0x50480000u,
    0xa0900000u, 0x41280000u,
);

// MS_TAB_19RR — lower-32 only, 76 entries, index = base*19 + k%19
const TAB19_LO: array<u32, 76> = array(
    0x00060474u, 0x000408e9u, 0x000011d3u, 0x000023a6u, 0x0000474cu,
    0x00008e98u, 0x00011d30u, 0x00023a60u, 0x000474c0u, 0x0000e981u,
    0x0001d302u, 0x0003a604u, 0x00074c08u, 0x00069811u, 0x00053023u,
    0x00026047u, 0x0004c08eu, 0x0001811du, 0x0003023au, 0x00002b4cu,
    0x00005698u, 0x0000ad30u, 0x00015a60u, 0x0002b4c0u, 0x00056980u,
    0x0002d301u, 0x0005a602u, 0x00034c05u, 0x0006980au, 0x00053015u,
    0x0002602bu, 0x0004c056u, 0x000180adu, 0x0003015au, 0x000602b4u,
    0x00040569u, 0x00000ad3u, 0x000015a6u, 0x00024456u, 0x000488acu,
    0x00011159u, 0x000222b2u, 0x00044564u, 0x00008ac9u, 0x00011592u,
    0x00022b24u, 0x00045648u, 0x0000ac91u, 0x00015922u, 0x0002b244u,
    0x00056488u, 0x0002c911u, 0x00059222u, 0x00032445u, 0x0006488au,
    0x00049115u, 0x0001222bu, 0x00072324u, 0x00064649u, 0x00048c93u,
    0x00011927u, 0x0002324eu, 0x0004649cu, 0x0000c939u, 0x00019272u,
    0x000324e4u, 0x000649c8u, 0x00049391u, 0x00012723u, 0x00024e46u,
    0x00049c8cu, 0x00013919u, 0x00027232u, 0x0004e464u, 0x0001c8c9u,
    0x00039192u,
);

// ── Data types ───────────────────────────────────────────────────────────────

struct Params {
    n_reads:  u32,
    K:        u32,  // k-mer length
    min_qual: u32,  // raw ASCII quality threshold
}

struct KmerEntry {
    canon_lo: u32,
    canon_hi: u32,
    nc_lo:    u32,
    nc_hi:    u32,
    bases:    u32,  // bits[1:0]=first base of canonical, bits[3:2]=last base
    pad:      u32,
}

// ── Bindings ─────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read>       seq_buf:      array<u32>;
@group(0) @binding(1) var<storage, read>       qual_buf:     array<u32>;
@group(0) @binding(2) var<storage, read>       read_offsets: array<u32>;
@group(0) @binding(3) var<storage, read>       read_lengths: array<u32>;
@group(0) @binding(4) var<storage, read>       kmer_offsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> output:       array<KmerEntry>;
@group(0) @binding(6) var<uniform>             params:       Params;

// ── Low-level helpers ────────────────────────────────────────────────────────

fn seq_byte(byte_idx: u32) -> u32 {
    return (seq_buf[byte_idx >> 2u] >> ((byte_idx & 3u) * 8u)) & 0xFFu;
}
fn qual_byte(byte_idx: u32) -> u32 {
    return (qual_buf[byte_idx >> 2u] >> ((byte_idx & 3u) * 8u)) & 0xFFu;
}

// encode_base: A→0, C→1, T→2, G→3  (works for both upper- and lower-case)
fn encode_base(ascii: u32) -> u32 { return (ascii >> 1u) & 3u; }

fn rc_base(enc: u32) -> u32 { return enc ^ 2u; }

fn is_n_base(ascii: u32) -> bool { return (ascii & 0xFu) == 14u; }

// ── u64-as-vec2u  (x = lo = bits[31:0],  y = hi = bits[63:32]) ──────────────

fn rotl1(v: vec2u) -> vec2u {
    return vec2u((v.x << 1u) | (v.y >> 31u), (v.y << 1u) | (v.x >> 31u));
}

fn rotr1(v: vec2u) -> vec2u {
    return vec2u((v.x >> 1u) | (v.y << 31u), (v.y >> 1u) | ((v.x & 1u) << 31u));
}

// swapbits_0_19_32_43_52_59  ("srol") — applied after rotl1 in ntHash
fn srol(v: vec2u) -> vec2u {
    let b0  =  v.x        & 1u;
    let b19 = (v.x >> 19u) & 1u;
    let b32 =  v.y        & 1u;
    let b43 = (v.y >> 11u) & 1u;
    let b52 = (v.y >> 20u) & 1u;
    let b59 = (v.y >> 27u) & 1u;
    let x = b0  ^ b19;
    let y = b32 ^ b19;
    let z = b43 ^ b32;
    let t = b52 ^ b43;
    let u = b59 ^ b52;
    let w = b0  ^ b59;
    return vec2u(
        v.x ^ (x | (y << 19u)),
        v.y ^ (z | (t << 11u) | (u << 20u) | (w << 27u)),
    );
}

// swapbits_18_31_42_51_58_63  ("sror") — applied after rotr1 in ntHash RC
fn sror(v: vec2u) -> vec2u {
    let b18 = (v.x >> 18u) & 1u;
    let b31 = (v.x >> 31u) & 1u;
    let b42 = (v.y >> 10u) & 1u;
    let b51 = (v.y >> 19u) & 1u;
    let b58 = (v.y >> 26u) & 1u;
    let b63 = (v.y >> 31u) & 1u;
    let x = b63 ^ b58;
    let y = b58 ^ b51;
    let z = b51 ^ b42;
    let t = b42 ^ b31;
    let u = b31 ^ b18;
    let w = b18 ^ b63;
    return vec2u(
        v.x ^ ((u << 31u) | (w << 18u)),
        v.y ^ ((x << 31u) | (y << 26u) | (z << 19u) | (t << 10u)),
    );
}

fn lt64(a: vec2u, b: vec2u) -> bool {
    return a.y < b.y || (a.y == b.y && a.x < b.x);
}

// ── ntHash helpers ───────────────────────────────────────────────────────────

// Combined removal mask for (encoded base b, k-mer length k) across all six
// MS_TAB split tables.  Upper-only tables contribute to .y; lower-only to .x.
fn ms_tab_mask(b: u32, k: u32) -> vec2u {
    let hi = TAB5_HI[b *  5u + k %  5u]
           | TAB7_HI[b *  7u + k %  7u]
           | TAB9_HI[b *  9u + k %  9u]
           | TAB11_HI[b * 11u + k % 11u];
    let lo = TAB13_LO[b * 13u + k % 13u]
           | TAB19_LO[b * 19u + k % 19u];
    return vec2u(lo, hi);
}

// Roll both hashes one position rightward along the sequence.
//   old_enc = encoded base leaving the window  (position wstart - 1 = pos - K)
//   new_enc = encoded base entering the window (position pos)
fn roll_fwd(
    fh: ptr<function, vec2u>,
    rh: ptr<function, vec2u>,
    old_enc: u32,
    new_enc: u32,
    k: u32,
) {
    // Forward: rotl1 → srol → XOR hash[new] → XOR ms_tab[old, k]
    var f = srol(rotl1(*fh));
    f.x ^= HASH_LO[new_enc];
    f.y ^= HASH_HI[new_enc];
    let m = ms_tab_mask(old_enc, k);
    f.x ^= m.x;
    f.y ^= m.y;
    *fh = f;

    // RC: XOR ms_tab[rc(new), k] → XOR rc_hash[old] → rotr1 → sror
    let m_rc = ms_tab_mask(rc_base(new_enc), k);
    var r = *rh;
    r.x ^= m_rc.x;
    r.y ^= m_rc.y;
    r.x ^= RC_HASH_LO[old_enc];
    r.y ^= RC_HASH_HI[old_enc];
    *rh = sror(rotr1(r));
}

// Initialize both hashes from scratch for the window at absolute byte position
// 'start', length k.  Forward iterates left-to-right; RC iterates right-to-left
// over the same window (matching NtHashIterator::new in nthash.rs).
fn init_hash(
    fh:    ptr<function, vec2u>,
    rh:    ptr<function, vec2u>,
    start: u32,
    k:     u32,
) {
    var f = vec2u(0u, 0u);
    var r = vec2u(0u, 0u);
    for (var i = 0u; i < k; i++) {
        let enc    = encode_base(seq_byte(start + i));
        let enc_rc = encode_base(seq_byte(start + k - 1u - i));
        f = srol(rotl1(f));
        f.x ^= HASH_LO[enc];
        f.y ^= HASH_HI[enc];
        r = srol(rotl1(r));
        r.x ^= RC_HASH_LO[enc_rc];
        r.y ^= RC_HASH_HI[enc_rc];
    }
    *fh = f;
    *rh = r;
}

// ── Main ─────────────────────────────────────────────────────────────────────

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let read_id = gid.x;
    if read_id >= params.n_reads { return; }

    let read_start = read_offsets[read_id];
    let read_len   = read_lengths[read_id];

    if read_len < params.K { return; }

    var out_idx      = kmer_offsets[read_id];
    var last_bad_pos = 0xFFFFFFFFu;  // sentinel: no bad base seen yet
    var fh           = vec2u(0u, 0u);
    var rh           = vec2u(0u, 0u);
    var prev_valid   = false;

    for (var pos = 0u; pos < read_len; pos++) {
        let bp        = read_start + pos;
        let seq_ascii  = seq_byte(bp);
        let qual_ascii = qual_byte(bp);

        if is_n_base(seq_ascii) || qual_ascii < params.min_qual {
            last_bad_pos = pos;
        }

        if pos >= params.K - 1u {
            let wstart = pos - params.K + 1u;
            let valid  = last_bad_pos == 0xFFFFFFFFu || last_bad_pos < wstart;

            if valid {
                let new_enc = encode_base(seq_ascii);

                if prev_valid {
                    // Roll: base at (pos - K) drops out, base at pos enters.
                    // pos >= K here (prev_valid requires pos >= K), so no underflow.
                    let old_enc = encode_base(seq_byte(read_start + pos - params.K));
                    roll_fwd(&fh, &rh, old_enc, new_enc, params.K);
                } else {
                    // Gap or first window: (re)initialize from scratch.
                    init_hash(&fh, &rh, read_start + wstart, params.K);
                }

                // Canonical = min hash, NC = max hash.
                let is_rc = lt64(rh, fh);
                let canon = select(fh, rh, is_rc);
                let nc    = select(rh, fh, is_rc);

                // bases encoding: bits[1:0] = first base of canonical k-mer,
                //                 bits[3:2] = last base of canonical k-mer.
                let first_fwd = encode_base(seq_byte(read_start + wstart));
                let last_fwd  = new_enc;
                // bases: bits[3:2] = first base of canonical k-mer,
                //        bits[1:0] = last  base of canonical k-mer.
                // This matches the CPU Kmer::get_curr_kmerhash_and_bases_and_kmer encoding:
                //   fwd: (first_fwd << 2) | last_fwd
                //   rc:  rc(first_fwd) | (rc(last_fwd) << 2)  [= last_canon | (first_canon<<2)]
                var bases: u32;
                if is_rc {
                    // Canonical RC: first_canon=rc(last_fwd), last_canon=rc(first_fwd)
                    bases = rc_base(first_fwd) | (rc_base(last_fwd) << 2u);
                } else {
                    // Canonical FWD: first_canon=first_fwd, last_canon=last_fwd
                    bases = last_fwd | (first_fwd << 2u);
                }

                output[out_idx] = KmerEntry(canon.x, canon.y, nc.x, nc.y, bases, 0u);
                out_idx += 1u;
            }

            prev_valid = valid;
        }
    }
}
