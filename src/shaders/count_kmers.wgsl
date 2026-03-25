// Phase 1 — count valid k-mers per read.
//
// One GPU thread per read.  Does NOT compute hashes — just counts windows whose
// every base has quality ≥ min_qual and is not an N.  Output is a u32 count
// per read, used by the CPU to compute a prefix sum (read_kmer_offsets) and
// size the Phase-2 output buffer.

struct Params {
    n_reads:  u32,
    K:        u32,  // k-mer length
    min_qual: u32,  // raw ASCII quality threshold (e.g. 33 + phred_offset)
}

// Packed 4 ASCII bytes per u32 (little-endian: byte 0 in bits [7:0]).
@group(0) @binding(0) var<storage, read>        seq_buf:     array<u32>;
@group(0) @binding(1) var<storage, read>        qual_buf:    array<u32>;
// read_byte_offsets[i] = byte position of read i's first base in seq_buf/qual_buf
@group(0) @binding(2) var<storage, read>        read_offsets: array<u32>;
// read_lengths[i] = number of bases in read i
@group(0) @binding(3) var<storage, read>        read_lengths: array<u32>;
@group(0) @binding(4) var<storage, read_write>  kmer_count:  array<u32>;
@group(0) @binding(5) var<uniform>              params:      Params;

fn seq_byte(byte_idx: u32) -> u32 {
    return (seq_buf[byte_idx >> 2u] >> ((byte_idx & 3u) * 8u)) & 0xFFu;
}
fn qual_byte(byte_idx: u32) -> u32 {
    return (qual_buf[byte_idx >> 2u] >> ((byte_idx & 3u) * 8u)) & 0xFFu;
}

// Returns true if the ASCII base is N or n.
fn is_n_base(ascii: u32) -> bool {
    return (ascii & 0xFu) == 14u;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let read_id  = gid.x;
    if read_id >= params.n_reads { return; }

    let read_start = read_offsets[read_id];
    let read_len   = read_lengths[read_id];

    if read_len < params.K {
        kmer_count[read_id] = 0u;
        return;
    }

    var count          = 0u;
    var last_bad_pos   = 0xFFFFFFFFu; // sentinel: no bad base seen yet

    for (var pos = 0u; pos < read_len; pos += 1u) {
        let bp  = read_start + pos;
        let seq = seq_byte(bp);
        let q   = qual_byte(bp);

        if is_n_base(seq) || q < params.min_qual {
            last_bad_pos = pos;
        }

        if pos >= params.K - 1u {
            let wstart = pos - params.K + 1u;
            // Valid window: no bad base within [wstart, pos]
            if last_bad_pos == 0xFFFFFFFFu || last_bad_pos < wstart {
                count += 1u;
            }
        }
    }

    kmer_count[read_id] = count;
}
