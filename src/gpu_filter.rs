//! GPU-accelerated radix sort + count + filter for kmer preprocessing.
//!
//! This module implements the bulk-mode kmer pipeline using wgpu compute shaders:
//! - 8-pass LSD radix sort (256 buckets per pass) on canonical hash
//! - Count runs of identical canonical hashes
//! - Filter by min_count threshold
//!
//! Data is partitioned into BUCKET_COUNT=64 buckets (top 6 bits of canonical hash)
//! at extraction time so each bucket fits within GPU buffer limits.
//!
//! Pipeline improvements over the naive version:
//! - P1: Single command encoder per radix pass (8 submits/bucket instead of 32)
//! - P2: Buffer pool pre-allocated at max-bucket size (no per-bucket allocation)
//! - P3: GPU-side histogram clearing (no CPU→GPU zero-fill transfers)
//! - P4: count+filter+scatter in one command encoder; intermediate count readback
//!       removed — scatter runs immediately after prefix sum, no CPU stall

use std::{
    collections::HashMap,
    hash::BuildHasherDefault,
    path::PathBuf,
};

use bytemuck::{Pod, Zeroable};
use nohash_hasher::NoHashHasher;
use wgpu::util::DeviceExt;

use crate::HashInfoSimple;

/// Number of partition buckets (top 6 bits of canonical hash → 64 buckets).
pub const BUCKET_COUNT: usize = 64;

// ──────────────────────────────────────────────────────────────────────────────
// GPU-side data structures (must match WGSL structs exactly)
// ──────────────────────────────────────────────────────────────────────────────

/// One kmer occurrence uploaded to the GPU (24 bytes, aligned).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuKmerEntry {
    /// Canonical hash bits [31:0]
    pub canon_lo: u32,
    /// Canonical hash bits [63:32]
    pub canon_hi: u32,
    /// Non-canonical hash bits [31:0]
    pub nc_lo:    u32,
    /// Non-canonical hash bits [63:32]
    pub nc_hi:    u32,
    /// First/last bases (u8 padded to u32)
    pub bases:    u32,
    /// Padding to 24 bytes
    pub pad:      u32,
}

/// One filtered kmer result returned from the GPU (24 bytes, aligned).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuKmerOut {
    /// Canonical hash bits [31:0]
    pub canon_lo: u32,
    /// Canonical hash bits [63:32]
    pub canon_hi: u32,
    /// Non-canonical hash bits [31:0]
    pub nc_lo:    u32,
    /// Non-canonical hash bits [63:32]
    pub nc_hi:    u32,
    /// First/last bases
    pub bases:    u32,
    /// Occurrence count
    pub count:    u32,
}

const KMER_ENTRY_SIZE: u64 = std::mem::size_of::<GpuKmerEntry>() as u64;
const KMER_OUT_SIZE:   u64 = std::mem::size_of::<GpuKmerOut>()   as u64;

// ──────────────────────────────────────────────────────────────────────────────
// Pipeline cache
// ──────────────────────────────────────────────────────────────────────────────

struct GpuPipelines {
    radix_count:     wgpu::ComputePipeline,
    radix_prefix_g:  wgpu::ComputePipeline,
    radix_wg_offset: wgpu::ComputePipeline,
    radix_scatter:   wgpu::ComputePipeline,
    prefix_local:    wgpu::ComputePipeline,
    prefix_block:    wgpu::ComputePipeline,
    prefix_prop:     wgpu::ComputePipeline,
    boundary:        wgpu::ComputePipeline,
    count_seg:       wgpu::ComputePipeline,
    filter_mark:     wgpu::ComputePipeline,
    scatter_out:     wgpu::ComputePipeline,
    clear_buf:       wgpu::ComputePipeline, // P3: GPU zero-fill
    count_kmers:     wgpu::ComputePipeline, // Phase 1: count valid k-mers per read
    extract_kmers:   wgpu::ComputePipeline, // Phase 2: extract k-mers with ntHash
}

fn compile_pipelines(device: &wgpu::Device) -> GpuPipelines {
    let mk = |src: &str| device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });

    let sm_radix_count     = mk(include_str!("shaders/radix_count.wgsl"));
    let sm_radix_prefix_g  = mk(include_str!("shaders/radix_prefix_global.wgsl"));
    let sm_radix_wg_offset = mk(include_str!("shaders/radix_wg_offset.wgsl"));
    let sm_radix_scatter   = mk(include_str!("shaders/radix_scatter.wgsl"));
    let sm_prefix_sum      = mk(include_str!("shaders/prefix_sum.wgsl"));
    let sm_boundary        = mk(include_str!("shaders/boundary_detect.wgsl"));
    let sm_count_seg       = mk(include_str!("shaders/count_segments.wgsl"));
    let sm_filter_mark     = mk(include_str!("shaders/filter_mark.wgsl"));
    let sm_scatter_out     = mk(include_str!("shaders/scatter_output.wgsl"));
    let sm_clear_buf       = mk(include_str!("shaders/clear_buffer.wgsl"));
    let sm_count_kmers     = mk(include_str!("shaders/count_kmers.wgsl"));
    let sm_extract_kmers   = mk(include_str!("shaders/extract_kmers.wgsl"));

    let cp = |sm: &wgpu::ShaderModule, ep: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:               Some(ep),
            layout:              None,
            module:              sm,
            entry_point:         Some(ep),
            compilation_options: Default::default(),
            cache:               Default::default(),
        })
    };

    GpuPipelines {
        radix_count:     cp(&sm_radix_count,     "main"),
        radix_prefix_g:  cp(&sm_radix_prefix_g,  "main"),
        radix_wg_offset: cp(&sm_radix_wg_offset, "main"),
        radix_scatter:   cp(&sm_radix_scatter,   "main"),
        prefix_local:    cp(&sm_prefix_sum,       "local_scan"),
        prefix_block:    cp(&sm_prefix_sum,       "block_scan"),
        prefix_prop:     cp(&sm_prefix_sum,       "propagate"),
        boundary:        cp(&sm_boundary,         "main"),
        count_seg:       cp(&sm_count_seg,        "main"),
        filter_mark:     cp(&sm_filter_mark,      "main"),
        scatter_out:     cp(&sm_scatter_out,      "main"),
        clear_buf:       cp(&sm_clear_buf,        "main"),
        count_kmers:     cp(&sm_count_kmers,     "main"),
        extract_kmers:   cp(&sm_extract_kmers,   "main"),
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Buffer pool  (P2: pre-allocated at max bucket size, reused across all 64 buckets)
// ──────────────────────────────────────────────────────────────────────────────

/// Pre-allocated GPU buffers shared across all 64 bucket iterations.
struct GpuSortPool {
    // Sort ping-pong buffers
    buf_a:         wgpu::Buffer, // max_n * KMER_ENTRY_SIZE
    buf_b:         wgpu::Buffer, // max_n * KMER_ENTRY_SIZE
    // Radix sort histograms
    global_hist:   wgpu::Buffer, // 256 * 4
    wg_hist:       wgpu::Buffer, // max_num_wg * 256 * 4
    global_prefix: wgpu::Buffer, // 256 * 4
    wg_prefix:     wgpu::Buffer, // max_num_wg * 256 * 4
    // Per-pass uniforms
    pass_idx:      wgpu::Buffer, // 4 bytes, UNIFORM | COPY_DST
    num_wg:        wgpu::Buffer, // 4 bytes, UNIFORM | COPY_DST
    // count+filter output (P4: pre-allocated, avoids per-bucket alloc + blocking readback)
    output_buf:    wgpu::Buffer, // max_n * KMER_OUT_SIZE
    // min_count uniform (constant across all buckets)
    min_count:     wgpu::Buffer, // 4 bytes, UNIFORM
}

impl GpuSortPool {
    fn new(device: &wgpu::Device, max_n: u32, min_count_val: u32) -> Self {
        let max_num_wg  = max_n.div_ceil(256);
        let wg_hist_sz  = max_num_wg as u64 * 256 * 4;

        let mk = |label: &'static str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label:              Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let s = wgpu::BufferUsages::STORAGE;
        let sc = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let u  = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let rd = wgpu::BufferUsages::STORAGE  | wgpu::BufferUsages::COPY_DST;

        GpuSortPool {
            buf_a:         mk("pool_sort_a",        max_n as u64 * KMER_ENTRY_SIZE, sc),
            buf_b:         mk("pool_sort_b",        max_n as u64 * KMER_ENTRY_SIZE, sc),
            global_hist:   mk("pool_global_hist",   256 * 4,                        s | wgpu::BufferUsages::COPY_DST),
            wg_hist:       mk("pool_wg_hist",       wg_hist_sz,                     s | wgpu::BufferUsages::COPY_DST),
            global_prefix: mk("pool_global_prefix", 256 * 4,                        s | wgpu::BufferUsages::COPY_DST),
            wg_prefix:     mk("pool_wg_prefix",     wg_hist_sz,                     s | wgpu::BufferUsages::COPY_DST),
            pass_idx:      mk("pool_pass_idx",      4,                              u),
            num_wg:        mk("pool_num_wg",        4,                              u),
            output_buf:    mk("pool_output",        max_n as u64 * KMER_OUT_SIZE,   rd | wgpu::BufferUsages::COPY_SRC),
            min_count: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some("pool_min_count"),
                contents: bytemuck::bytes_of(&min_count_val),
                usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helper: inline prefix sum into an existing command encoder (no separate submit)
// ──────────────────────────────────────────────────────────────────────────────

/// Add up to 3 prefix-sum compute passes into `enc`.
/// `block_sums` must be pre-allocated with at least `ceil(n/256) * 4` bytes.
fn inline_prefix_sum(
    enc:        &mut wgpu::CommandEncoder,
    device:     &wgpu::Device,
    pipes:      &GpuPipelines,
    buf:        &wgpu::Buffer,
    n:          u32,
    block_sums: &wgpu::Buffer,
) {
    let num_wg = n.div_ceil(256);

    // Pass A: local_scan
    {
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   None,
            layout:  &pipes.prefix_local.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: block_sums.as_entire_binding() },
            ],
        });
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipes.prefix_local);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(num_wg, 1, 1);
    }

    if num_wg > 1 {
        // Pass B: block_scan
        {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:   None,
                layout:  &pipes.prefix_block.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 1, resource: block_sums.as_entire_binding() },
                ],
            });
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipes.prefix_block);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        // Pass C: propagate
        {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:   None,
                layout:  &pipes.prefix_prop.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: block_sums.as_entire_binding() },
                ],
            });
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipes.prefix_prop);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(num_wg, 1, 1);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Convert CPU tuple vec to GPU entry vec.
fn to_gpu_entries(bucket: &[(u64, u64, u8)]) -> Vec<GpuKmerEntry> {
    bucket.iter().map(|&(hc, hnc, b)| GpuKmerEntry {
        canon_lo: hc as u32,
        canon_hi: (hc >> 32) as u32,
        nc_lo:    hnc as u32,
        nc_hi:    (hnc >> 32) as u32,
        bases:    b as u32,
        pad:      0,
    }).collect()
}

/// Clear a buffer on the GPU within an existing command encoder.
/// Dispatches `ceil(n_u32 / 256)` workgroups (or 1, whichever is larger).
fn add_clear_pass(
    enc:    &mut wgpu::CommandEncoder,
    device: &wgpu::Device,
    pipes:  &GpuPipelines,
    buf:    &wgpu::Buffer,
    n_u32:  u32,  // number of u32 elements to clear
) {
    let wgs = n_u32.div_ceil(256).max(1);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:   None,
        layout:  &pipes.clear_buf.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
    });
    let mut pass = enc.begin_compute_pass(&Default::default());
    pass.set_pipeline(&pipes.clear_buf);
    pass.set_bind_group(0, &bg, &[]);
    pass.dispatch_workgroups(wgs, 1, 1);
}

// ──────────────────────────────────────────────────────────────────────────────
// P1+P2+P3: GPU radix sort using pool buffers + single encoder per pass + GPU clear
// ──────────────────────────────────────────────────────────────────────────────

/// Run GPU radix sort.  Entries are uploaded into `pool.buf_a`; after 8 passes
/// (even) the sorted result sits in `pool.buf_a`.
fn gpu_radix_sort(
    device:  &wgpu::Device,
    queue:   &wgpu::Queue,
    pipes:   &GpuPipelines,
    pool:    &GpuSortPool,
    entries: &[GpuKmerEntry],
) {
    let n       = entries.len() as u32;
    let num_wg  = n.div_ceil(256);

    // Update per-bucket uniforms
    queue.write_buffer(&pool.num_wg,  0, bytemuck::bytes_of(&num_wg));
    // Upload entries into buf_a (P2: reuses pre-allocated buffer)
    queue.write_buffer(&pool.buf_a,   0, bytemuck::cast_slice(entries));

    let bufs = [&pool.buf_a, &pool.buf_b];

    for pass in 0u32..8u32 {
        let src = bufs[(pass % 2) as usize];
        let dst = bufs[((pass + 1) % 2) as usize];

        // Update pass_idx uniform (4 bytes — negligible bandwidth)
        queue.write_buffer(&pool.pass_idx, 0, bytemuck::bytes_of(&pass));

        // Build bind groups for this pass
        let bg_count = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   None,
            layout:  &pipes.radix_count.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pool.global_hist.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pool.wg_hist.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pool.pass_idx.as_entire_binding() },
            ],
        });
        let bg_prefix_g = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   None,
            layout:  &pipes.radix_prefix_g.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pool.global_hist.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pool.global_prefix.as_entire_binding() },
            ],
        });
        let bg_wg_offset = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   None,
            layout:  &pipes.radix_wg_offset.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pool.wg_hist.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pool.wg_prefix.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pool.num_wg.as_entire_binding() },
            ],
        });
        let bg_scatter = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   None,
            layout:  &pipes.radix_scatter.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pool.global_prefix.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pool.wg_prefix.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pool.pass_idx.as_entire_binding() },
            ],
        });

        // P1: single encoder for all 8 compute passes of this radix pass
        let mut enc = device.create_command_encoder(&Default::default());

        // P3: GPU-side clear of histograms (replaces write_buffer zeros)
        add_clear_pass(&mut enc, device, pipes, &pool.global_hist,   256);
        add_clear_pass(&mut enc, device, pipes, &pool.wg_hist,       num_wg * 256);
        add_clear_pass(&mut enc, device, pipes, &pool.global_prefix, 256);
        add_clear_pass(&mut enc, device, pipes, &pool.wg_prefix,     num_wg * 256);

        // radix_count
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&pipes.radix_count);
            cp.set_bind_group(0, &bg_count, &[]);
            cp.dispatch_workgroups(num_wg, 1, 1);
        }
        // radix_prefix_global
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&pipes.radix_prefix_g);
            cp.set_bind_group(0, &bg_prefix_g, &[]);
            cp.dispatch_workgroups(1, 1, 1);
        }
        // radix_wg_offset
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&pipes.radix_wg_offset);
            cp.set_bind_group(0, &bg_wg_offset, &[]);
            cp.dispatch_workgroups(1, 1, 1);
        }
        // radix_scatter
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&pipes.radix_scatter);
            cp.set_bind_group(0, &bg_scatter, &[]);
            cp.dispatch_workgroups(num_wg, 1, 1);
        }

        queue.submit([enc.finish()]);
    }
    // After 8 passes (even), sorted result is in pool.buf_a ✓
}

// ──────────────────────────────────────────────────────────────────────────────
// P4: count+filter+scatter in one submit; no intermediate blocking readback
// ──────────────────────────────────────────────────────────────────────────────

/// Run count+filter pipeline on the sorted data in `pool.buf_a`.
/// Writes filtered results to `pool.output_buf`.  Returns the number of
/// filtered entries (a single 8-byte blocking readback after scatter).
async fn gpu_count_filter(
    device: &wgpu::Device,
    queue:  &wgpu::Queue,
    pipes:  &GpuPipelines,
    pool:   &GpuSortPool,
    n:      u32,
) -> u32 {
    let num_wg = n.div_ceil(256);

    // Per-call allocations: intermediate buffers that are bucket-size-dependent.
    // Not pooled to keep the pool memory footprint modest.
    let mk = |label: &'static str, size: u64, usage: wgpu::BufferUsages| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label), size, usage, mapped_at_creation: false,
        })
    };
    let sc = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
    let sd = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

    let boundary_buf     = mk("boundary",     n as u64 * 4, sc);
    let segment_id_buf   = mk("segment_id",   n as u64 * 4, sc);
    let seg_count_buf    = mk("seg_count",    n as u64 * 4, sd); // atomic u32, needs zero-init
    let output_mask_buf  = mk("output_mask",  n as u64 * 4, sc);
    let output_index_buf = mk("output_index", n as u64 * 4, sc);
    // block_sums for the two prefix sums
    let block_sums_a     = mk("block_sums_a", num_wg as u64 * 4, sd);
    let block_sums_b     = mk("block_sums_b", num_wg as u64 * 4, sd);
    // 2-u32 readback for output_count = last(output_index) + last(output_mask)
    let readback_buf = mk("readback_cnt", 8,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);

    // Build bind groups
    let bg_boundary = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:   None,
        layout:  &pipes.boundary.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: pool.buf_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: boundary_buf.as_entire_binding() },
        ],
    });
    let bg_count_seg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:   None,
        layout:  &pipes.count_seg.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: segment_id_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: seg_count_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: boundary_buf.as_entire_binding() },
        ],
    });
    let bg_filter_mark = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:   None,
        layout:  &pipes.filter_mark.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: boundary_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: segment_id_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: seg_count_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: output_mask_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: pool.min_count.as_entire_binding() },
        ],
    });
    let bg_scatter_out = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:   None,
        layout:  &pipes.scatter_out.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: pool.buf_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output_mask_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: output_index_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: segment_id_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: seg_count_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: pool.output_buf.as_entire_binding() },
        ],
    });

    // P4: single command encoder for the entire count+filter+scatter pipeline.
    // Scatter runs immediately after the prefix sums — no CPU stall between
    // "compute prefix" and "create output buffer + submit scatter".
    let mut enc = device.create_command_encoder(&Default::default());

    // Clear seg_count (atomic counters must start at zero)
    add_clear_pass(&mut enc, device, pipes, &seg_count_buf, n);

    // boundary_detect
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipes.boundary);
        pass.set_bind_group(0, &bg_boundary, &[]);
        pass.dispatch_workgroups(num_wg, 1, 1);
    }

    // Copy boundary → segment_id, then prefix_sum(segment_id)
    enc.copy_buffer_to_buffer(&boundary_buf, 0, &segment_id_buf, 0, n as u64 * 4);
    inline_prefix_sum(&mut enc, device, pipes, &segment_id_buf, n, &block_sums_a);

    // count_segments
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipes.count_seg);
        pass.set_bind_group(0, &bg_count_seg, &[]);
        pass.dispatch_workgroups(num_wg, 1, 1);
    }

    // filter_mark
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipes.filter_mark);
        pass.set_bind_group(0, &bg_filter_mark, &[]);
        pass.dispatch_workgroups(num_wg, 1, 1);
    }

    // Copy output_mask → output_index, then prefix_sum(output_index)
    enc.copy_buffer_to_buffer(&output_mask_buf, 0, &output_index_buf, 0, n as u64 * 4);
    inline_prefix_sum(&mut enc, device, pipes, &output_index_buf, n, &block_sums_b);

    // scatter_output → pool.output_buf (P4: pre-allocated, no intermediate readback)
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipes.scatter_out);
        pass.set_bind_group(0, &bg_scatter_out, &[]);
        pass.dispatch_workgroups(num_wg, 1, 1);
    }

    // Copy last elements for count readback (8 bytes total)
    let last = (n - 1) as u64 * 4;
    enc.copy_buffer_to_buffer(&output_index_buf, last, &readback_buf, 0, 4);
    enc.copy_buffer_to_buffer(&output_mask_buf,  last, &readback_buf, 4, 4);

    queue.submit([enc.finish()]);

    // Single blocking wait — GPU has already run scatter before we read the count
    let output_count = {
        let slice = readback_buf.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        #[cfg(not(target_arch = "wasm32"))]
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.await.unwrap().unwrap();
        let data = slice.get_mapped_range();
        let vals: &[u32] = bytemuck::cast_slice(&data);
        vals[0] + vals[1] // exclusive_prefix[last] + mask[last]
    };
    readback_buf.unmap();

    output_count
}

/// Readback filtered results from `pool.output_buf` to a CPU Vec.
async fn readback_output(
    device: &wgpu::Device,
    queue:  &wgpu::Queue,
    pool:   &GpuSortPool,
    count:  u32,
) -> Vec<GpuKmerOut> {
    if count == 0 { return Vec::new(); }

    let size = count as u64 * KMER_OUT_SIZE;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label:              Some("staging"),
        size,
        usage:              wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(&pool.output_buf, 0, &staging, 0, size);
    queue.submit([enc.finish()]);

    let slice = staging.slice(..);
    let (tx, rx) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    #[cfg(not(target_arch = "wasm32"))]
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.await.unwrap().unwrap();

    let data = slice.get_mapped_range();
    let result: Vec<GpuKmerOut> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    result
}

/// Process one bucket: GPU radix sort → count+filter → readback.
async fn process_bucket(
    device:    &wgpu::Device,
    queue:     &wgpu::Queue,
    pipes:     &GpuPipelines,
    pool:      &GpuSortPool,
    bucket:    &[(u64, u64, u8)],
) -> Vec<GpuKmerOut> {
    if bucket.is_empty() { return Vec::new(); }

    let entries = to_gpu_entries(bucket);
    let n = entries.len() as u32;

    // Sort (P1+P2+P3)
    gpu_radix_sort(device, queue, pipes, pool, &entries);

    // Count + filter (P4)
    let out_count = gpu_count_filter(device, queue, pipes, pool, n).await;

    // Readback
    readback_output(device, queue, pool, out_count).await
}

// ──────────────────────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────────────────────

/// GPU-accelerated radix sort + count + filter over partitioned kmer buckets.
///
/// Returns `themap` accumulating all kmers with count ≥ min_count.
pub async fn gpu_sort_count_filter(
    buckets:    &[Vec<(u64, u64, u8)>; BUCKET_COUNT],
    min_count:  u16,
    _do_fit:    bool,
    _out_path:  &mut Option<PathBuf>,
    power_pref: wgpu::PowerPreference,
) -> HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>> {
    // Init GPU
    #[cfg(target_arch = "wasm32")]
    let backends = wgpu::Backends::BROWSER_WEBGPU;
    #[cfg(not(target_arch = "wasm32"))]
    let backends = wgpu::Backends::all();
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends, ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: power_pref, ..Default::default()
        })
        .await
        .expect("No GPU adapter found");

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("sparrowhawk_gpu"), ..Default::default()
        })
        .await
        .expect("Failed to get GPU device");

    crate::logw(&format!("GPU: {}", adapter.get_info().name), Some("info"));

    // Compile pipelines once
    let pipes = compile_pipelines(&device);

    // P2: create pool at max bucket size (one allocation for all 64 buckets)
    let max_n = buckets.iter().map(|b| b.len()).max().unwrap_or(0) as u32;
    if max_n == 0 {
        return HashMap::with_hasher(BuildHasherDefault::default());
    }
    let pool = GpuSortPool::new(&device, max_n, min_count as u32);

    let mut themap: HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>> =
        HashMap::with_hasher(BuildHasherDefault::default());

    for (bucket_idx, bucket) in buckets.iter().enumerate() {
        if bucket.is_empty() { continue; }
        crate::logw(
            &format!("GPU bucket {}/{} ({} entries)", bucket_idx + 1, BUCKET_COUNT, bucket.len()),
            Some("info"),
        );

        let results = process_bucket(&device, &queue, &pipes, &pool, bucket).await;

        for r in results {
            let hc  = (r.canon_lo as u64) | ((r.canon_hi as u64) << 32);
            let hnc = (r.nc_lo    as u64) | ((r.nc_hi    as u64) << 32);
            let b   = r.bases as u8;
            let cnt = r.count.min(u16::MAX as u32) as u16;

            themap.entry(hc).or_insert(HashInfoSimple {
                hnc,
                b,
                pre:    Vec::new(),
                post:   Vec::new(),
                counts: cnt,
            });
        }
    }

    themap
}

// ──────────────────────────────────────────────────────────────────────────────
// GPU k-mer extraction (Phase 1 + Phase 2) — raw sequences → GpuKmerEntry
// ──────────────────────────────────────────────────────────────────────────────

/// Number of reads processed per GPU extraction dispatch.
const EXTR_CHUNK_READS: usize = 20_000;

/// Uniform params layout for count_kmers.wgsl and extract_kmers.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ExtrParams {
    n_reads:  u32,
    k:        u32,
    min_qual: u32,
    _pad:     u32,  // pad to 16 bytes for uniform buffer alignment
}

/// Pad a byte slice to the nearest multiple of 4 (appending zero bytes).
fn pad_to_u32(data: &[u8]) -> Vec<u8> {
    let pad = (4 - data.len() % 4) % 4;
    let mut out = data.to_vec();
    out.extend_from_slice(&[0u8; 3][..pad]);
    out
}

/// Phase 1: run count_kmers.wgsl over one chunk of reads.
/// Returns `kmer_count[i]` = number of valid k-mer windows in read i.
async fn gpu_run_phase1(
    device:       &wgpu::Device,
    queue:        &wgpu::Queue,
    pipes:        &GpuPipelines,
    seq_bytes:    &[u8],     // raw ASCII sequence data for this chunk
    qual_bytes:   &[u8],     // raw ASCII quality data for this chunk
    read_offsets: &[u32],    // byte offset of each read (relative to chunk start)
    read_lengths: &[u32],    // length of each read in bases
    k:            u32,
    min_qual:     u32,
) -> Vec<u32> {
    let n_reads = read_lengths.len() as u32;
    let num_wg  = n_reads.div_ceil(256);

    let seq_padded  = pad_to_u32(seq_bytes);
    let qual_padded = pad_to_u32(qual_bytes);

    let use_sc  = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let use_scr = use_sc | wgpu::BufferUsages::COPY_SRC;

    let seq_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("p1_seq"), contents: &seq_padded, usage: use_sc,
    });
    let qual_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("p1_qual"), contents: &qual_padded, usage: use_sc,
    });
    let off_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("p1_offsets"), contents: bytemuck::cast_slice(read_offsets), usage: use_sc,
    });
    let len_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("p1_lengths"), contents: bytemuck::cast_slice(read_lengths), usage: use_sc,
    });
    let cnt_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("p1_kmer_count"), size: n_reads as u64 * 4, usage: use_scr,
        mapped_at_creation: false,
    });
    let params = ExtrParams { n_reads, k, min_qual, _pad: 0 };
    let par_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("p1_params"), contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:   None,
        layout:  &pipes.count_kmers.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: seq_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: qual_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: off_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: len_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: cnt_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: par_buf.as_entire_binding() },
        ],
    });

    let rb = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("p1_rb"), size: n_reads as u64 * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipes.count_kmers);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(num_wg, 1, 1);
    }
    enc.copy_buffer_to_buffer(&cnt_buf, 0, &rb, 0, n_reads as u64 * 4);
    queue.submit([enc.finish()]);

    let slice = rb.slice(..);
    let (tx, rx) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    #[cfg(not(target_arch = "wasm32"))]
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.await.unwrap().unwrap();
    let data   = slice.get_mapped_range();
    let counts = bytemuck::cast_slice::<u8, u32>(&data).to_vec();
    drop(data);
    rb.unmap();
    counts
}

/// Phase 2: run extract_kmers.wgsl over one chunk of reads.
/// `kmer_offsets[i]` is the exclusive prefix sum of Phase-1 counts.
/// Returns the extracted `GpuKmerEntry` records (unsorted).
async fn gpu_run_phase2(
    device:        &wgpu::Device,
    queue:         &wgpu::Queue,
    pipes:         &GpuPipelines,
    seq_bytes:     &[u8],
    qual_bytes:    &[u8],
    read_offsets:  &[u32],
    read_lengths:  &[u32],
    kmer_offsets:  &[u32],
    total_kmers:   u32,
    k:             u32,
    min_qual:      u32,
) -> Vec<GpuKmerEntry> {
    let n_reads = read_lengths.len() as u32;
    let num_wg  = n_reads.div_ceil(256);

    let seq_padded  = pad_to_u32(seq_bytes);
    let qual_padded = pad_to_u32(qual_bytes);

    let use_sc  = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let use_scr = use_sc | wgpu::BufferUsages::COPY_SRC;

    let seq_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("p2_seq"), contents: &seq_padded, usage: use_sc,
    });
    let qual_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("p2_qual"), contents: &qual_padded, usage: use_sc,
    });
    let off_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("p2_offsets"), contents: bytemuck::cast_slice(read_offsets), usage: use_sc,
    });
    let len_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("p2_lengths"), contents: bytemuck::cast_slice(read_lengths), usage: use_sc,
    });
    let koff_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("p2_kmer_offsets"), contents: bytemuck::cast_slice(kmer_offsets), usage: use_sc,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("p2_output"), size: total_kmers as u64 * KMER_ENTRY_SIZE,
        usage: use_scr, mapped_at_creation: false,
    });
    let params = ExtrParams { n_reads, k, min_qual, _pad: 0 };
    let par_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("p2_params"), contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:   None,
        layout:  &pipes.extract_kmers.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: seq_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: qual_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: off_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: len_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: koff_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: par_buf.as_entire_binding() },
        ],
    });

    let rb = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("p2_rb"), size: total_kmers as u64 * KMER_ENTRY_SIZE,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipes.extract_kmers);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(num_wg, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &rb, 0, total_kmers as u64 * KMER_ENTRY_SIZE);
    queue.submit([enc.finish()]);

    let slice = rb.slice(..);
    let (tx, rx) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    #[cfg(not(target_arch = "wasm32"))]
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.await.unwrap().unwrap();
    let data    = slice.get_mapped_range();
    let entries = bytemuck::cast_slice::<u8, GpuKmerEntry>(&data).to_vec();
    drop(data);
    rb.unmap();
    entries
}

/// Sort extracted k-mer entries by canonical hash, then update a
/// `HashMap<canon_hash → (count, nc_hash, bases)>` countmap.
fn update_countmap_from_entries(
    entries:  &mut Vec<GpuKmerEntry>,
    countmap: &mut HashMap<u64, (u16, u64, u8), BuildHasherDefault<NoHashHasher<u64>>>,
) {
    if entries.is_empty() { return; }

    entries.sort_unstable_by(|a, b| {
        let ha = (a.canon_lo as u64) | ((a.canon_hi as u64) << 32);
        let hb = (b.canon_lo as u64) | ((b.canon_hi as u64) << 32);
        ha.cmp(&hb)
    });

    let mut i    = 0usize;
    let mut c    = 0u16;
    let mut canon = (entries[0].canon_lo as u64) | ((entries[0].canon_hi as u64) << 32);

    while i < entries.len() {
        let hc = (entries[i].canon_lo as u64) | ((entries[i].canon_hi as u64) << 32);
        if hc != canon {
            let hnc = (entries[i-1].nc_lo as u64) | ((entries[i-1].nc_hi as u64) << 32);
            let e = countmap.entry(canon).or_insert((0, hnc, entries[i-1].bases as u8));
            e.0 = e.0.saturating_add(c);
            canon = hc;
            c = 1;
        } else {
            c = c.saturating_add(1);
        }
        i += 1;
    }
    // Flush last group
    let last = entries.len() - 1;
    let hnc  = (entries[last].nc_lo as u64) | ((entries[last].nc_hi as u64) << 32);
    let e    = countmap.entry(canon).or_insert((0, hnc, entries[last].bases as u8));
    e.0 = e.0.saturating_add(c);
}

/// GPU-accelerated k-mer extraction pipeline.
///
/// Processes reads in chunks of `EXTR_CHUNK_READS`:
///   Phase 1 (GPU)  — count valid k-mer windows per read (`count_kmers.wgsl`)
///   Phase 1b (CPU) — exclusive prefix sum → `kmer_offsets`; allocate output buffer
///   Phase 2 (GPU)  — extract k-mers with ntHash (`extract_kmers.wgsl`)
///   CPU            — sort extracted entries; update accumulator countmap
///
/// After all chunks: filter by `min_count`, return `themap`.
///
/// `seq_data` / `qual_data` are raw ASCII bytes (all reads concatenated).
/// `read_offsets[i]` is the byte position of read i in `seq_data`.
pub async fn gpu_extract_count_filter(
    seq_data:     &[u8],
    qual_data:    &[u8],
    read_offsets: &[u32],
    read_lengths: &[u32],
    k:            u32,
    min_qual:     u32,
    min_count:    u16,
    power_pref:   wgpu::PowerPreference,
) -> HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>> {
    #[cfg(target_arch = "wasm32")]
    let backends = wgpu::Backends::BROWSER_WEBGPU;
    #[cfg(not(target_arch = "wasm32"))]
    let backends = wgpu::Backends::all();
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends, ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: power_pref, ..Default::default()
        })
        .await
        .expect("No GPU adapter found");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("sparrowhawk_extr"), ..Default::default()
        })
        .await
        .expect("Failed to get GPU device");
    crate::logw(&format!("GPU extraction: {}", adapter.get_info().name), Some("info"));

    let pipes = compile_pipelines(&device);

    let n_reads = read_lengths.len();
    let n_chunks = n_reads.div_ceil(EXTR_CHUNK_READS);

    let mut countmap: HashMap<u64, (u16, u64, u8), BuildHasherDefault<NoHashHasher<u64>>> =
        HashMap::with_hasher(BuildHasherDefault::default());

    for chunk_idx in 0..n_chunks {
        let chunk_start = chunk_idx * EXTR_CHUNK_READS;
        let chunk_end   = (chunk_start + EXTR_CHUNK_READS).min(n_reads);

        crate::logw(
            &format!("GPU extract chunk {}/{}", chunk_idx + 1, n_chunks),
            Some("info"),
        );

        // Byte range for this chunk's raw data
        let seq_start = read_offsets[chunk_start] as usize;
        let seq_end   = if chunk_end < n_reads {
            read_offsets[chunk_end] as usize
        } else {
            seq_data.len()
        };
        let chunk_seq  = &seq_data[seq_start..seq_end];
        let chunk_qual = &qual_data[seq_start..seq_end];

        // Rebase offsets to be relative to this chunk's seq start
        let chunk_offsets: Vec<u32> = read_offsets[chunk_start..chunk_end]
            .iter()
            .map(|&o| o - seq_start as u32)
            .collect();
        let chunk_lengths = &read_lengths[chunk_start..chunk_end];

        // ── Phase 1: GPU counts ──────────────────────────────────────────────
        let kmer_count = gpu_run_phase1(
            &device, &queue, &pipes,
            chunk_seq, chunk_qual, &chunk_offsets, chunk_lengths,
            k, min_qual,
        ).await;

        // ── Phase 1b: CPU prefix sum ─────────────────────────────────────────
        let mut kmer_offsets_vec = Vec::with_capacity(chunk_end - chunk_start);
        let mut running = 0u32;
        for &c in &kmer_count {
            kmer_offsets_vec.push(running);
            running = running.saturating_add(c);
        }
        let total_kmers = running;
        if total_kmers == 0 { continue; }

        // ── Phase 2: GPU extraction ──────────────────────────────────────────
        let mut extracted = gpu_run_phase2(
            &device, &queue, &pipes,
            chunk_seq, chunk_qual, &chunk_offsets, chunk_lengths,
            &kmer_offsets_vec, total_kmers, k, min_qual,
        ).await;

        // ── CPU sort + accumulate ────────────────────────────────────────────
        update_countmap_from_entries(&mut extracted, &mut countmap);
    }

    // Final filter: only keep k-mers with count ≥ min_count
    let mut themap: HashMap<u64, HashInfoSimple, BuildHasherDefault<NoHashHasher<u64>>> =
        HashMap::with_hasher(BuildHasherDefault::default());
    for (hc, (cnt, hnc, b)) in countmap {
        if cnt >= min_count {
            themap.insert(hc, HashInfoSimple {
                hnc,
                b,
                pre:    Vec::new(),
                post:   Vec::new(),
                counts: cnt,
            });
        }
    }
    themap
}

/// Returns [(preference_index, adapter_name), …] for the UI dropdown.
/// Index 0 = None (browser decides), Index 1 = HighPerformance, Index 2 = LowPower.
pub async fn enumerate_gpu_adapters() -> Vec<(u32, String)> {
    #[cfg(target_arch = "wasm32")]
    let backends = wgpu::Backends::BROWSER_WEBGPU;
    #[cfg(not(target_arch = "wasm32"))]
    let backends = wgpu::Backends::all();
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends, ..Default::default()
    });

    let mut result = Vec::new();
    let mut seen   = std::collections::HashSet::new();
    for (idx, pref) in [
        (0u32, wgpu::PowerPreference::None),
        (1u32, wgpu::PowerPreference::HighPerformance),
        (2u32, wgpu::PowerPreference::LowPower),
    ] {
        if let Ok(adapter) = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: pref, ..Default::default()
        }).await {
            let name = adapter.get_info().name;
            if seen.insert(name.clone()) {
                result.push((idx, name));
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(hc: u64, hnc: u64, bases: u8) -> GpuKmerEntry {
        GpuKmerEntry {
            canon_lo: hc as u32,
            canon_hi: (hc >> 32) as u32,
            nc_lo:    hnc as u32,
            nc_hi:    (hnc >> 32) as u32,
            bases:    bases as u32,
            pad:      0,
        }
    }

    fn empty_countmap() -> HashMap<u64, (u16, u64, u8), BuildHasherDefault<NoHashHasher<u64>>> {
        HashMap::with_hasher(BuildHasherDefault::default())
    }

    // ── pad_to_u32 ────────────────────────────────────────────────────────────

    #[test]
    fn pad_to_u32_empty() {
        assert_eq!(pad_to_u32(&[]).len(), 0);
    }

    #[test]
    fn pad_to_u32_lengths() {
        for len in 0usize..=16 {
            let out = pad_to_u32(&vec![1u8; len]);
            assert_eq!(out.len() % 4, 0, "len={len} not multiple of 4");
            if len == 0 {
                assert_eq!(out.len(), 0);
            } else {
                let expected = len.div_ceil(4) * 4;
                assert_eq!(out.len(), expected, "len={len}");
            }
        }
    }

    #[test]
    fn pad_to_u32_padding_bytes_are_zero() {
        let data = vec![0xFFu8; 3];
        let out = pad_to_u32(&data);
        assert_eq!(out.len(), 4);
        assert_eq!(out[3], 0); // padding byte is zero
    }

    #[test]
    fn pad_to_u32_exact_multiple_unchanged() {
        let data = vec![0xAAu8; 8];
        let out = pad_to_u32(&data);
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|&b| b == 0xAA));
    }

    // ── to_gpu_entries ────────────────────────────────────────────────────────

    #[test]
    fn to_gpu_entries_splits_hash_correctly() {
        let hc: u64 = 0xDEADBEEF_CAFEBABEu64;
        let hnc: u64 = 0x1234_5678_9ABC_DEF0u64;
        let entries = to_gpu_entries(&[(hc, hnc, 0b10_01u8)]);
        assert_eq!(entries[0].canon_lo, 0xCAFEBABEu32);
        assert_eq!(entries[0].canon_hi, 0xDEADBEEFu32);
        assert_eq!(entries[0].nc_lo,   0x9ABC_DEF0u32);
        assert_eq!(entries[0].nc_hi,   0x1234_5678u32);
        assert_eq!(entries[0].bases,   0b10_01u32);
        assert_eq!(entries[0].pad,     0);
    }

    #[test]
    fn to_gpu_entries_round_trip() {
        let hc: u64 = 0xDEADBEEF_CAFEBABEu64;
        let entries = to_gpu_entries(&[(hc, 0, 0)]);
        let reconstructed = (entries[0].canon_lo as u64) | ((entries[0].canon_hi as u64) << 32);
        assert_eq!(reconstructed, hc);
    }

    // ── GpuKmerEntry / GpuKmerOut layout ─────────────────────────────────────

    #[test]
    fn gpu_kmer_entry_size_and_align() {
        assert_eq!(std::mem::size_of::<GpuKmerEntry>(), 24);
        assert_eq!(std::mem::align_of::<GpuKmerEntry>(), 4);
    }

    #[test]
    fn gpu_kmer_out_size_and_align() {
        assert_eq!(std::mem::size_of::<GpuKmerOut>(), 24);
        assert_eq!(std::mem::align_of::<GpuKmerOut>(), 4);
    }

    // ── update_countmap_from_entries ──────────────────────────────────────────

    #[test]
    fn update_countmap_empty_input() {
        let mut entries = Vec::new();
        let mut map = empty_countmap();
        update_countmap_from_entries(&mut entries, &mut map);
        assert!(map.is_empty());
    }

    #[test]
    fn update_countmap_single_entry() {
        let h = 0x1111_2222_3333_4444u64;
        let nc = 0xAAAA_BBBB_CCCC_DDDDu64;
        let mut entries = vec![make_entry(h, nc, 0b01)];
        let mut map = empty_countmap();
        update_countmap_from_entries(&mut entries, &mut map);
        assert!(map.contains_key(&h));
        assert_eq!(map[&h].0, 1);
    }

    #[test]
    fn update_countmap_two_hashes() {
        let h1 = 0x0000_0000_0000_0001u64;
        let h2 = 0x0000_0000_0000_0002u64;
        let mut entries = vec![make_entry(h1, 0, 0), make_entry(h1, 0, 0), make_entry(h2, 0, 0)];
        let mut map = empty_countmap();
        update_countmap_from_entries(&mut entries, &mut map);
        assert_eq!(map[&h1].0, 2);
        assert_eq!(map[&h2].0, 1);
    }

    #[test]
    fn update_countmap_five_same_hash() {
        let h = 0xFFFF_FFFF_FFFF_FFFEu64;
        let mut entries: Vec<_> = (0..5).map(|_| make_entry(h, 0, 0)).collect();
        let mut map = empty_countmap();
        update_countmap_from_entries(&mut entries, &mut map);
        assert_eq!(map[&h].0, 5);
    }

    #[test]
    fn update_countmap_accumulates_across_calls() {
        let h = 0x9999_8888_7777_6666u64;
        let mut map = empty_countmap();
        let mut e1: Vec<_> = (0..3).map(|_| make_entry(h, 0, 0)).collect();
        update_countmap_from_entries(&mut e1, &mut map);
        let mut e2: Vec<_> = (0..2).map(|_| make_entry(h, 0, 0)).collect();
        update_countmap_from_entries(&mut e2, &mut map);
        assert_eq!(map[&h].0, 5);
    }
}
