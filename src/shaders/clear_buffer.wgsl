// GPU-side buffer fill with zeros.
// Dispatched as ceil(n/256) workgroups of 256 threads.
// Uses arrayLength to guard — safe to dispatch with any number of workgroups ≤ ceil(size/1024).

@group(0) @binding(0) var<storage, read_write> buf: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x < arrayLength(&buf) {
        buf[gid.x] = 0u;
    }
}
