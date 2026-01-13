//! BLAKE3 benchmark for OpenVM
//!
//! This benchmark measures BLAKE3 hashing performance.
//! Currently uses pure Rust implementation (no native extension).
//! After adding native BLAKE3 extension, re-run to compare cycle counts.

use core::hint::black_box;
use openvm as _;

use openvm_blake3::blake3;

// Use 32KB input for benchmark (smaller than sha256 benchmark due to pure Rust implementation)
// Once native extension is added, we can increase this
const INPUT_LENGTH_BYTES: usize = 32 * 1024;

pub fn main() {
    let mut input = Vec::with_capacity(INPUT_LENGTH_BYTES);

    // Initialize with pseudo-random values (same pattern as sha256 benchmark)
    let mut val: u64 = 1;
    for _ in 0..INPUT_LENGTH_BYTES {
        input.push(val as u8);
        val = ((val.wrapping_mul(8191)) << 7) ^ val;
    }

    // Prevent optimizer from optimizing away the computation
    black_box(blake3(&black_box(input)));
}
