#![cfg_attr(any(target_os = "none", target_os = "openvm"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
use alloc::{vec, vec::Vec};

openvm::entry!(main);

use core::hint::black_box;

use openvm as _;
use openvm_sha2::Sha256;

const INPUT_LENGTH_BYTES: usize = 384 * 1024;

pub fn main() {
    let mut input = Vec::with_capacity(INPUT_LENGTH_BYTES);

    // Initialize with pseudo-random values
    let mut val: u64 = 1;
    for _ in 0..INPUT_LENGTH_BYTES {
        input.push(val as u8);
        val = ((val.wrapping_mul(8191)) << 7) ^ val;
    }

    // Prevent optimizer from optimizing away the computation
    let mut sha256 = Sha256::new();
    sha256.update(black_box(&input));
    black_box(sha256.finalize());
}
