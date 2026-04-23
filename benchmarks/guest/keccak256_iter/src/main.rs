#![cfg_attr(target_os = "none", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
use alloc::{vec, vec::Vec};

openvm::entry!(main);

use core::hint::black_box;
use openvm as _;

use openvm_keccak256::keccak256;

const ITERATIONS: usize = 65_000;

pub fn main() {
    // Initialize with hash of an empty vector
    let mut hash = black_box(keccak256(&vec![]));

    // Iteratively apply keccak256
    for _ in 0..ITERATIONS {
        hash = keccak256(&hash);
    }

    // Prevent optimizer from optimizing away the computation
    black_box(hash);
}
