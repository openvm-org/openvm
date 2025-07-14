use core::hint::black_box;
use openvm as _;

use openvm_keccak256::keccak256;

const ITERATIONS: usize = 10_000;

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
