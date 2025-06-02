use core::hint::black_box;
use openvm as _;

use openvm_sha256_guest::sha256;

const ITERATIONS: usize = 150_000;

pub fn main() {
    // Initialize with hash of an empty vector
    let mut hash = black_box(sha256(&vec![]));

    // Iteratively apply sha256
    for _ in 0..ITERATIONS {
        hash = sha256(&hash);
    }

    // Prevent optimizer from optimizing away the computation
    black_box(hash);
}
