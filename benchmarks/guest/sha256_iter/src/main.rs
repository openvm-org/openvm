#![cfg_attr(any(target_os = "none", target_os = "openvm"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

openvm::entry!(main);

use core::hint::black_box;

use openvm as _;
use openvm_sha2::Sha256;

const ITERATIONS: usize = 150_000;

fn sha256_digest(input: &[u8]) -> [u8; 32] {
    let mut sha256 = Sha256::new();
    sha256.update(black_box(input));
    sha256.finalize()
}

pub fn main() {
    // Initialize with hash of an empty vector
    let mut hash = black_box(sha256_digest(&[]));

    // Iteratively apply sha256
    for _ in 0..ITERATIONS {
        hash = black_box(sha256_digest(&hash));
    }

    // Prevent optimizer from optimizing away the computation
    black_box(hash);
}
