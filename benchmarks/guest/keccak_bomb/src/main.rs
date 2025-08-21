use core::hint::black_box;
use openvm as _;

use openvm_keccak256::keccak256;

const INPUT_LENGTH_BYTES: usize = 500 * 1024 * 1024; // 500MB

pub fn main() {
    let mut input = Vec::with_capacity(INPUT_LENGTH_BYTES);
    unsafe {
        input.set_len(INPUT_LENGTH_BYTES);
    }

    // Prevent optimizer from optimizing away the computation
    let input = black_box(input);
    black_box(keccak256(&input));
}
