use core::hint::black_box;
use openvm as _;

use openvm_keccak256::keccak256;

const INPUT_LENGTH_BYTES: usize = 100 * 1024; // 100 KB

pub fn main() {
    let mut input = Vec::with_capacity(INPUT_LENGTH_BYTES);

    // Initialize with pseudo-random values
    let mut val: u64 = 1;
    for _ in 0..INPUT_LENGTH_BYTES {
        input.push(val as u8);
        val = ((val.wrapping_mul(8191)) << 7) ^ val;
    }

    // Prevent optimizer from optimizing away the computation
    let input = black_box(input);
    black_box(keccak256(&input));
}
