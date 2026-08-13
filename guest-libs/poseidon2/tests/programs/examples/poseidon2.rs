#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;
use core::hint::black_box;

use openvm::io::read;
use openvm_poseidon2::hash_u32s;

openvm::entry!(main);

pub fn main() {
    let num_test_vectors: u32 = read();
    for _ in 0..num_test_vectors {
        let input: Vec<u32> = read();
        let expected_output: Vec<u32> = read();
        let output = hash_u32s(black_box(&input)).to_vec();

        if output != expected_output {
            panic!(
                "input: {:?}, expected_output: {:?}, output: {:?}",
                input, expected_output, output
            );
        }
    }
}
