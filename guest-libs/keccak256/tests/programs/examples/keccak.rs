#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;
use core::hint::black_box;

use openvm::io::read;
use openvm_keccak256::keccak256;

openvm::entry!(main);

pub fn main() {
    let num_test_vectors: u32 = read();
    for _ in 0..num_test_vectors {
        let input: Vec<u8> = read();
        let expected_output: Vec<u8> = read();
        let output = keccak256(black_box(&input)).to_vec();

        if output != expected_output {
            panic!(
                "input: {:?}, expected_output: {:?}, output: {:?}",
                input, expected_output, output
            );
        }
    }
}
