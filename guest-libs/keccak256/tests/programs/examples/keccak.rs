#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;
use core::hint::black_box;

use openvm::io::read;
use openvm_keccak256::keccak256;

openvm::entry!(main);

pub fn main() {
    let input: Vec<u8> = read();
    let expected_output: Vec<u8> = read();
    let output = keccak256(&input).to_vec();

    if output != expected_output {
        panic!(
            "input: {:?}, expected_output: {:?}, output: {:?}",
            input, expected_output, output
        );
    }
}
