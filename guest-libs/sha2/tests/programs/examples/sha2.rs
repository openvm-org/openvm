#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;
use core::hint::black_box;

use openvm::io::read;
use openvm_sha2::{Digest, Sha256, Sha384, Sha512};

openvm::entry!(main);

pub fn main() {
    let sha2_type: u32 = read();
    let num_test_vectors: u32 = read();
    for _ in 0..num_test_vectors {
        let input: Vec<u8> = read();
        let expected_output: Vec<u8> = read();

        let output = match sha2_type {
            256 => Sha256::digest(black_box(&input)).to_vec(),
            384 => Sha384::digest(black_box(&input)).to_vec(),
            512 => Sha512::digest(black_box(&input)).to_vec(),
            _ => panic!(),
        };
        if output != expected_output {
            panic!(
                "input: {:?}, expected_output: {:?}, output: {:?}",
                input, expected_output, output
            );
        }
    }
}
