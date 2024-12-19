#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;
use core::hint::black_box;

use openvm_sha256_guest::sha256;
use hex::FromHex;

openvm::entry!(main);

pub fn main() {
    let test_vectors = [
        ("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"), // ShortMsgKAT_256 Len = 0
    ];
    for (input, expected_output) in test_vectors.iter() {
        let input = Vec::from_hex(input).unwrap();
        let expected_output = Vec::from_hex(expected_output).unwrap();
        let output = sha256(&black_box(input));
        if output != *expected_output {
            panic!();
        }
    }
}
