#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;
use core::hint::black_box;

use hex::FromHex;
use openvm::io::println;
use openvm_sha256_guest::sha256;

openvm::entry!(main);

pub fn main() {
    let test_vectors = [
        ("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"), // Len = 0
        ("CC", "1dd8312636f6a0bf3d21fa2855e63072507453e93a5ced4301b364e91c9d87d6"), // Len = 8
        ("deadaf", "a268cb1a3bce99d1d9a281728e3c65316219366b60909c5c50fe5f783bb500af"), // Len = 24
        ("acd4d7fc5906116fe15021451e77fbf025d7c312ddb745b98fd4e8c7aa403c4ec26288b5641407c62efa9870ac3d5e18e7780a155d20213e38b9d7acc55c51d6", "27ec5083c93838b85ddbd27ac6a9b188bdf37a2ef9e5fe59347cb1bb897c178d"), // Len = 512
        ("4d11ec5b57d8989566b864df237ca4ad6742b23cda95172fd5707efa5d4969338f2ca557473210722da038864064bf5e5feb826c8d3928819455490b067b813d45a3e36ad5b9f0a049e68961a37bb5607bb4951d33e00a3ee13ec19c3a4cdbd8580892fce193bd2d9deb1f8045f4c4349fb4a45fb39b2f0d435d3758c8", "fa8bcb9f8112dbf0e6b9a74effe2a2fb9c1bc1612f52006dbd7af93b2deb4365"), // Len = 1000
    ];
    for (input, expected_output) in test_vectors.iter() {
        let input = Vec::from_hex(input).unwrap();
        let expected_output = Vec::from_hex(expected_output).unwrap();
        let output = sha256(&black_box(input));
        if output != *expected_output {
            panic!();
        }
    }

    println("PASS");
}
