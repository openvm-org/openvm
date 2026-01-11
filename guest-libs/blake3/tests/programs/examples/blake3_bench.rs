//! BLAKE3 benchmark program for OpenVM
//!
//! This program hashes various inputs using BLAKE3 and verifies against known test vectors.
//! Run this to measure cycle count for BLAKE3 hashing without native extension support.

#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;
use core::hint::black_box;
use openvm_blake3::blake3;

openvm::entry!(main);

pub fn main() {
    // Test vectors from BLAKE3 specification
    let test_vectors = [
        // (input_hex, expected_hash_hex)
        (
            "",  // empty input
            "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
        ),
        (
            "00",  // single zero byte
            "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213"
        ),
        (
            "000102",  // 3 bytes
            "e1be4d7a8ab5560aa4199eea339849ba8e293d55ca0a81006726d184519e647f"
        ),
    ];

    // Run test vectors
    for (input_hex, expected_hex) in test_vectors.iter() {
        let input = hex::decode(input_hex).unwrap_or_else(|_| Vec::new());
        let expected = hex::decode(expected_hex).unwrap();
        
        // Use black_box to prevent optimizer from removing the computation
        let output = blake3(&black_box(input));
        
        if output.as_slice() != expected.as_slice() {
            panic!("BLAKE3 hash mismatch!");
        }
    }

    // Benchmark: hash "hello world" (common test case)
    let hello_world = b"hello world";
    let hash = blake3(black_box(hello_world));
    
    // Verify hello world hash
    let expected_hello = hex::decode(
        "d74981efa70a0c880b8d8c1985d075dbcbf679b99a5f9914e5aaf96b831a9e24"
    ).unwrap();
    
    if hash.as_slice() != expected_hello.as_slice() {
        panic!("hello world hash mismatch!");
    }

    // Benchmark: hash 1KB of zeros (shows scaling)
    let zeros_1kb = [0u8; 1024];
    let _ = blake3(black_box(&zeros_1kb));
}
