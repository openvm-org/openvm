#![cfg_attr(all(target_os = "zkvm", not(feature = "std")), no_main)]
#![cfg_attr(all(target_os = "zkvm", not(feature = "std")), no_std)]

extern crate alloc;

use alloc::{format, string::String, vec};

use openvm_sha2::{Digest, Sha256};

openvm::entry!(main);

fn println(s: String) {
    #[cfg(target_os = "zkvm")]
    openvm::io::println(s);
    #[cfg(not(target_os = "zkvm"))]
    println!("{}", s);
}

pub fn main() {
    let num_bytes: u32 = openvm::io::read();

    println(format!("SHA-256 bench: hashing {} bytes", num_bytes));

    // Feed data to SHA-256 in 4 KB chunks to avoid a single huge allocation.
    const CHUNK_SIZE: usize = 4096;
    let chunk = vec![0xABu8; CHUNK_SIZE];

    let full_chunks = (num_bytes as usize) / CHUNK_SIZE;
    let remainder = (num_bytes as usize) % CHUNK_SIZE;

    let mut hasher = Sha256::new();
    for _ in 0..full_chunks {
        hasher.update(&chunk);
    }
    if remainder > 0 {
        hasher.update(&chunk[..remainder]);
    }
    let output = hasher.finalize();

    println(format!("SHA-256 result: {:x}", output));
}
