#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(target_os = "zkvm", no_std)]

// src/main.rs
use openvm::io::{read, reveal};

openvm::entry!(main);

fn main() {
    let n: u64 = read();
    let mut a: u64 = 0;
    let mut b: u64 = 1;
    for _ in 0..n {
        let c: u64 = a.wrapping_add(b);
        a = b;
        b = c;
    }
    reveal(a as u32, 0);
    reveal((a >> 32) as u32, 1);
}
