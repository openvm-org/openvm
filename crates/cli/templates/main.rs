// src/main.rs
#![cfg_attr(target_os = "none", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm::io::{read, reveal_u64};

openvm::entry!(main);

pub fn main() {
    let n: u64 = read();
    let mut a: u64 = 0;
    let mut b: u64 = 1;
    for _ in 0..n {
        let c: u64 = a.wrapping_add(b);
        a = b;
        b = c;
    }
    reveal_u64(a, 0);
}
