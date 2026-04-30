#![cfg_attr(any(target_os = "none", target_os = "openvm"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

openvm::entry!(main);

use core::hint::black_box;
use openvm::io::reveal_u64;

const N: u64 = 900_000;

pub fn main() {
    let mut a: u64 = 0;
    let mut b: u64 = 1;
    for _ in 0..black_box(N) {
        let c: u64 = a.wrapping_add(b);
        a = b;
        b = c;
    }
    reveal_u64(a, 0);
}
