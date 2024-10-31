#![no_main]
#![no_std]

use core::hint::black_box;

axvm::entry!(main);

pub fn main() {
    let n: u64 = black_box(100000);
    let mut a: u64 = 0;
    let mut b: u64 = 1;
    for _ in 0..n {
        let c: u64 = a.wrapping_add(b);
        a = b;
        b = c;
    }
    let _ = black_box(a);
}
