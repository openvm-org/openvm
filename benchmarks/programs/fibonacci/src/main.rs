#![no_main]
#![no_std]

use core::hint::black_box;

axvm::entry!(main);

pub fn main() {
    let n = 1 << 20;
    let mut a: u32 = black_box(0);
    let mut b: u32 = black_box(1);
    for _ in 1..n {
        let sum = a.wrapping_add(b);
        a = b;
        b = sum;
    }
    let _ = black_box(b);
}
