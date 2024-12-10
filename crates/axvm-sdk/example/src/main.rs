#![no_main]
#![no_std]

axvm::entry!(main);

pub fn main() {
    let n = core::hint::black_box(1 << 10);
    let mut a: u32 = 0;
    let mut b: u32 = 1;
    for _ in 1..n {
        let sum = a + b;
        a = b;
        b = sum;
    }
    if a == 0 {
        panic!();
    }
}