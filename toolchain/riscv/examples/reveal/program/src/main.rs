#![no_main]
#![no_std]
use axvm::io::reveal;

axvm::entry!(main);

pub fn main() {
    let x: u32 = core::hint::black_box(123);
    let y: u32 = core::hint::black_box(456);
    reveal(x, 0);
    reveal(y, 2);
}
