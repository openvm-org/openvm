use core::hint::black_box;
use openvm::io::reveal_u32;

const N: u32 = 900_000;

pub fn main() {
    let mut a: u32 = 0;
    let mut b: u32 = 1;
    for _ in 0..black_box(N) {
        let c: u32 = a.wrapping_add(b);
        a = b;
        b = c;
    }
    reveal_u32(a, 0);
}
