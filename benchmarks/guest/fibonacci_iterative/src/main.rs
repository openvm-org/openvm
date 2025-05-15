use core::hint::black_box;
use openvm as _;

const N: u64 = 500_000;

pub fn main() {
    let mut a: u64 = 0;
    let mut b: u64 = 1;
    for _ in 0..black_box(N) {
        let c: u64 = a.wrapping_add(b);
        a = b;
        b = c;
    }
    black_box(a);
}
