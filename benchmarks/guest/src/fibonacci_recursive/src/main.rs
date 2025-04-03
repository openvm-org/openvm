extern crate openvm;

use core::hint::black_box;

const N: u64 = 25;

pub fn main() {
    black_box(fibonacci(black_box(N)));
}

fn fibonacci(n: u64) -> u64 {
    if n == 0 {
        0
    } else if n == 1 {
        1
    } else {
        let a = fibonacci(n - 2);
        let b = fibonacci(n - 1);
        a.wrapping_add(b)
    }
}
