use core::hint::black_box;
use openvm as _;

const N: u64 = 25;

pub fn main() {
    let n = black_box(N);
    black_box(fibonacci(n));
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
