use core::hint::black_box;
use openvm::io::reveal_u32;

const N: u32 = 27;

pub fn main() {
    let n = black_box(N);
    let result = fibonacci(n);
    reveal_u32(result, 0);
}

fn fibonacci(n: u32) -> u32 {
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
