#![cfg_attr(target_os = "none", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

openvm::entry!(main);

fn fibonacci(n: u64) -> (u64, u64) {
    if n <= 1 {
        return (0, n);
    }
    let mut a: u64 = 0;
    let mut b: u64 = 1;
    for _ in 2..=n {
        let sum = a + b;
        a = b;
        b = sum;
    }
    (a, b)
}

pub fn main() {
    // arbitrary n that results in more than 1 segment
    let n = core::hint::black_box(1 << 5);

    let mut a = 0;
    let mut b = 0;
    // calculate nth fibonacci number n times
    for _ in 0..n {
        (a, b) = fibonacci(n);
    }

    if a == 0 {
        panic!();
    }

    openvm::io::reveal_u64(a, 0);
    openvm::io::reveal_u64(b, 1);
}
