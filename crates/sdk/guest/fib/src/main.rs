#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(target_os = "zkvm", no_std)]

openvm::entry!(main);

fn fibonacci(n: u32) -> (u32, u32) {
    if n <= 1 {
        return (0, n);
    }
    let mut a: u32 = 0;
    let mut b: u32 = 1;
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

    openvm::io::reveal_u32(a, 0);
    openvm::io::reveal_u32(b, 1);
}
