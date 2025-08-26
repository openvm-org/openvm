extern crate alloc;

use std::hint::black_box;

use openvm_algebra_guest::{DivUnsafe, IntMod};
use openvm_pairing::bn254::Bn254Fp;

openvm::init!();

fn fermat() {
    let mut pow = Bn254Fp::MODULUS;
    pow[0] -= 2;

    let mut a = Bn254Fp::from_u32(1234);
    let mut res = Bn254Fp::from_u32(1);
    let inv = res.clone().div_unsafe(&a);

    for pow_bit in pow {
        for j in 0..8 {
            if pow_bit & (1 << j) != 0 {
                res *= &a;
            }
            a *= a.clone();
        }
    }

    // https://en.wikipedia.org/wiki/Fermat%27s_little_theorem
    assert_eq!(res, inv);
}

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
    let (a, b) = fibonacci(n);

    fermat();

    if a == 0 {
        panic!();
    }

    openvm::io::reveal_u32(a, 0);
    openvm::io::reveal_u32(b, 1);
}
