#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::eq_op)]
use core::array;

use openvm::io::print;
use openvm_bigint_guest::I256;
openvm::entry!(main);

const N: usize = 16;
type Matrix = [[I256; N]; N];

pub fn get_matrix(val: i8) -> Matrix {
    array::from_fn(|_| array::from_fn(|_| I256::from_i8(val)))
}

pub fn mult(a: &Matrix, b: &Matrix) -> Matrix {
    let mut c = get_matrix(0);
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                c[i][j] += &a[i][k] * &b[k][j];
            }
        }
    }
    c
}

pub fn get_identity_matrix() -> Matrix {
    let mut res = get_matrix(0);
    for i in 0..N {
        res[i][i] = I256::from_i8(1);
    }
    res
}

/// Computes base^exp using binary exponentiation.
pub fn matrix_exp(mut base: Matrix, mut exp: I256) -> Matrix {
    let mut result = get_identity_matrix();
    let one = I256::from_i8(1);
    while exp > I256::from_i8(0) {
        if (&exp & &one) == one {
            result = mult(&result, &base);
        }
        base = mult(&base, &base);
        exp >>= &one;
    }
    result
}

pub fn main() {
    let a: Matrix = get_identity_matrix();
    let c = matrix_exp(a, I256::from_i32(1234567));
    if c != get_identity_matrix() {
        print("FAIL: the resulting matrix should have been the identity matrix");
        panic!();
    }

    let one = I256::from_i8(1);
    let neg_one = I256::from_i8(-1);
    let zero = I256::from_i8(0);

    let a: Matrix = get_matrix(-1);
    let c = matrix_exp(a, I256::from_i8(51));
    let two_to_200 = &neg_one << &I256::from_i32(200);

    for i in 0..N {
        for j in 0..N {
            if c[i][j] != two_to_200 {
                print("FAIL: the resulting matrix is incorrect");
                panic!();
            }
        }
    }

    // Shift right tests
    if &two_to_200 >> &I256::from_i32(200) != neg_one {
        print("FAIL: -2^200 >> 200 == -1 test failed");
        panic!();
    }
    if &two_to_200 >> &I256::from_i32(201) != neg_one {
        print("FAIL: -2^200 >> 201 == -1 test failed");
        panic!();
    }

    if &neg_one >> &I256::from_i32(200) != neg_one {
        print("FAIL: -1 >> 200 == -1 test failed");
        panic!();
    }

    // Xor tests
    if &two_to_200 ^ &two_to_200 != zero {
        print("FAIL: -2^200 ^ -2^200 == 0 test failed");
        panic!();
    }

    if &two_to_200 ^ &one != &two_to_200 + &one {
        print("FAIL: -2^200 ^ 1 == -2^200 + 1 test failed");
        panic!();
    }

    // Or tests

    if &one | &one != one {
        print("FAIL: 1 | 1 == 1 test failed");
        panic!();
    }

    if &two_to_200 | &one != &two_to_200 + &one {
        print("FAIL: -2^200 | 1 = -2^200 + 1 test failed");
        panic!();
    }

    // Other tests
    if &zero - &one >= zero {
        print("FAIL: 0 - 1 <= 0 test failed");
        panic!();
    }

    if neg_one >= zero {
        print("FAIL: -1 <= 0 test failed");
        panic!();
    }

    if &zero - &one + &one != zero {
        print("FAIL: 0 - 1 + 1 == 0 test failed (should have wrapped)");
        panic!();
    }

    if &one << &I256::from_i32(256) != one {
        print("FAIL: 1 << 256 == 1 test failed");
        panic!();
    }

    if two_to_200.clone() != two_to_200 {
        print("FAIL: 2^200 clone test failed");
        panic!();
    }

    print("PASS");
}
