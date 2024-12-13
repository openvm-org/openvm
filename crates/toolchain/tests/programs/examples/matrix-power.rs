#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

openvm::entry!(main);
use core::array;

use openvm::io::print;
use openvm_bigint_guest::U256;

const N: usize = 16;
type Matrix = [[U256; N]; N];

pub fn get_matrix(val: u8) -> Matrix {
    array::from_fn(|_| array::from_fn(|_| U256::from_u8(val)))
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
        res[i][i] = U256::from_u8(1);
    }
    res
}

/// Computes base^exp using binary exponentiation.
pub fn bin_exp(mut base: Matrix, mut exp: U256) -> Matrix {
    let mut result = get_identity_matrix();
    let one = U256::from_u8(1);
    while exp > U256::from_u8(0) {
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
    let c = bin_exp(a, U256::from_u32(1234567));
    if c != get_identity_matrix() {
        print("FAIL: the resulting matrix should have been the identity matrix");
        panic!();
    }

    let one = U256::from_u8(1);
    let zero = U256::from_u8(0);

    let a: Matrix = get_matrix(1);
    let c = bin_exp(a, U256::from_u8(51));
    let two_to_200 = &one << &U256::from_u8(200);

    for i in 0..N {
        for j in 0..N {
            if c[i][j] != two_to_200 {
                print("FAIL: the resulting matrix is incorect");
                panic!();
            }
        }
    }

    // Shift right tests
    if &two_to_200 >> &U256::from_u8(200) != one {
        print("FAIL: 2^200 >> 200 == 1 test failed");
        panic!();
    }
    if &two_to_200 >> &U256::from_u8(201) != zero {
        print("FAIL: 2^200 >> 201 == 0 test failed");
        panic!();
    }

    // Xor tests
    if &two_to_200 ^ &two_to_200 != zero {
        print("FAIL: 2^200 ^ 2^200 == 0 test failed");
        panic!();
    }

    if &two_to_200 ^ &one != &two_to_200 + &one {
        print("FAIL: 2^200 ^ 1 == 2^200 + 1 test failed");
        panic!();
    }

    // Or tests
    if &one | &one != one {
        print("FAIL: 1 | 1 == 1 test failed");
        panic!();
    }

    if &two_to_200 | &one != &two_to_200 + &one {
        print("FAIL: 2^200 | 1 = 2^200 + 1 test failed");
        panic!();
    }

    // Other tests
    if &zero - &one <= zero {
        print("FAIL: 0 - 1 > 0 test failed (should have wrapped)");
        panic!();
    }

    if &zero - &one + &one != zero {
        print("FAIL: 0 - 1 + 1 == 0 test failed (should have wrapped)");
        panic!();
    }

    if &one << &U256::from_u32(256) != one {
        print("FAIL: 1 << 256 == 1 test failed");
        panic!();
    }

    if &one << &U256::from_u32(261) != U256::from_u8(32) {
        print("FAIL: 1 << 261 == 32 test failed");
        panic!();
    }

    if two_to_200.clone() != two_to_200 {
        print("FAIL: 2^200 clone test failed");
        panic!();
    }

    print("PASS");
}