#![allow(clippy::needless_range_loop)]
use core::array;

use alloy_primitives::I256;

openvm::entry!(main);

const N: usize = 16;
type Matrix = [[I256; N]; N];

pub fn get_matrix(val: i32) -> Matrix {
    array::from_fn(|_| array::from_fn(|_| I256::try_from(val).unwrap()))
}

pub fn mult(a: &Matrix, b: &Matrix) -> Matrix {
    let mut c = get_matrix(0);
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

pub fn get_identity_matrix() -> Matrix {
    let mut res = get_matrix(0);
    for i in 0..N {
        res[i][i] = I256::try_from(1i32).unwrap();
    }
    res
}

pub fn main() {
    let a: Matrix = get_identity_matrix();
    let b: Matrix = get_matrix(-28);
    let c: Matrix = mult(&a, &b);
    assert_eq!(c, b);
}
