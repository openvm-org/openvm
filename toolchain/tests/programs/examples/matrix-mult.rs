#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

axvm::entry!(main);
use axvm::intrinsics::*;

const N: usize = 4;
type Matrix = [[U256; N]; N];

pub fn mult(a: &Matrix, b: &Matrix) -> Matrix {
    let mut c = [[U256::from_u8(0); N]; N];
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                c[i][j] += &a[i][k] * &b[k][j];
            }
        }
    }
    c
}

pub fn main() {
    let a: Matrix = [[U256::from_u8(28); N]; N];
    let b: Matrix = [[U256::from_u8(93); N]; N];
    let c = mult(&mult(&a, &b), &mult(&a, &b));
    
}
