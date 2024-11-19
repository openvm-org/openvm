use core::ops::Mul;

use axvm_algebra::Field;
use num_bigint::{BigUint, Sign};

mod bls12_381;
mod bn254;

pub trait ExpBigInt: Field {
    /// Exponentiates a field element by a BigUint with sign
    fn exp_bigint(&self, sign: Sign, k: BigUint) -> Self
    where
        for<'a> &'a Self: Mul<&'a Self, Output = Self>,
    {
        if k == BigUint::from(0u32) {
            return Self::ONE;
        }

        let mut x = self.clone();

        if sign == Sign::Minus {
            x = Self::ONE.div_unsafe(&x);
        }

        let mut res = Self::ONE;

        let x_sq = &x * &x;
        let ops = [x.clone(), x_sq.clone(), &x_sq * &x];

        let bytes = k.to_bytes_be();
        for &b in bytes.iter() {
            let mut mask = 0xc0;
            for j in 0..4 {
                res = &res * &res * &res * &res;
                let c = (b & mask) >> (6 - 2 * j);
                if c != 0 {
                    res *= &ops[(c - 1) as usize];
                }
                mask >>= 2;
            }
        }
        res
    }
}

impl<F: Field> ExpBigInt for F where for<'a> &'a Self: Mul<&'a Self, Output = Self> {}
