use core::ops::Mul;

use axvm_algebra::Field;
use num_bigint::BigInt;

mod bls12_381;
mod bn254;

pub trait ExpBigInt: Field {
    /// Exponentiates a field element by a BigInt
    fn exp_bigint(&self, k: BigInt) -> Self
    where
        for<'a> &'a Self: Mul<&'a Self, Output = Self>,
    {
        if k == BigInt::from(0) {
            return Self::ONE;
        }

        let mut e = k.clone();
        let mut x = self.clone();

        if k < BigInt::from(0) {
            x = Self::ONE.div_unsafe(&x);
            e = -k;
        }

        let mut res = Self::ONE;

        let x_sq = &x * &x;
        let ops = [x.clone(), x_sq.clone(), &x_sq * &x];

        let bytes = e.to_bytes_be();
        for &b in bytes.1.iter() {
            let mut mask = 0xc0;
            for j in 0..4 {
                // res = res.square().square()
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
