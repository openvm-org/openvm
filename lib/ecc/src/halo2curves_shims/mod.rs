use core::ops::Mul;

use axvm_algebra::Field;
use num_bigint::Sign;

mod bls12_381;
mod bn254;

pub trait ExpBytes: Field {
    /// Exponentiates a field element by a value with a sign in big endian byte order
    fn exp_bytes(&self, sign: Sign, bytes_be: &[u8]) -> Self
    where
        for<'a> &'a Self: Mul<&'a Self, Output = Self>,
    {
        if is_one(bytes_be) {
            return Self::ONE;
        }

        let mut x = self.clone();

        if sign == Sign::Minus {
            x = Self::ONE.div_unsafe(&x);
        }

        let mut res = Self::ONE;

        let x_sq = &x * &x;
        let ops = [x.clone(), x_sq.clone(), &x_sq * &x];

        for &b in bytes_be.iter() {
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

impl<F: Field> ExpBytes for F where for<'a> &'a Self: Mul<&'a Self, Output = Self> {}

fn is_one(v: &[u8]) -> bool {
    if v.is_empty() {
        return false;
    }
    // Check all bytes except the last one are 0
    v[..v.len() - 1].iter().all(|&b| b == 0) && v[v.len() - 1] == 1
}
