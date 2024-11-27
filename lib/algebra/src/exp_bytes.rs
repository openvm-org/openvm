use core::ops::Mul;

use crate::Field;

pub trait ExpBytes: Field {
    /// Exponentiates a field element by a value with a sign in big endian byte order
    fn exp_bytes(&self, is_positive: bool, bytes_be: &[u8]) -> Self
    where
        for<'a> &'a Self: Mul<&'a Self, Output = Self>,
    {
        let mut x = self.clone();

        if !is_positive {
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
