use alloc::vec::Vec;

use ff::Field;

pub trait ExpBigInt<F: Field>: Field {
    /// Exponentiates a field element by a BigInt
    fn exp_bigint(&self, is_positive: bool, k: Vec<u8>) -> Self {
        self.clone()
        // if Vec::len(k) == 0 || Vec::len(k) == 1 && k[0] == 0 {
        //     return Self::ONE;
        // }

        // let mut e = k.clone();
        // let mut x = *self;

        // if k < BigInt::from(0) {
        //     x = x.invert().unwrap();
        //     e = -k;
        // }

        // let mut res = Self::ONE;

        // let x_sq = x.square();
        // let ops = [x, x_sq, x_sq * x];

        // let bytes = e.to_bytes_be();
        // for &b in bytes.1.iter() {
        //     let mut mask = 0xc0;
        //     for j in 0..4 {
        //         res = res.square().square();
        //         let c = (b & mask) >> (6 - 2 * j);
        //         if c != 0 {
        //             res *= &ops[(c - 1) as usize];
        //         }
        //         mask >>= 2;
        //     }
        // }

        // res
    }
}
