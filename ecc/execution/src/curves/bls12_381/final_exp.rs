use halo2curves_axiom::bls12_381::{Fq, Fq12, Fq2};
use num::BigInt;

use super::{Bls12_381, FINAL_EXP_FACTOR, LAMBDA, POLY_FACTOR, T};
use crate::common::{EcPoint, ExpBigInt, FinalExp};

#[allow(non_snake_case)]
impl FinalExp<Fq, Fq2, Fq12> for Bls12_381 {
    fn assert_final_exp_is_one(&self, f: Fq12, _P: &[EcPoint<Fq>], _Q: &[EcPoint<Fq2>]) {
        let (c, s) = self.final_exp_hint(f);

        let c_q = c.frobenius_map();
        let c_exp = c.exp_bigint(T.clone());
        let c_mul = c_q * c_exp;

        assert_eq!(f * s, c_mul);
    }

    // Adapted from the gnark implementation:
    // https://github.com/Consensys/gnark/blob/af754dd1c47a92be375930ae1abfbd134c5310d8/std/algebra/emulated/fields_bls12381/hints.go#L273
    fn final_exp_hint(&self, f: Fq12) -> (Fq12, Fq12) {
        // 1. get p-th root inverse
        let mut exp = FINAL_EXP_FACTOR.clone() * BigInt::from(27);
        let mut root = f.exp_bigint(exp.clone());
        let root_pth_inv: Fq12;
        if root == Fq12::one() {
            root_pth_inv = Fq12::one();
        } else {
            let exp_inv = exp.modinv(&POLY_FACTOR.clone()).unwrap();
            exp = -exp_inv % POLY_FACTOR.clone();
            root_pth_inv = root.exp_bigint(exp);
        }

        // 2.1. get order of 3rd primitive root
        let three = BigInt::from(3);
        let mut order_3rd_power: u32 = 0;
        exp = POLY_FACTOR.clone() * FINAL_EXP_FACTOR.clone();

        root = f.exp_bigint(exp.clone());
        if root == Fq12::one() {
            order_3rd_power = 0;
        }
        root = root.exp_bigint(three.clone());
        if root == Fq12::one() {
            order_3rd_power = 1;
        }
        root = root.exp_bigint(three.clone());
        if root == Fq12::one() {
            order_3rd_power = 2;
        }
        root = root.exp_bigint(three.clone());
        if root == Fq12::one() {
            order_3rd_power = 3;
        }

        // 2.2. get 27th root inverse
        let root_27th_inv: Fq12;
        if order_3rd_power == 0 {
            root_27th_inv = Fq12::one();
        } else {
            let order_3rd = three.pow(order_3rd_power);
            exp = POLY_FACTOR.clone() * FINAL_EXP_FACTOR.clone();
            root = f.exp_bigint(exp.clone());
            let exp_inv = exp.modinv(&order_3rd).unwrap();
            exp = -exp_inv % order_3rd;
            root_27th_inv = root.exp_bigint(exp);
        }

        // 2.3. shift the Miller loop result so that millerLoop * scalingFactor
        // is of order finalExpFactor
        let s = root_pth_inv * root_27th_inv;
        let f = f * s;

        // 3. get the witness residue
        // lambda = q - u, the optimal exponent
        exp = LAMBDA.clone().modinv(&FINAL_EXP_FACTOR.clone()).unwrap();
        let c = f.exp_bigint(exp);

        (c, s)
    }
}
