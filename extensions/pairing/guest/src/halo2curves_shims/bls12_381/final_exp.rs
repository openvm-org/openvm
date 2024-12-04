use axvm_ecc_guest::{
    algebra::{ExpBytes, Field},
    AffinePoint,
};
use halo2curves_axiom::bls12_381::{Fq, Fq12, Fq2};
use num_bigint::BigUint;

use super::{Bls12_381, FINAL_EXP_FACTOR, LAMBDA, POLY_FACTOR};
use crate::pairing::{FinalExp, MultiMillerLoop};

#[allow(non_snake_case)]
impl FinalExp for Bls12_381 {
    type Fp = Fq;
    type Fp2 = Fq2;
    type Fp12 = Fq12;

    fn assert_final_exp_is_one(
        f: &Self::Fp12,
        P: &[AffinePoint<Self::Fp>],
        Q: &[AffinePoint<Self::Fp2>],
    ) {
        let (c, s) = Self::final_exp_hint(f);

        // f * s = c^{q - x}
        // f * s = c^q * c^-x
        // f * c^x * c^-q * s = 1,
        //   where fc = f * c'^x (embedded Miller loop with c conjugate inverse),
        //   and the curve seed x = -0xd201000000010000
        //   the miller loop computation includes a conjugation at the end because the value of the
        //   seed is negative, so we need to conjugate the miller loop input c as c'. We then substitute
        //   y = -x to get c^-y and finally compute c'^-y as input to the miller loop:
        // f * c'^-y * c^-q * s = 1
        let c_inv = c.invert().unwrap();
        let c_conj_inv = c.conjugate().invert().unwrap();
        let c_q_inv = c_inv.frobenius_map();

        // fc = f_{Miller,x,Q}(P) * c^{x}
        // where
        //   fc = conjugate( f_{Miller,-x,Q}(P) * c'^{-x} ), with c' denoting the conjugate of c
        let fc = Self::multi_miller_loop_embedded_exp(P, Q, Some(c_conj_inv));

        assert_eq!(fc * c_q_inv * s, Fq12::ONE);
    }

    // Adapted from the gnark implementation:
    // https://github.com/Consensys/gnark/blob/af754dd1c47a92be375930ae1abfbd134c5310d8/std/algebra/emulated/fields_bls12381/hints.go#L273
    // returns c (residueWitness) and s (scalingFactor)
    fn final_exp_hint(f: &Self::Fp12) -> (Self::Fp12, Self::Fp12) {
        // 1. get p-th root inverse
        let mut exp = FINAL_EXP_FACTOR.clone() * BigUint::from(27u32);
        let mut root = f.exp_bytes(true, &exp.to_bytes_be());
        let root_pth_inv: Fq12;
        if root == Fq12::ONE {
            root_pth_inv = Fq12::ONE;
        } else {
            let exp_inv = exp.modinv(&POLY_FACTOR.clone()).unwrap();
            exp = exp_inv % POLY_FACTOR.clone();
            root_pth_inv = root.exp_bytes(false, &exp.to_bytes_be());
        }

        // 2.1. get order of 3rd primitive root
        let three = BigUint::from(3u32);
        let mut order_3rd_power: u32 = 0;
        exp = POLY_FACTOR.clone() * FINAL_EXP_FACTOR.clone();

        root = f.exp_bytes(true, &exp.to_bytes_be());
        let three_be = three.to_bytes_be();
        // NOTE[yj]: we can probably remove this first check as an optimization since we initizlize order_3rd_power to 0
        if root == Fq12::ONE {
            order_3rd_power = 0;
        }
        root = root.exp_bytes(true, &three_be);
        if root == Fq12::ONE {
            order_3rd_power = 1;
        }
        root = root.exp_bytes(true, &three_be);
        if root == Fq12::ONE {
            order_3rd_power = 2;
        }
        root = root.exp_bytes(true, &three_be);
        if root == Fq12::ONE {
            order_3rd_power = 3;
        }

        // 2.2. get 27th root inverse
        let root_27th_inv: Fq12;
        if order_3rd_power == 0 {
            root_27th_inv = Fq12::ONE;
        } else {
            let order_3rd = three.pow(order_3rd_power);
            exp = POLY_FACTOR.clone() * FINAL_EXP_FACTOR.clone();
            root = f.exp_bytes(true, &exp.to_bytes_be());
            let exp_inv = exp.modinv(&order_3rd).unwrap();
            exp = exp_inv % order_3rd;
            root_27th_inv = root.exp_bytes(false, &exp.to_bytes_be());
        }

        // 2.3. shift the Miller loop result so that millerLoop * scalingFactor
        // is of order finalExpFactor
        let s = root_pth_inv * root_27th_inv;
        let f = f * s;

        // 3. get the witness residue
        // lambda = q - u, the optimal exponent
        exp = LAMBDA.clone().modinv(&FINAL_EXP_FACTOR.clone()).unwrap();
        let c = f.exp_bytes(true, &exp.to_bytes_be());

        (c, s)
    }
}
