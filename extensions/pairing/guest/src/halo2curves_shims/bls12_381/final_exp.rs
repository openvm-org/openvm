use halo2curves_axiom::bls12_381::{Fq, Fq12, Fq2};
use num_bigint::BigUint;
use openvm_ecc_guest::{
    algebra::{ExpBytes, Field},
    AffinePoint,
};

use super::{Bls12_381, FINAL_EXP_FACTOR, LAMBDA, POLY_FACTOR};
use crate::pairing::{FinalExp, MultiMillerLoop};

// The paper only describes the implementation for Bn254, so we use the gnark implementation for
// Bls12_381.
#[allow(non_snake_case)]
impl FinalExp for Bls12_381 {
    type Fp = Fq;
    type Fp2 = Fq2;
    type Fp12 = Fq12;

    // Adapted from the gnark implementation:
    // https://github.com/Consensys/gnark/blob/af754dd1c47a92be375930ae1abfbd134c5310d8/std/algebra/emulated/fields_bls12381/e12_pairing.go#L394C1-L395C1
    fn assert_final_exp_is_one(
        f: &Self::Fp12,
        P: &[AffinePoint<Self::Fp>],
        Q: &[AffinePoint<Self::Fp2>],
    ) {
        let (c, s) = Self::final_exp_hint(f);

        // The gnark implementation checks that f * s = c^{q - x} where x is the curve seed.
        // We check an equivalent condition: f * c^x * c^-q * s = 1.
        // This is because we can compute f * c^x by embedding the c^x computation in the miller
        // loop.

        // Since the Bls12_381 curve has a negative seed, the miller loop for Bls12_381 is computed
        // as f_{Miller,x,Q}(P) = conjugate( f_{Miller,-x,Q}(P) * c^{-x} ).
        // We will pass in the conjugate inverse of c into the miller loop so that we compute
        // fc = f_{Miller,x,Q}(P)
        //    = conjugate( f_{Miller,-x,Q}(P) * c'^{-x} )  (where c' is the conjugate inverse of c)
        //    = f_{Miller,x,Q}(P) * c^x
        let c_conj_inv = c.conjugate().invert().unwrap();
        let c_inv = c.invert().unwrap();
        let c_q_inv = c_inv.frobenius_map();
        let fc = Self::multi_miller_loop_embedded_exp(P, Q, Some(c_conj_inv));

        assert_eq!(fc * c_q_inv * s, Fq12::ONE);
    }

    // Adapted from the gnark implementation:
    // https://github.com/Consensys/gnark/blob/af754dd1c47a92be375930ae1abfbd134c5310d8/std/algebra/emulated/fields_bls12381/hints.go#L273
    // returns c (residueWitness) and s (scalingFactor)
    // The Gnark implementation is based on https://eprint.iacr.org/2024/640.pdf
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
        // NOTE[yj]: we can probably remove this first check as an optimization since we initialize
        // order_3rd_power to 0
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
