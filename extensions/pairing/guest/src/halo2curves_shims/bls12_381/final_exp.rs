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
        let f_final_exp = f.exp_bytes(true, &FINAL_EXP_FACTOR.to_bytes_be());
        let root = f_final_exp.exp_bytes(true, &BigUint::from(27u32).to_bytes_be());

        // 1. get p-th root inverse
        let root_pth_inv = if root == Fq12::ONE {
            Fq12::ONE
        } else {
            let exp_inv = (FINAL_EXP_FACTOR.clone() * BigUint::from(27u32))
                .modinv(&POLY_FACTOR.clone())
                .unwrap();
            root.exp_bytes(false, &exp_inv.to_bytes_be())
        };

        let root = f_final_exp.exp_bytes(true, &POLY_FACTOR.to_bytes_be());
        // 2.2. get 27th root inverse
        let root_27th_inv =
            if root.exp_bytes(true, &BigUint::from(27u32).to_bytes_be()) == Fq12::ONE {
                root.exp_bytes(false, &BigUint::from(10u32).to_bytes_be())
            } else {
                Fq12::ONE
            };

        // 2.3. shift the Miller loop result so that millerLoop * scalingFactor
        // is of order finalExpFactor
        let s = root_pth_inv * root_27th_inv;
        let f = f * s;

        // 3. get the witness residue
        // lambda = q - u, the optimal exponent
        let exp = LAMBDA.clone().modinv(&FINAL_EXP_FACTOR.clone()).unwrap();
        let c = f.exp_bytes(true, &exp.to_bytes_be());

        (c, s)
    }
}
