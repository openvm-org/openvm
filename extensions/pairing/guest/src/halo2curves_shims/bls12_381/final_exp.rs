use alloc::vec::Vec;
use core::convert::TryInto;
use halo2curves_axiom::bls12_381::{Fq, Fq12, Fq2};
use lazy_static::lazy_static;
use num_bigint::BigUint;
use num_traits::Zero;
use openvm_ecc_guest::{
    algebra::{ExpBytes, Field},
    AffinePoint,
};

use super::{Bls12_381, FINAL_EXP_FACTOR, LAMBDA, POLY_FACTOR};
use crate::pairing::{FinalExp, MultiMillerLoop};

lazy_static! {
    static ref FINAL_EXP_FACTOR_TIMES_27: BigUint = FINAL_EXP_FACTOR.clone() * BigUint::from(27u32);
    static ref FINAL_EXP_FACTOR_TIMES_27_BE: Vec<u8> = FINAL_EXP_FACTOR_TIMES_27.to_bytes_be();
    static ref PTH_ROOT_INV_EXP_BE: Vec<u8> = {
        let exp_inv = FINAL_EXP_FACTOR_TIMES_27
            .modinv(&POLY_FACTOR.clone())
            .unwrap();
        let exp = neg_mod(&exp_inv, &POLY_FACTOR);
        exp.to_bytes_be()
    };
    static ref POLY_FACTOR_TIMES_FINAL_EXP: BigUint =
        POLY_FACTOR.clone() * FINAL_EXP_FACTOR.clone();
    static ref POLY_FACTOR_TIMES_FINAL_EXP_BE: Vec<u8> = POLY_FACTOR_TIMES_FINAL_EXP.to_bytes_be();
    static ref THREE: BigUint = BigUint::from(3u32);
    static ref THREE_BE: Vec<u8> = THREE.to_bytes_be();
    static ref ROOT27_EXPONENT_BYTES: [Vec<u8>; 3] = {
        let exponent = POLY_FACTOR_TIMES_FINAL_EXP.clone();
        let moduli = [THREE.clone(), THREE.clone().pow(2), THREE.clone().pow(3)];
        let mut exps = Vec::with_capacity(3);
        for modulus in moduli.iter() {
            let inv = exponent.modinv(modulus).unwrap();
            let exp = neg_mod(&inv, modulus);
            exps.push(exp.to_bytes_be());
        }
        exps.try_into().expect("three exponents")
    };
    static ref LAMBDA_INV_MOD_FINAL_EXP_BE: Vec<u8> = {
        let exponent = LAMBDA.clone().modinv(&FINAL_EXP_FACTOR.clone()).unwrap();
        exponent.to_bytes_be()
    };
}

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
        final_exp_witness(f)
    }
}

fn final_exp_witness(f: &Fq12) -> (Fq12, Fq12) {
    // 1. Compute the p-th root inverse factor.
    let root = f.exp_bytes(true, FINAL_EXP_FACTOR_TIMES_27_BE.as_slice());
    let root_pth_inverse = if root == Fq12::ONE {
        Fq12::ONE
    } else {
        root.exp_bytes(true, PTH_ROOT_INV_EXP_BE.as_slice())
    };

    // 2.1 Determine the order of the 3rd primitive root.
    let mut order_3rd_power = 0u32;
    let mut root_order = f.exp_bytes(true, POLY_FACTOR_TIMES_FINAL_EXP_BE.as_slice());
    if root_order == Fq12::ONE {
        order_3rd_power = 0;
    }
    root_order = root_order.exp_bytes(true, THREE_BE.as_slice());
    if root_order == Fq12::ONE {
        order_3rd_power = 1;
    }
    root_order = root_order.exp_bytes(true, THREE_BE.as_slice());
    if root_order == Fq12::ONE {
        order_3rd_power = 2;
    }
    root_order = root_order.exp_bytes(true, THREE_BE.as_slice());
    if root_order == Fq12::ONE {
        order_3rd_power = 3;
    }

    // 2.2 Compute the 27th root inverse.
    let root_27th_inverse = if order_3rd_power == 0 {
        Fq12::ONE
    } else {
        let exponent_bytes = &ROOT27_EXPONENT_BYTES[(order_3rd_power - 1) as usize];
        let root = f.exp_bytes(true, POLY_FACTOR_TIMES_FINAL_EXP_BE.as_slice());
        root.exp_bytes(true, exponent_bytes.as_slice())
    };

    // 2.3 Shift the Miller loop output by the scaling factor so that the result
    // has order FINAL_EXP_FACTOR.
    let scaling_factor = root_pth_inverse * root_27th_inverse;
    let shifted = f * scaling_factor;

    // 3. Compute the residue witness with exponent lambda^{-1} mod FINAL_EXP_FACTOR.
    let residue_witness = shifted.exp_bytes(true, LAMBDA_INV_MOD_FINAL_EXP_BE.as_slice());

    (residue_witness, scaling_factor)
}

fn neg_mod(value: &BigUint, modulus: &BigUint) -> BigUint {
    let value_mod = value % modulus;
    if value_mod.is_zero() {
        BigUint::from(0u32)
    } else {
        modulus.clone() - value_mod
    }
}
