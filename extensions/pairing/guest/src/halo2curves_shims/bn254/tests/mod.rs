use core::mem::transmute;

use halo2curves_axiom::{
    bn256::{Fq12, Gt},
    pairing::MillerLoopResult,
};
use num_bigint::BigUint;
use num_traits::Pow;
use openvm_algebra_guest::ExpBytes;

use crate::bn254::{BN254_MODULUS, BN254_ORDER};

#[cfg(test)]
mod test_final_exp;
#[cfg(test)]
mod test_line;
#[cfg(test)]
mod test_miller_loop;

// Manual final exponentiation because halo2curves `MillerLoopResult` doesn't have constructor
pub fn final_exp(f: Fq12) -> Fq12 {
    let p = BN254_MODULUS.clone();
    let r = BN254_ORDER.clone();
    let exp: BigUint = (p.pow(12u32) - BigUint::from(1u32)) / r;
    ExpBytes::exp_bytes(&f, true, &exp.to_bytes_be())
}

// Gt(Fq12) is not public
pub fn assert_miller_results_eq(a: Gt, b: Fq12) {
    let a = a.final_exponentiation();
    let b = final_exp(b);
    assert_eq!(unsafe { transmute::<Gt, Fq12>(a) }, b);
}
