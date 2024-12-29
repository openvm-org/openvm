use core::mem::transmute;

use halo2curves_axiom::bls12_381::{Fq12, MillerLoopResult};
use num_bigint_dig::BigUint;
use num_traits::Pow;
use openvm_algebra_guest::ExpBytes;

use crate::bls12_381::{BLS12_381_MODULUS, BLS12_381_ORDER};

#[cfg(test)]
mod test_final_exp;
#[cfg(test)]
mod test_line;
#[cfg(test)]
mod test_miller_loop;

// Manual final exponentiation because halo2curves `MillerLoopResult` doesn't have constructor
pub fn final_exp(f: Fq12) -> Fq12 {
    let p = BLS12_381_MODULUS.clone();
    let r = BLS12_381_ORDER.clone();
    let exp: BigUint = (p.pow(12u32) - BigUint::from(1u32)) / r;
    ExpBytes::exp_bytes(&f, true, &exp.to_bytes_be())
}

// Gt(Fq12) is not public
pub fn assert_miller_results_eq(a: MillerLoopResult, b: Fq12) {
    // [jpw] This doesn't work:
    // assert_eq!(a.final_exponentiation(), unsafe { transmute(final_exp(b)) });
    let a = unsafe { transmute::<MillerLoopResult, Fq12>(a) };
    assert_eq!(final_exp(a), final_exp(b));
}
