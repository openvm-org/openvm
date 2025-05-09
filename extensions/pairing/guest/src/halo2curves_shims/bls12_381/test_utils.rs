use core::mem::transmute;

use halo2curves_axiom::bls12_381::{Fq12, MillerLoopResult};
use hex_literal::hex;
use lazy_static::lazy_static;
use num_bigint::BigUint;
use num_traits::Pow;
use openvm_algebra_guest::ExpBytes;

lazy_static! {
    pub static ref BLS12_381_MODULUS: BigUint = BigUint::from_bytes_be(&hex!(
        "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab"
    ));
    pub static ref BLS12_381_ORDER: BigUint = BigUint::from_bytes_be(&hex!(
        "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001"
    ));
}

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
