use core::mem::transmute;

use halo2curves_axiom::{
    bn256::{Fq12, Gt},
    pairing::MillerLoopResult,
};
use hex_literal::hex;
use lazy_static::lazy_static;
use num_bigint::BigUint;
use num_traits::Pow;
use openvm_algebra_guest::ExpBytes;

lazy_static! {
    pub static ref BN254_MODULUS: BigUint = BigUint::from_bytes_be(&hex!(
        "30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47"
    ));
    pub static ref BN254_ORDER: BigUint = BigUint::from_bytes_be(&hex!(
        "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001"
    ));
}

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
