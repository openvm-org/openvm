use alloc::vec::Vec;
use core::mem::transmute;

use halo2curves_axiom::{
    bn256::{Fq, Fq12, Fq2, G1Affine, G2Affine, Gt},
    pairing::MillerLoopResult,
};
use hex_literal::hex;
use itertools::izip;
use lazy_static::lazy_static;
use num_bigint::BigUint;
use num_traits::Pow;
use openvm_algebra_guest::ExpBytes;
use openvm_ecc_guest::AffinePoint;
use rand::{rngs::StdRng, SeedableRng};

#[cfg(test)]
mod test_final_exp;
#[cfg(test)]
mod test_line;
#[cfg(test)]
mod test_miller_loop;

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

#[allow(non_snake_case)]
#[allow(clippy::type_complexity)]
pub fn generate_test_points_bn254(
    rand_seeds: &[u64],
) -> (
    Vec<G1Affine>,
    Vec<G2Affine>,
    Vec<AffinePoint<Fq>>,
    Vec<AffinePoint<Fq2>>,
) {
    let (P_vec, Q_vec) = rand_seeds
        .iter()
        .map(|seed| {
            let mut rng0 = StdRng::seed_from_u64(*seed);
            let p = G1Affine::random(&mut rng0);
            let mut rng1 = StdRng::seed_from_u64(*seed * 2);
            let q = G2Affine::random(&mut rng1);
            (p, q)
        })
        .unzip::<_, _, Vec<_>, Vec<_>>();
    let (P_ecpoints, Q_ecpoints) = izip!(P_vec.clone(), Q_vec.clone())
        .map(|(P, Q)| {
            (
                AffinePoint { x: P.x, y: P.y },
                AffinePoint { x: Q.x, y: Q.y },
            )
        })
        .unzip::<_, _, Vec<_>, Vec<_>>();
    (P_vec, Q_vec, P_ecpoints, Q_ecpoints)
}
