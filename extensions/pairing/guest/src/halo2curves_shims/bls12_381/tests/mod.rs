use alloc::vec::Vec;
use core::mem::transmute;

use halo2curves_axiom::bls12_381::{Fq, Fq12, Fq2, G1Affine, G2Affine, MillerLoopResult};
use itertools::izip;
use num_bigint::BigUint;
use num_traits::Pow;
use openvm_algebra_guest::ExpBytes;
use openvm_ecc_guest::AffinePoint;
use rand::{rngs::StdRng, SeedableRng};

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

#[allow(non_snake_case)]
#[allow(clippy::type_complexity)]
pub fn generate_test_points_bls12_381(
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
