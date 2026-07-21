use alloc::{vec, vec::Vec};

use halo2curves_axiom::{
    bls12_381::{Fq, Fq12, Fq2, Fr, G1Affine, G2Affine},
    ff::Field,
};
use itertools::izip;
use num_bigint::BigUint;
use num_traits::Num;
use openvm_ecc_guest::{algebra::ExpBytes, AffinePoint};
use rand08::{rngs::StdRng, SeedableRng};

use crate::{
    halo2curves_shims::bls12_381::{final_exp::final_exp_hint_basic, Bls12_381, SEED_NEG},
    pairing::{FinalExp, MultiMillerLoop},
};

#[test]
#[allow(non_snake_case)]
fn test_bls12_381_final_exp_hint() {
    let (_P_vec, _Q_vec, P_ecpoints, Q_ecpoints) =
        generate_test_points_bls12_381(&[Fr::from(3), Fr::from(6)], &[Fr::from(8), Fr::from(4)]);

    let f = Bls12_381::multi_miller_loop(&P_ecpoints, &Q_ecpoints);
    let (c, s) = Bls12_381::final_exp_hint(&f);

    let q = BigUint::from_str_radix(
        "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab",
        16,
    ).unwrap();
    let c_qt = c.exp_bytes(true, &q.to_bytes_be()) * c.exp_bytes(true, &SEED_NEG.to_bytes_be());

    assert_eq!(f * s, c_qt);
}

#[cfg(feature = "blst")]
#[test]
fn test_blst_final_exp_hint_matches_basic() {
    let mut inputs = vec![Fq12::ZERO, Fq12::ONE];
    let mut rng = StdRng::seed_from_u64(0x41b5_1238);
    inputs.extend((0..4).map(|_| Fq12::random(&mut rng)));

    for f in inputs {
        let expected = final_exp_hint_basic(&f);
        let actual = Bls12_381::final_exp_hint(&f);
        assert_eq!(actual, expected);
    }
}

#[test]
#[allow(non_snake_case)]
fn test_bls12_381_assert_final_exp_is_one_scalar_ones() {
    assert_final_exp_one(&[Fr::from(1), Fr::from(2)], &[Fr::from(2), Fr::from(1)]);
}

#[test]
#[allow(non_snake_case)]
fn test_bls12_381_assert_final_exp_is_one_scalar_other() {
    assert_final_exp_one(&[Fr::from(5), Fr::from(2)], &[Fr::from(10), Fr::from(25)]);
}

#[allow(non_snake_case)]
fn assert_final_exp_one(a: &[Fr; 2], b: &[Fr; 2]) {
    let (_P_vec, _Q_vec, P_ecpoints, Q_ecpoints) = generate_test_points_bls12_381(a, b);
    let f = Bls12_381::multi_miller_loop(&P_ecpoints, &Q_ecpoints);
    Bls12_381::assert_final_exp_is_one(&f, &P_ecpoints, &Q_ecpoints);
}

#[allow(non_snake_case)]
#[allow(clippy::type_complexity)]
fn generate_test_points_bls12_381(
    a: &[Fr; 2],
    b: &[Fr; 2],
) -> (
    Vec<G1Affine>,
    Vec<G2Affine>,
    Vec<AffinePoint<Fq>>,
    Vec<AffinePoint<Fq2>>,
) {
    let mut P_vec = Vec::new();
    let mut Q_vec = Vec::new();
    for i in 0..2 {
        let p = G1Affine::generator() * a[i];
        let mut p = G1Affine::from(p);
        if i % 2 == 1 {
            p.y = -p.y;
        }
        let q = G2Affine::generator() * b[i];
        let q = G2Affine::from(q);
        P_vec.push(p);
        Q_vec.push(q);
    }
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
