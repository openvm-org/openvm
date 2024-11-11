use axvm_ecc::{
    curve::bls12381::{Fq, Fq2, G1Affine, G2Affine},
    field::ExpBigInt,
    pairing::{FinalExp, MultiMillerLoop},
    point::{AffineCoords, AffinePoint},
};
use halo2curves_axiom::bls12_381::Fr;
use itertools::izip;
use num::{BigInt, Num};

use crate::curves::bls12_381::{Bls12_381, BLS12_381_PBE_LEN, SEED_NEG};

#[test]
#[allow(non_snake_case)]
fn test_bls12_381_final_exp_hint() {
    let (_P_vec, _Q_vec, P_ecpoints, Q_ecpoints) =
        // generate_test_points_bls12_381(&[Fr::from(3), Fr::from(6)], &[Fr::from(8), Fr::from(4)]);
        generate_test_points_bls12_381(&[Fr::from(1), Fr::from(1)], &[Fr::from(1), Fr::from(1)]);

    let bls12_381 = Bls12_381;
    let f = bls12_381.multi_miller_loop::<BLS12_381_PBE_LEN>(&P_ecpoints, &Q_ecpoints);
    let (c, s) = bls12_381.final_exp_hint(f);

    let q = BigInt::from_str_radix(
        "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab",
        16,
    ).unwrap();
    let c_qt = c.exp_bigint(q) * c.exp_bigint(SEED_NEG.clone());

    assert_eq!(f * s, c_qt);
}

#[test]
#[allow(non_snake_case)]
fn test_bls12_381_assert_final_exp_is_one_scalar_ones() {
    assert_final_exp_one(&[Fr::from(1), Fr::from(1)], &[Fr::from(1), Fr::from(1)]);
}

#[test]
#[allow(non_snake_case)]
fn test_bls12_381_assert_final_exp_is_one_scalar_other() {
    assert_final_exp_one(&[Fr::from(5), Fr::from(2)], &[Fr::from(10), Fr::from(25)]);
}

#[allow(non_snake_case)]
fn assert_final_exp_one(a: &[Fr; 2], b: &[Fr; 2]) {
    let (_P_vec, _Q_vec, P_ecpoints, Q_ecpoints) = generate_test_points_bls12_381(a, b);
    let bls12_381 = Bls12_381;
    let f = bls12_381.multi_miller_loop::<BLS12_381_PBE_LEN>(&P_ecpoints, &Q_ecpoints);
    bls12_381.assert_final_exp_is_one(f, &P_ecpoints, &Q_ecpoints);
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
    let mut P_vec = vec![];
    let mut Q_vec = vec![];
    for i in 0..2 {
        let p = G1Affine::generator() * a[i];
        let mut p = G1Affine::from(p);
        if i % 2 == 1 {
            p = p.neg();
        }
        let q = G2Affine::generator() * b[i];
        let q = G2Affine::from(q);
        P_vec.push(p);
        Q_vec.push(q);
    }
    let (P_ecpoints, Q_ecpoints) = izip!(P_vec.clone(), Q_vec.clone())
        .map(|(P, Q)| {
            (
                AffinePoint { x: P.x(), y: P.y() },
                AffinePoint { x: Q.x(), y: Q.y() },
            )
        })
        .unzip::<_, _, Vec<_>, Vec<_>>();
    (P_vec, Q_vec, P_ecpoints, Q_ecpoints)
}
