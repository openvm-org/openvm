use alloc::vec::Vec;
use core::ops::Neg;

use halo2curves_axiom::bn256::{Fq, Fq2, Fr, G1Affine, G2Affine};
use itertools::izip;
use num_bigint::BigUint;
use num_traits::Num;
use openvm_ecc_guest::{algebra::ExpBytes, AffinePoint};

use crate::{
    halo2curves_shims::bn254::Bn254,
    pairing::{FinalExp, MultiMillerLoop},
};

#[test]
#[allow(non_snake_case)]
fn test_bn254_final_exp_hint() {
    let (_P_vec, _Q_vec, P_ecpoints, Q_ecpoints) =
        generate_test_points_generator_scalar::<2>(&[3, -6], &[8, 4]);

    let f = Bn254::multi_miller_loop(&P_ecpoints, &Q_ecpoints);
    let (c, u) = Bn254::final_exp_hint(&f);

    let q = BigUint::from_str_radix(
        "21888242871839275222246405745257275088696311157297823662689037894645226208583",
        10,
    )
    .unwrap();
    let six_x_plus_2: BigUint = BigUint::from_str_radix("29793968203157093288", 10).unwrap();
    let q_pows = q.clone().pow(3) - q.clone().pow(2) + q;
    let lambda = six_x_plus_2.clone() + q_pows.clone();

    let c_lambda = c.exp_bytes(true, &lambda.to_bytes_be());

    assert_eq!(f * u, c_lambda);
}

#[test]
#[allow(non_snake_case)]
fn test_bn254_assert_final_exp_is_one_scalar0() {
    assert_final_exp_one(&[1, 2], &[-2, 1]);
}

#[test]
#[allow(non_snake_case)]
fn test_bn254_assert_final_exp_is_one_scalar1() {
    assert_final_exp_one(&[-5, -2], &[-10, 25]);
}

#[allow(non_snake_case)]
fn assert_final_exp_one<const N: usize>(a: &[i32; N], b: &[i32; N]) {
    let (_P_vec, _Q_vec, P_ecpoints, Q_ecpoints) = generate_test_points_generator_scalar::<N>(a, b);
    let f = Bn254::multi_miller_loop(P_ecpoints.as_slice(), Q_ecpoints.as_slice());
    Bn254::assert_final_exp_is_one(&f, &P_ecpoints, &Q_ecpoints);
}

/// Generates test points for N number of points for an elliptic curve pairing, where the inputs `a`
/// and `b` are scalars of generators in G1 and G2, respectively. Importantly, for every even index,
/// the generator P point is negated (reflected an the x-axis). Outputs the vectors of P and Q
/// points as well as the corresponding P and Q EcPoint structs.
#[allow(non_snake_case)]
#[allow(clippy::type_complexity)]
pub fn generate_test_points_generator_scalar<const N: usize>(
    a: &[i32; N],
    b: &[i32; N],
) -> (
    Vec<G1Affine>,
    Vec<G2Affine>,
    Vec<AffinePoint<Fq>>,
    Vec<AffinePoint<Fq2>>,
) {
    assert!(
        N.is_multiple_of(2),
        "Must have even number of P and Q scalars"
    );
    let mut P_vec: Vec<G1Affine> = Vec::new();
    let mut Q_vec: Vec<G2Affine> = Vec::new();
    for i in 0..N {
        let s_a = Fr::from(a[i].unsigned_abs() as u64);
        let p = G1Affine::generator() * s_a;
        let mut p = G1Affine::from(p);
        if a[i].is_negative() {
            p = p.neg();
        }
        let s_b = Fr::from(b[i].unsigned_abs() as u64);
        let q = G2Affine::generator() * s_b;
        let mut q = G2Affine::from(q);
        if b[i].is_negative() {
            q = q.neg();
        }
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
