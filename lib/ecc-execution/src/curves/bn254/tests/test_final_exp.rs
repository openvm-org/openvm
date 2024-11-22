use axvm_ecc::{
    halo2curves_shims::ExpBigInt,
    pairing::{FinalExp, MultiMillerLoop},
    AffinePoint,
};
use halo2curves_axiom::bn256::{Fq, Fq2, G1Affine, G2Affine};
use itertools::izip;
use num_bigint::{BigUint, Sign};
use num_traits::Num;

use crate::curves::bn254::Bn254;

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

    let c_lambda = c.exp_bigint(Sign::Plus, lambda);
    assert_eq!(f * u, c_lambda);
}

#[test]
#[allow(non_snake_case)]
fn test_bn254_assert_final_exp_is_one_scalar_ones() {
    assert_final_exp_one(&[1, 2], &[-2, 1]);
}

#[test]
#[allow(non_snake_case)]
fn test_bn254_assert_final_exp_is_one_scalar_other() {
    assert_final_exp_one(&[5, 2], &[-10, 25]);
}

#[allow(non_snake_case)]
fn assert_final_exp_one<const N: usize>(a: &[i32; N], b: &[i32; N]) {
    let (_P_vec, _Q_vec, P_ecpoints, Q_ecpoints) = generate_test_points_generator_scalar::<N>(a, b);
    let f = Bn254::multi_miller_loop(&P_ecpoints, &Q_ecpoints);
    Bn254::assert_final_exp_is_one(&f, &P_ecpoints, &Q_ecpoints);
}

/// Generates test points for N number of points for an elliptic curve pairing, where the inputs `a` and `b` are
/// scalars of generators in G1 and G2, respectively. Importantly, for every even index, the generator P point is
/// negated (reflected an the x-axis). Outputs the vectors of P and Q points as well as the corresponding P and Q
/// EcPoint structs.
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
    assert!(N % 2 == 0, "Must have even number of P and Q scalars");

    // fn get_felt_abs<Fp: Field>(v: i32) -> Fp {
    //     if v == 0 {
    //         return Fp::ZERO;
    //     }
    //     let mut fp = Fp::ONE;
    //     for _ in 1..v.abs() {
    //         fp += Fp::ONE;
    //     }
    //     fp
    // }

    let mut P_vec: Vec<G1Affine> = vec![];
    let mut Q_vec: Vec<G2Affine> = vec![];
    for i in 0..N {
        let p_a = Fq::from_raw([a[i].unsigned_abs() as u64, 0, 0, 0]);
        let mut p = G1Affine::generator();
        p.x *= p_a;
        p.y *= p_a;
        // let p_a = get_felt_abs::<Fq>(a[i]);
        // let mut p_mul: G1Affine = if a[i].is_negative() {
        //     A1::new(p_a * p.x(), p_a * p.y())
        // } else {
        //     A1::new(p_a * p.x(), p_a * p.y())
        // };
        if i % 2 == 1 {
            p = -p;
        }
        let q_b = Fq::from_raw([b[i].unsigned_abs() as u64, 0, 0, 0]);
        let mut q = G2Affine::generator();
        q.x.c0 *= q_b;
        q.x.c1 *= q_b;
        q.y.c0 *= q_b;
        q.y.c1 *= q_b;
        // let q_b = get_felt_abs::<Fp2>(b[i]);
        // let q_mul: A2 = if b[i].is_negative() {
        //     A2::new(q_b * q.x(), q_b * q.y())
        // } else {
        //     A2::new(q_b * q.x(), q_b * q.y())
        // };
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
