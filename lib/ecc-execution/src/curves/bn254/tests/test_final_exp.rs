use axvm_ecc::{
    halo2curves_shims::ExpBigInt,
    pairing::{FinalExp, MultiMillerLoop},
};
use halo2curves_axiom::bn256::{Fq, Fq2, G1Affine, G2Affine};
use num_bigint::{BigUint, Sign};
use num_traits::Num;

use crate::{curves::bn254::Bn254, tests::utils::generate_test_points_generator_scalar};

#[test]
#[allow(non_snake_case)]
fn test_bn254_final_exp_hint() {
    let (_P_vec, _Q_vec, P_ecpoints, Q_ecpoints) =
        generate_test_points_generator_scalar::<G1Affine, G2Affine, Fq, Fq2, 2>(&[3, -6], &[8, 4]);

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
    assert_final_exp_one(&[1, 1], &[-1, 1]);
}

#[test]
#[allow(non_snake_case)]
fn test_bn254_assert_final_exp_is_one_scalar_other() {
    assert_final_exp_one(&[5, 2], &[-10, 25]);
}

#[allow(non_snake_case)]
fn assert_final_exp_one<const N: usize>(a: &[i32; N], b: &[i32; N]) {
    let (_P_vec, _Q_vec, P_ecpoints, Q_ecpoints) =
        generate_test_points_generator_scalar::<G1Affine, G2Affine, Fq, Fq2, N>(a, b);
    let f = Bn254::multi_miller_loop(&P_ecpoints, &Q_ecpoints);
    Bn254::assert_final_exp_is_one(&f, &P_ecpoints, &Q_ecpoints);
}
