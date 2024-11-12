use axvm_ecc::{
    curve::bn254::{Fq, Fq2, Fr, G1Affine, G2Affine},
    field::ExpBigInt,
    pairing::{FinalExp, MultiMillerLoop},
};
use num::{BigInt, Num};

use crate::{
    curves::bn254::{Bn254, BN254_PBE_NAF_LEN},
    tests::utils::generate_test_points_generator_scalar,
};

#[test]
#[allow(non_snake_case)]
fn test_bn254_final_exp_hint() {
    let (_P_vec, _Q_vec, P_ecpoints, Q_ecpoints) =
        generate_test_points_generator_scalar::<G1Affine, G2Affine, Fr, Fq, Fq2, 2>(
            &[Fr::from(3), Fr::from(6)],
            &[Fr::from(8), Fr::from(4)],
        );

    let bn254 = Bn254;
    let f = bn254.multi_miller_loop(&P_ecpoints, &Q_ecpoints);
    let (c, u) = bn254.final_exp_hint(f);

    let q = BigInt::from_str_radix(
        "21888242871839275222246405745257275088696311157297823662689037894645226208583",
        10,
    )
    .unwrap();
    let six_x_plus_2: BigInt = BigInt::from_str_radix("29793968203157093288", 10).unwrap();
    let q_pows = q.clone().pow(3) - q.clone().pow(2) + q;
    let lambda = six_x_plus_2.clone() + q_pows.clone();

    let c_lambda = c.exp_bigint(lambda);
    assert_eq!(f * u, c_lambda);
}

#[test]
#[allow(non_snake_case)]
fn test_bn254_assert_final_exp_is_one_scalar_ones() {
    assert_final_exp_one(&[Fr::from(1), Fr::from(1)], &[Fr::from(1), Fr::from(1)]);
}

#[test]
#[allow(non_snake_case)]
fn test_bn254_assert_final_exp_is_one_scalar_other() {
    assert_final_exp_one(&[Fr::from(5), Fr::from(2)], &[Fr::from(10), Fr::from(25)]);
}

#[allow(non_snake_case)]
fn assert_final_exp_one<const N: usize>(a: &[Fr; N], b: &[Fr; N]) {
    let (_P_vec, _Q_vec, P_ecpoints, Q_ecpoints) =
        generate_test_points_generator_scalar::<G1Affine, G2Affine, Fr, Fq, Fq2, N>(a, b);
    let bn254 = Bn254;
    let f = bn254.multi_miller_loop(&P_ecpoints, &Q_ecpoints);
    bn254.assert_final_exp_is_one(f, &P_ecpoints, &Q_ecpoints);
}
