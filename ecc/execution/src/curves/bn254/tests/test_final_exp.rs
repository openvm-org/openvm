use halo2curves_axiom::bn256::{Fq, Fq2, Fr, G1Affine, G2Affine};
use num::{BigInt, Num};

use crate::{
    common::{ExpBigInt, FeltPrint, FinalExp, MultiMillerLoop},
    curves::bn254::Bn254,
    tests::utils::{
        generate_test_points, generate_test_points_generator, generate_test_points_generator_scalar,
    },
};

#[test]
#[allow(non_snake_case)]
fn test_final_exp_hint() {
    let rand_seeds = [496];
    let (_P_vec, _Q_vec, P_ecpoints, Q_ecpoints) =
        generate_test_points::<G1Affine, G2Affine, Fq, Fq2>(&rand_seeds);

    let bn254 = Bn254;
    let f = bn254.multi_miller_loop(&P_ecpoints, &Q_ecpoints);
    let (c, u) = bn254.final_exp_hint(f);
    println!("c: {:#?}", c);
    println!("u: {:#?}", u);
    // ------ native exponentiation ------
    let q = BigInt::from_str_radix(
        "21888242871839275222246405745257275088696311157297823662689037894645226208583",
        10,
    )
    .unwrap();
    let six_x_plus_2: BigInt = BigInt::from_str_radix("29793968203157093288", 10).unwrap();
    let q_pows = q.clone().pow(3) - q.clone().pow(2) + q;
    let lambda = six_x_plus_2.clone() + q_pows.clone();

    let c_lambda = c.exp(lambda);
    assert_eq!(f, c_lambda * u);
}

#[test]
#[allow(non_snake_case)]
fn test_assert_final_exp_is_one() {
    let (_P_vec, _Q_vec, P_ecpoints, Q_ecpoints) =
        generate_test_points_generator_scalar::<G1Affine, G2Affine, Fr, Fq, Fq2, 2>(
            &[Fr::from(1), Fr::from(1)],
            &[Fr::from(1), Fr::from(1)],
            // &[Fr::from(5), Fr::from(2)],
            // &[Fr::from(10), Fr::from(25)],
        );
    println!("P_ecpoints: {:#?}", P_ecpoints);
    println!("Q_ecpoints: {:#?}", Q_ecpoints);

    let bn254 = Bn254;
    let f = bn254.multi_miller_loop(&P_ecpoints, &Q_ecpoints);

    let f_cmp1 = bn254.multi_miller_loop(&[P_ecpoints[0].clone()], &[Q_ecpoints[0].clone()]);
    let f_cmp2 = bn254.multi_miller_loop(&[P_ecpoints[1].clone()], &[Q_ecpoints[1].clone()]);
    f_cmp1.felt_print("f_cmp1");
    f_cmp2.felt_print("f_cmp2");
    let f_cmp = f_cmp1 * f_cmp2;
    f_cmp.felt_print("f_cmp");
    assert_eq!(f, f_cmp);
    bn254.assert_final_exp_is_one(f, &P_ecpoints, &Q_ecpoints);
}
