use halo2curves_axiom::bn256::{Fq, Fq2, G1Affine, G2Affine};

use crate::{
    common::{FinalExp, MultiMillerLoop},
    curves::bn254::Bn254,
    tests::utils::{generate_test_points, generate_test_points_generator},
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
}

#[test]
#[allow(non_snake_case)]
fn test_assert_final_exp_is_one() {
    let (_P_vec, _Q_vec, P_ecpoints, Q_ecpoints) =
        generate_test_points_generator::<G1Affine, G2Affine, Fq, Fq2>();

    let bn254 = Bn254;
    let f = bn254.multi_miller_loop(&P_ecpoints, &Q_ecpoints);
    bn254.assert_final_exp_is_one(f, &P_ecpoints, &Q_ecpoints);
}
