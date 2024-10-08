use halo2curves_axiom::{
    bn256::{Fq, Fq12, Fq2, G1Affine, G2Affine, G2Prepared, Gt},
    pairing::MillerLoopResult,
};

use crate::{common::MultiMillerLoop, curves::bn254::Bn254, tests::utils::generate_test_points};

#[allow(non_snake_case)]
fn run_miller_loop_test(rand_seeds: &[u64]) {
    let (P_vec, Q_vec, P_ecpoints, Q_ecpoints) =
        generate_test_points::<G1Affine, G2Affine, Fq, Fq2>(rand_seeds);

    // Compare against halo2curves implementation
    let g2_prepareds = Q_vec
        .iter()
        .map(|q| G2Prepared::from(*q))
        .collect::<Vec<_>>();
    let terms = P_vec.iter().zip(g2_prepareds.iter()).collect::<Vec<_>>();
    let compare_miller = halo2curves_axiom::bn256::multi_miller_loop(terms.as_slice());
    let compare_final = compare_miller.final_exponentiation();

    // Run the multi-miller loop
    let bn254 = Bn254;
    let f = bn254.multi_miller_loop(P_ecpoints.as_slice(), Q_ecpoints.as_slice());

    let wrapped_f = Gt(f);
    let final_f = wrapped_f.final_exponentiation();

    // Run halo2curves final exponentiation on our multi_miller_loop output
    assert_eq!(final_f, compare_final);
}

#[test]
#[allow(non_snake_case)]
fn test_single_miller_loop_bn254() {
    let rand_seeds = [925];
    run_miller_loop_test(&rand_seeds);
}

#[test]
#[allow(non_snake_case)]
fn test_multi_miller_loop_bn254() {
    let rand_seeds = [8, 15, 29, 55, 166];
    run_miller_loop_test(&rand_seeds);
}

#[test]
#[allow(non_snake_case)]
fn test_miller_loop_final_exp() {
    let rand_seeds = [619];
    let (P_vec, Q_vec, P_ecpoints, Q_ecpoints) =
        generate_test_points::<G1Affine, G2Affine, Fq, Fq2>(&rand_seeds);

    // Run the multi-miller loop with embedded exponents
    let bn254 = Bn254;
    let f = bn254.multi_miller_loop_embedded_exp(
        P_ecpoints.as_slice(),
        Q_ecpoints.as_slice(),
        Some(Fq12::one()),
    );
    // f.assert_final_exp_is_one();

    // // Compare against miller loop with final exp
    // let f_compare = bn254.multi_miller_loop(P_ecpoints.as_slice(), Q_ecpoints.as_slice());
    // let wrapped_f_compare = Gt(f_compare);
    // let final_f_compare = wrapped_f_compare.final_exponentiation();

    // assert_eq!(f, final_f_compare);
}
