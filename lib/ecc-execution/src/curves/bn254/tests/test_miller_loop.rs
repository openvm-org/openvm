use axvm_ecc::{
    curve::bn254::{Fq, Fq2, G1Affine, G2Affine},
    pairing::MultiMillerLoop,
};
use halo2curves_axiom::{
    bn256::{G2Prepared, Gt},
    pairing::MillerLoopResult,
};

use crate::{
    curves::bn254::{Bn254, BN254_PBE_NAF_LEN},
    tests::utils::generate_test_points,
};

#[allow(non_snake_case)]
fn run_miller_loop_test(rand_seeds: &[u64]) {
    let (P_vec, Q_vec, P_ecpoints, Q_ecpoints) =
        generate_test_points::<G1Affine, G2Affine, Fq, Fq2>(rand_seeds);

    // Compare against halo2curves implementation
    let g2_prepareds = Q_vec
        .iter()
        .map(|q| G2Prepared::from(*q.inner()))
        .collect::<Vec<_>>();
    let terms = P_vec
        .iter()
        .map(|p| p.inner())
        .zip(g2_prepareds.iter())
        .collect::<Vec<_>>();
    let compare_miller = halo2curves_axiom::bn256::multi_miller_loop(terms.as_slice());
    let compare_final = compare_miller.final_exponentiation();

    // Run the multi-miller loop
    let bn254 = Bn254;
    let f =
        bn254.multi_miller_loop::<BN254_PBE_NAF_LEN>(P_ecpoints.as_slice(), Q_ecpoints.as_slice());

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
