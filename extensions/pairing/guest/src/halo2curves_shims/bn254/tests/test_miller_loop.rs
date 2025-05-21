use alloc::vec::Vec;

use halo2curves_axiom::bn256::G2Prepared;

use super::generate_test_points_bn254;
use crate::{
    halo2curves_shims::bn254::{test_utils::assert_miller_results_eq, Bn254},
    pairing::MultiMillerLoop,
};

#[allow(non_snake_case)]
fn run_miller_loop_test(rand_seeds: &[u64]) {
    let (P_vec, Q_vec, P_ecpoints, Q_ecpoints) = generate_test_points_bn254(rand_seeds);

    // Compare against halo2curves implementation
    let g2_prepareds = Q_vec
        .iter()
        .map(|q| G2Prepared::from(*q))
        .collect::<Vec<_>>();
    let terms = P_vec.iter().zip(g2_prepareds.iter()).collect::<Vec<_>>();
    let compare_miller = halo2curves_axiom::bn256::multi_miller_loop(terms.as_slice());
    // Run the multi-miller loop
    let f = Bn254::multi_miller_loop(P_ecpoints.as_slice(), Q_ecpoints.as_slice());
    assert_miller_results_eq(compare_miller, f);
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
