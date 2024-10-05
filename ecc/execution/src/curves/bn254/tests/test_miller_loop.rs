use halo2curves_axiom::{
    bn256::{G1Affine, G2Affine, G2Prepared, Gt},
    pairing::MillerLoopResult,
};
use itertools::izip;
use rand::{rngs::StdRng, SeedableRng};

use crate::{
    common::{EcPoint, MultiMillerLoop},
    curves::bn254::Bn254,
};

#[allow(non_snake_case)]
fn run_miller_loop_test(rand_seeds: &[u64]) {
    let (P_vec, Q_vec) = rand_seeds
        .iter()
        .map(|seed| {
            let mut rng0 = StdRng::seed_from_u64(*seed);
            let p = G1Affine::random(&mut rng0);
            let mut rng1 = StdRng::seed_from_u64(*seed * 2);
            let q = G2Affine::random(&mut rng1);
            (p, q)
        })
        .unzip::<_, _, Vec<_>, Vec<_>>();
    let (P_ecpoints, Q_ecpoints) = izip!(P_vec.clone(), Q_vec.clone())
        .map(|(P, Q)| (EcPoint { x: P.x, y: P.y }, EcPoint { x: Q.x, y: Q.y }))
        .unzip::<_, _, Vec<_>, Vec<_>>();

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
