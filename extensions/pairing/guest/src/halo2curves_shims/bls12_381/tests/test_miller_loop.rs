use alloc::vec::Vec;

use halo2curves_axiom::bls12_381::{Fq12, G1Affine, G2Affine, G2Prepared};
use openvm_ecc_guest::{algebra::Field, AffinePoint};
use rand::{rngs::StdRng, SeedableRng};
use subtle::ConditionallySelectable;

use super::generate_test_points_bls12_381;
use crate::{
    halo2curves_shims::bls12_381::{
        test_utils::{assert_miller_results_eq, final_exp},
        Bls12_381,
    },
    pairing::{Evaluatable, LineMulMType, MillerStep, MultiMillerLoop},
};

#[allow(non_snake_case)]
fn run_miller_loop_test(rand_seeds: &[u64]) {
    let (P_vec, Q_vec, P_ecpoints, Q_ecpoints) = generate_test_points_bls12_381(rand_seeds);

    // Compare against halo2curves implementation
    let g2_prepareds = Q_vec
        .iter()
        .map(|q| G2Prepared::from(*q))
        .collect::<Vec<_>>();
    let terms = P_vec.iter().zip(g2_prepareds.iter()).collect::<Vec<_>>();
    let compare_miller = halo2curves_axiom::bls12_381::multi_miller_loop(terms.as_slice());
    // Run the multi-miller loop
    let f = Bls12_381::multi_miller_loop(P_ecpoints.as_slice(), Q_ecpoints.as_slice());
    assert_miller_results_eq(compare_miller, f);
}

#[test]
#[allow(non_snake_case)]
fn test_single_miller_loop_bls12_381() {
    let rand_seeds = [888];
    run_miller_loop_test(&rand_seeds);
}

#[test]
#[allow(non_snake_case)]
fn test_multi_miller_loop_bls12_381() {
    let rand_seeds = [8, 15, 29, 55, 166];
    run_miller_loop_test(&rand_seeds);
}

#[test]
#[allow(non_snake_case)]
#[allow(unused_assignments)]
fn test_f_mul() {
    // Generate random G1 and G2 points
    let mut rng0 = StdRng::seed_from_u64(2);
    let P = G1Affine::random(&mut rng0);
    let mut rng1 = StdRng::seed_from_u64(2 * 2);
    let Q = G2Affine::random(&mut rng1);
    let either_identity = P.is_identity() | Q.is_identity();
    let P = G1Affine::conditional_select(&P, &G1Affine::generator(), either_identity);
    let Q = G2Affine::conditional_select(&Q, &G2Affine::generator(), either_identity);

    let P_ecpoint = AffinePoint { x: P.x, y: P.y };
    let Q_ecpoint = AffinePoint { x: Q.x, y: Q.y };

    // Setup constants
    let y_inv = P_ecpoint.y.invert().unwrap();
    let x_over_y = P_ecpoint.x * y_inv;

    // We want to check that Fp12 * (l_(S+Q+S) is equal to Fp12 * (l_(2S) * l_(S+Q))
    let mut f = Fq12::one();
    let mut Q_acc = Q_ecpoint.clone();

    // Initial step: double
    let (Q_acc_init, l_init) = Bls12_381::miller_double_step(&Q_ecpoint);
    let l_init = l_init.evaluate(&(x_over_y, y_inv));
    f = Bls12_381::mul_by_023(&f, &l_init);

    // Test Q_acc_init == Q + Q
    let Q2 = Q + Q;
    let Q2 = G2Affine::from(Q2);
    assert_eq!(Q2.x, Q_acc_init.x);
    assert_eq!(Q2.y, Q_acc_init.y);

    Q_acc = Q_acc_init;

    // Now Q_acc is in a state where we can do a left vs right side test of double-and-add vs double
    // then add:

    // Left side test: Double and add
    let (Q_acc_daa, l_S_plus_Q, l_S_plus_Q_plus_S) =
        Bls12_381::miller_double_and_add_step(&Q_acc, &Q_ecpoint);
    let l_S_plus_Q_plus_S = l_S_plus_Q_plus_S.evaluate(&(x_over_y, y_inv));
    let l_S_plus_Q = l_S_plus_Q.evaluate(&(x_over_y, y_inv));
    let l_prod0 = Bls12_381::mul_023_by_023(&l_S_plus_Q, &l_S_plus_Q_plus_S);
    let f_mul = Bls12_381::mul_by_02345(&f, &l_prod0);

    // Test Q_acc_da == 2(2Q) + Q
    let Q4 = Q2 + Q2;
    let Q4_Q = Q4 + Q;
    let Q4_Q = G2Affine::from(Q4_Q);
    assert_eq!(Q4_Q.x, Q_acc_daa.x);
    assert_eq!(Q4_Q.y, Q_acc_daa.y);

    // Right side test: Double, then add
    let (Q_acc_d, l_2S) = Bls12_381::miller_double_step(&Q_acc);
    let (Q_acc_a, l_2S_plus_Q) = Bls12_381::miller_add_step(&Q_acc_d, &Q_ecpoint);
    let l_2S = l_2S.evaluate(&(x_over_y, y_inv));
    let l_2S_plus_Q = l_2S_plus_Q.evaluate(&(x_over_y, y_inv));
    let l_prod1 = Bls12_381::mul_023_by_023(&l_2S, &l_2S_plus_Q);
    let f_prod_mul = Bls12_381::mul_by_02345(&f, &l_prod1);

    // Test line functions match
    let f_line_daa = Bls12_381::mul_by_02345(&Fq12::ONE, &l_prod0);
    let f_line_da = Bls12_381::mul_by_02345(&Fq12::ONE, &l_prod1);
    assert_eq!(final_exp(f_line_daa), final_exp(f_line_da));
    // Test Q_acc_a == 2(2Q) + Q
    assert_eq!(Q4_Q.x, Q_acc_a.x);
    assert_eq!(Q4_Q.y, Q_acc_a.y);

    // assert_eq!(f_mul, f_prod_mul);
    assert_eq!(Q_acc_daa.x, Q_acc_a.x);
    assert_eq!(Q_acc_daa.y, Q_acc_a.y);

    assert_eq!(final_exp(f_mul), final_exp(f_prod_mul));
}
