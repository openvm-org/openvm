use halo2curves_axiom::bls12_381::{
    Fq, Fq12, Fq2, G1Affine, G2Affine, G2Prepared, MillerLoopResult,
};
use itertools::izip;
use rand::{rngs::StdRng, SeedableRng};
use subtle::ConditionallySelectable;

use crate::{
    common::{
        miller_add_step, miller_double_and_add_step, miller_double_step, multi_miller_loop, EcPoint,
    },
    curves::bls12_381::{
        line::{mul_023_by_023, mul_by_023, mul_by_02345},
        BLS12_381_XI, GNARK_BLS12_381_PBE,
    },
};

#[test]
#[allow(non_snake_case)]
fn test_multi_miller_loop_bls12_381() {
    // Generate random G1 and G2 points
    let rand_seeds = [8, 15, 29, 55, 166];
    let (P_vec, Q_vec) = rand_seeds
        .iter()
        .map(|seed| {
            let mut rng0 = StdRng::seed_from_u64(*seed);
            let p = G1Affine::random(&mut rng0);
            let mut rng1 = StdRng::seed_from_u64(*seed * 2);
            let q = G2Affine::random(&mut rng1);
            let either_identity = p.is_identity() | q.is_identity();
            let p = G1Affine::conditional_select(&p, &G1Affine::generator(), either_identity);
            let q = G2Affine::conditional_select(&q, &G2Affine::generator(), either_identity);
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
    let compare_miller = halo2curves_axiom::bls12_381::multi_miller_loop(terms.as_slice());
    let compare_final = compare_miller.final_exponentiation();

    // Run the multi-miller loop
    let f = multi_miller_loop::<Fq, Fq2, Fq12>(
        P_ecpoints.as_slice(),
        Q_ecpoints.as_slice(),
        GNARK_BLS12_381_PBE.as_slice(),
        BLS12_381_XI,
    );

    let wrapped_f = MillerLoopResult(f);
    let final_f = wrapped_f.final_exponentiation();

    // Run halo2curves final exponentiation on our multi_miller_loop output
    assert_eq!(final_f, compare_final);
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

    let P_ecpoint = EcPoint { x: P.x, y: P.y };
    let Q_ecpoint = EcPoint { x: Q.x, y: Q.y };

    // Setup constants
    let y_inv = P_ecpoint.y.invert().unwrap();
    let x_over_y = P_ecpoint.x * y_inv;

    // We want to check that Fp12 * (l_(S+Q+S) is equal to Fp12 * (l_(2S) * l_(S+Q))
    let mut f = Fq12::one();
    let mut Q_acc = Q_ecpoint.clone();

    // Initial step: double
    let (Q_acc_init, l_init) = miller_double_step::<Fq, Fq2>(Q_ecpoint.clone());
    let l_init = l_init.evaluate(x_over_y, y_inv);
    f = mul_by_023::<Fq, Fq2, Fq12>(f, l_init);

    // Test Q_acc_init == Q + Q
    let Q2 = Q + Q;
    let Q2 = G2Affine::from(Q2);
    assert_eq!(Q2.x, Q_acc_init.x);
    assert_eq!(Q2.y, Q_acc_init.y);

    Q_acc = Q_acc_init;

    // Now Q_acc is in a state where we can do a left vs right side test of double-and-add vs double then add:

    // Left side test: Double and add
    let (Q_acc_daa, l_S_plus_Q, l_S_plus_Q_plus_S) =
        miller_double_and_add_step::<Fq, Fq2>(Q_acc.clone(), Q_ecpoint.clone());
    let l_S_plus_Q_plus_S = l_S_plus_Q_plus_S.evaluate(x_over_y, y_inv);
    let l_S_plus_Q = l_S_plus_Q.evaluate(x_over_y, y_inv);
    let l_prod0 = mul_023_by_023(l_S_plus_Q, l_S_plus_Q_plus_S, BLS12_381_XI);
    let f_mul = mul_by_02345::<Fq, Fq2, Fq12>(f, l_prod0);

    // Test Q_acc_da == 2(2Q) + Q
    let Q4 = Q2 + Q2;
    let Q4_Q = Q4 + Q;
    let Q4_Q = G2Affine::from(Q4_Q);
    assert_eq!(Q4_Q.x, Q_acc_daa.x);
    assert_eq!(Q4_Q.y, Q_acc_daa.y);

    // Right side test: Double, then add
    let (Q_acc_d, l_2S) = miller_double_step::<Fq, Fq2>(Q_acc.clone());
    let (Q_acc_a, l_2S_plus_Q) = miller_add_step::<Fq, Fq2>(Q_acc_d, Q_ecpoint.clone());
    let l_2S = l_2S.evaluate(x_over_y, y_inv);
    let l_2S_plus_Q = l_2S_plus_Q.evaluate(x_over_y, y_inv);
    let l_prod1 = mul_023_by_023(l_2S, l_2S_plus_Q, BLS12_381_XI);
    let f_prod_mul = mul_by_02345::<Fq, Fq2, Fq12>(f, l_prod1);

    // Test line functions match
    let f_line_daa = mul_by_02345::<Fq, Fq2, Fq12>(Fq12::one(), l_prod0);
    let f_line_daa_final = MillerLoopResult(f_line_daa);
    let f_line_daa_final = f_line_daa_final.final_exponentiation();
    let f_line_da = mul_by_02345::<Fq, Fq2, Fq12>(Fq12::one(), l_prod1);
    let f_line_da_final = MillerLoopResult(f_line_da);
    let f_line_da_final = f_line_da_final.final_exponentiation();
    assert_eq!(f_line_daa_final, f_line_da_final);

    // Test Q_acc_a == 2(2Q) + Q
    assert_eq!(Q4_Q.x, Q_acc_a.x);
    assert_eq!(Q4_Q.y, Q_acc_a.y);

    // assert_eq!(f_mul, f_prod_mul);
    assert_eq!(Q_acc_daa.x, Q_acc_a.x);
    assert_eq!(Q_acc_daa.y, Q_acc_a.y);

    let wrapped_f_mul = MillerLoopResult(f_mul);
    let final_f_mul = wrapped_f_mul.final_exponentiation();

    let wrapped_f_prod_mul = MillerLoopResult(f_prod_mul);
    let final_f_prod_mul = wrapped_f_prod_mul.final_exponentiation();

    assert_eq!(final_f_mul, final_f_prod_mul);
}
