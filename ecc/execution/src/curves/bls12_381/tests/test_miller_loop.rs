use halo2curves_axiom::bls12_381::{
    Fq, Fq12, Fq2, G1Affine, G2Affine, G2Prepared, MillerLoopResult, G2,
};
use rand::{rngs::StdRng, SeedableRng};
use subtle::{Choice, ConditionallySelectable};

use crate::{
    common::EcPoint,
    curves::bls12_381::{
        line::{evaluate_line, mul_023_by_023, mul_by_023, mul_by_02345},
        miller_add_step, miller_double_and_add_step, miller_double_step, multi_miller_loop,
        multi_miller_loop_separate_double_plus_add,
        multi_miller_loop_separate_double_plus_add_custom_debug, BLS12_381_XI, GNARK_BLS12_381_PBE,
    },
};

#[test]
#[allow(non_snake_case)]
fn test_multi_miller_loop_bls12_381() {
    // Generate random G1 and G2 points
    // let mut rng0 = StdRng::seed_from_u64(8);
    // let P = G1Affine::random(&mut rng0);
    // let mut rng1 = StdRng::seed_from_u64(8 * 2);
    // let Q = G2Affine::random(&mut rng1);
    // let either_identity = P.is_identity() | Q.is_identity();
    // let P = G1Affine::conditional_select(&P, &G1Affine::generator(), either_identity);
    // let Q = G2Affine::conditional_select(&Q, &G2Affine::generator(), either_identity);

    let P = G1Affine::generator();
    let Q = G2Affine::generator();

    let P_ecpoint = EcPoint { x: P.x, y: P.y };
    let Q_ecpoint = EcPoint { x: Q.x, y: Q.y };

    // Compare against halo2curves implementation
    // let g2_prepared = G2Prepared::from(Q);
    // let compare_miller = halo2curves_axiom::bls12_381::multi_miller_loop(&[(&P, &g2_prepared)]);
    // let compare_final = compare_miller.final_exponentiation();
    let compare_final = halo2curves_axiom::bls12_381::pairing(&P, &Q);

    // Run the multi-miller loop
    let f = multi_miller_loop::<Fq, Fq2, Fq12>(
        // let f = multi_miller_loop_separate_double_plus_add::<Fq, Fq2, Fq12>(
        &[P_ecpoint],
        &[Q_ecpoint],
        GNARK_BLS12_381_PBE.as_slice(),
        BLS12_381_XI,
    );

    // let f = multi_miller_loop_separate_double_plus_add_custom_debug::<Fq, Fq2, Fq12>(
    //     &[P_ecpoint],
    //     &[Q_ecpoint],
    //     GNARK_BLS12_381_PBE.as_slice(),
    //     BLS12_381_XI,
    //     |label, idx, elem: EcPoint<Fq2>| {
    //         let mut Q_acc_aff = G2Affine::generator();
    //         Q_acc_aff.x = elem.x;
    //         Q_acc_aff.y = elem.y;
    //         let Q_acc_proj = G2::from(Q_acc_aff);
    //         println!("{}, {}: {:?}", idx, label, Q_acc_proj.x.c0.0);
    //     },
    //     |label, idx, elem: Fq12| {
    //         println!("{}, {}: {:?}", idx, label, elem.c0.c0.c0.0);
    //     },
    // );

    // let f = multi_miller_loop_gnark::<Fq, Fq2, Fq12>(
    //     &[P_ecpoint],
    //     &[Q_ecpoint],
    //     GNARK_BLS12_381_PBE.as_slice(),
    //     BLS12_381_XI,
    //     |label, ecpt: EcPoint<Fq2>| {
    //         println!("{}", label);
    //         println!("ecpt.x.c0.0: {:?}", ecpt.x.c0.0);
    //         println!("ecpt.x.c0.1: {:?}", ecpt.x.c1.0);
    //         println!("ecpt.y.c0.0: {:?}", ecpt.y.c0.0);
    //         println!("ecpt.y.c0.1: {:?}", ecpt.y.c1.0);
    //     },
    // );

    let wrapped_f = MillerLoopResult(f);
    let final_f = wrapped_f.final_exponentiation();
    println!("final_f: {:#?}", final_f);

    let cf = final_f.0;
    println!("cf.c0.c0.c0: {:?}", cf.c0.c0.c0.0);
    println!("cf.c0.c0.c1: {:?}", cf.c0.c0.c1.0);
    println!("cf.c0.c1.c0: {:?}", cf.c0.c1.c0.0);
    println!("cf.c0.c1.c1: {:?}", cf.c0.c1.c1.0);
    println!("cf.c0.c2.c0: {:?}", cf.c0.c2.c0.0);
    println!("cf.c0.c2.c1: {:?}", cf.c0.c2.c1.0);
    println!("cf.c1.c0.c0: {:?}", cf.c1.c0.c0.0);
    println!("cf.c1.c0.c1: {:?}", cf.c1.c0.c1.0);
    println!("cf.c1.c1.c0: {:?}", cf.c1.c1.c0.0);
    println!("cf.c1.c1.c1: {:?}", cf.c1.c1.c1.0);
    println!("cf.c1.c2.c0: {:?}", cf.c1.c2.c0.0);
    println!("cf.c1.c2.c1: {:?}", cf.c1.c2.c1.0);

    // Run halo2curves final exponentiation on our multi_miller_loop output
    assert_eq!(final_f, compare_final);
}

#[test]
#[allow(non_snake_case)]
fn test_on_curve() {
    let two = Fq::one() + Fq::one();
    let three = Fq::one() + Fq::one() + Fq::one();
    let mut P = G1Affine::default();
    P.x = Fq::zero();
    P.y = Fq::one();
    let mut Q: G2Affine = G2Affine::default();
    Q.x = Fq2 {
        c0: Fq::zero(),
        c1: Fq::one(),
    };
    Q.y = Fq2 { c0: two, c1: three };
    let P_on_curve: bool = P.is_on_curve().into();
    let Q_on_curve: bool = Q.is_on_curve().into();
    assert!(P_on_curve);
    assert!(Q_on_curve);
}

#[test]
#[allow(non_snake_case)]
fn test_f_mul() {
    // Generate random G1 and G2 points
    // let mut rng0 = StdRng::seed_from_u64(8);
    // let P = G1Affine::random(&mut rng0);
    // let mut rng1 = StdRng::seed_from_u64(8 * 2);
    // let Q = G2Affine::random(&mut rng1);
    // let either_identity = P.is_identity() | Q.is_identity();
    // let P = G1Affine::conditional_select(&P, &G1Affine::generator(), either_identity);
    // let Q = G2Affine::conditional_select(&Q, &G2Affine::generator(), either_identity);

    // let two = Fq::one() + Fq::one();
    // let three = Fq::one() + Fq::one() + Fq::one();
    // let mut P = G1Affine::default();
    // P.x = Fq::zero();
    // P.y = Fq::one();
    // let mut Q: G2Affine = G2Affine::default();
    // Q.x = Fq2 {
    //     c0: Fq::zero(),
    //     c1: Fq::one(),
    // };
    // Q.y = Fq2 { c0: two, c1: three };

    let P = G1Affine::generator();
    let Q = G2Affine::generator();

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
    let l_init = evaluate_line::<Fq, Fq2>(l_init, x_over_y.clone(), y_inv.clone());
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
    let l_S_plus_Q_plus_S =
        evaluate_line::<Fq, Fq2>(l_S_plus_Q_plus_S, x_over_y.clone(), y_inv.clone());
    let l_S_plus_Q = evaluate_line::<Fq, Fq2>(l_S_plus_Q, x_over_y.clone(), y_inv.clone());
    let l_prod0 = mul_023_by_023(l_S_plus_Q, l_S_plus_Q_plus_S, BLS12_381_XI);
    let f_mul = mul_by_02345::<Fq, Fq2, Fq12>(f.clone(), l_prod0);

    // Test Q_acc_da == 2(2Q) + Q
    let Q4 = Q2 + Q2;
    let Q4_Q = Q4 + Q;
    let Q4_Q = G2Affine::from(Q4_Q);
    assert_eq!(Q4_Q.x, Q_acc_daa.x);
    assert_eq!(Q4_Q.y, Q_acc_daa.y);

    // Right side test: Double, then add
    let (Q_acc_d, l_2S) = miller_double_step::<Fq, Fq2>(Q_acc.clone());
    let (Q_acc_a, l_2S_plus_Q) = miller_add_step::<Fq, Fq2>(Q_acc_d, Q_ecpoint.clone());
    let l_2S = evaluate_line::<Fq, Fq2>(l_2S, x_over_y.clone(), y_inv.clone());
    let l_2S_plus_Q = evaluate_line::<Fq, Fq2>(l_2S_plus_Q, x_over_y.clone(), y_inv.clone());
    let l_prod1 = mul_023_by_023(l_2S, l_2S_plus_Q, BLS12_381_XI);
    let f_prod_mul = mul_by_02345::<Fq, Fq2, Fq12>(f.clone(), l_prod1);

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
