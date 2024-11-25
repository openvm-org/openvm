use alloc::vec::Vec;

use axvm_algebra::{field::FieldExtension, IntMod};
use group::ff::Field;
use halo2curves_axiom::{
    bn256::{Fq, Fq12, Fq2, Fq6, Fr, G1Affine, G2Affine, G2Prepared, Gt, FROBENIUS_COEFF_FQ12_C1},
    pairing::MillerLoopResult,
};
use num_bigint::BigUint;
use num_traits::Num;
use rand::{rngs::StdRng, SeedableRng};

use super::{Fp, Fp12, Fp2};
use crate::{
    bn254::Bn254,
    pairing::{
        fp2_invert_assign, fp6_invert_assign, fp6_square_assign, ExpBytes, FinalExp,
        MultiMillerLoop, PairingCheck, PairingIntrinsics,
    },
    AffinePoint,
};

fn convert_bn254_halo2_fq_to_fp(x: Fq) -> Fp {
    let bytes = x.to_bytes();
    Fp::from_le_bytes(&bytes)
}

fn convert_bn254_halo2_fq2_to_fp2(x: Fq2) -> Fp2 {
    Fp2::new(
        convert_bn254_halo2_fq_to_fp(x.c0),
        convert_bn254_halo2_fq_to_fp(x.c1),
    )
}

fn convert_bn254_halo2_fq12_to_fp12(x: Fq12) -> Fp12 {
    Fp12 {
        c: x.to_coeffs().map(convert_bn254_halo2_fq2_to_fp2),
    }
}

fn convert_bn254_fp_to_halo2_fq(x: Fp) -> Fq {
    let bytes =
        x.0.chunks(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
    Fq::from_raw(bytes)
}

fn convert_bn254_fp2_to_halo2_fq2(x: Fp2) -> Fq2 {
    Fq2 {
        c0: convert_bn254_fp_to_halo2_fq(x.c0.clone()),
        c1: convert_bn254_fp_to_halo2_fq(x.c1.clone()),
    }
}

fn convert_bn254_fp12_to_halo2_fq12(x: Fp12) -> Fq12 {
    let c = x.to_coeffs();
    Fq12 {
        c0: Fq6 {
            c0: convert_bn254_fp2_to_halo2_fq2(c[0].clone()),
            c1: convert_bn254_fp2_to_halo2_fq2(c[2].clone()),
            c2: convert_bn254_fp2_to_halo2_fq2(c[4].clone()),
        },
        c1: Fq6 {
            c0: convert_bn254_fp2_to_halo2_fq2(c[1].clone()),
            c1: convert_bn254_fp2_to_halo2_fq2(c[3].clone()),
            c2: convert_bn254_fp2_to_halo2_fq2(c[5].clone()),
        },
    }
}

fn convert_bn254_affine_point_halo2_fq_to_fp(p: AffinePoint<Fq>) -> AffinePoint<Fp> {
    AffinePoint::new(
        convert_bn254_halo2_fq_to_fp(p.x),
        convert_bn254_halo2_fq_to_fp(p.y),
    )
}

fn convert_bn254_affine_point_halo2_fq2_to_fp2(p: AffinePoint<Fq2>) -> AffinePoint<Fp2> {
    AffinePoint::new(
        convert_bn254_halo2_fq2_to_fp2(p.x),
        convert_bn254_halo2_fq2_to_fp2(p.y),
    )
}

#[test]
fn test_bn254_frobenius_coeffs() {
    #[allow(clippy::needless_range_loop)]
    for i in 0..12 {
        for j in 0..5 {
            assert_eq!(
                Bn254::FROBENIUS_COEFFS[i][j],
                convert_bn254_halo2_fq2_to_fp2(FROBENIUS_COEFF_FQ12_C1[i].pow([j as u64 + 1])),
                "FROBENIUS_COEFFS[{}][{}] failed",
                i,
                j
            )
        }
    }
}

#[test]
fn test_bn254_frobenius() {
    let mut rng = StdRng::seed_from_u64(15);
    for pow in 0..12 {
        let fq = Fq12::random(&mut rng);
        let fq_frob = fq.frobenius_map(pow);

        let fp = convert_bn254_halo2_fq12_to_fp12(fq);
        let fp_frob = fp.frobenius_map(pow);

        assert_eq!(fp_frob, convert_bn254_halo2_fq12_to_fp12(fq_frob));
    }
}

#[test]
fn test_fp12_invert() {
    let mut rng = StdRng::seed_from_u64(15);
    let fq = Fq12::random(&mut rng);
    let fq_inv = fq.invert().unwrap();

    let fp = convert_bn254_halo2_fq12_to_fp12(fq);
    let fp_inv = fp.invert();
    assert_eq!(fp_inv, convert_bn254_halo2_fq12_to_fp12(fq_inv));
}

#[test]
fn test_fp6_invert() {
    let mut rng = StdRng::seed_from_u64(20);
    let fq6 = Fq6::random(&mut rng);
    let fq6_inv = fq6.invert().unwrap();

    let fp6c0 = convert_bn254_halo2_fq2_to_fp2(fq6.c0);
    let fp6c1 = convert_bn254_halo2_fq2_to_fp2(fq6.c1);
    let fp6c2 = convert_bn254_halo2_fq2_to_fp2(fq6.c2);
    let mut fp6 = [fp6c0, fp6c1, fp6c2];
    fp6_invert_assign::<Fp, Fp2>(&mut fp6, &Bn254::XI);

    let fq6_invc0 = convert_bn254_halo2_fq2_to_fp2(fq6_inv.c0);
    let fq6_invc1 = convert_bn254_halo2_fq2_to_fp2(fq6_inv.c1);
    let fq6_invc2 = convert_bn254_halo2_fq2_to_fp2(fq6_inv.c2);
    let fq6_inv = [fq6_invc0, fq6_invc1, fq6_invc2];
    assert_eq!(fp6, fq6_inv);
}

#[test]
fn test_fp2_invert() {
    let mut rng = StdRng::seed_from_u64(25);
    let fq2 = Fq2::random(&mut rng);
    let fq2_inv = fq2.invert().unwrap();

    let mut fp2 = convert_bn254_halo2_fq2_to_fp2(fq2).to_coeffs();
    fp2_invert_assign::<Fp>(&mut fp2);
    assert_eq!(fp2, convert_bn254_halo2_fq2_to_fp2(fq2_inv).to_coeffs());
}

#[test]
fn test_fp6_square() {
    let mut rng = StdRng::seed_from_u64(45);
    let fq6 = Fq6::random(&mut rng);
    let fq6_sq = fq6.square();

    let fp6c0 = convert_bn254_halo2_fq2_to_fp2(fq6.c0);
    let fp6c1 = convert_bn254_halo2_fq2_to_fp2(fq6.c1);
    let fp6c2 = convert_bn254_halo2_fq2_to_fp2(fq6.c2);
    let mut fp6 = [fp6c0, fp6c1, fp6c2];
    fp6_square_assign::<Fp, Fp2>(&mut fp6, &Bn254::XI);

    let fq6_sqc0 = convert_bn254_halo2_fq2_to_fp2(fq6_sq.c0);
    let fq6_sqc1 = convert_bn254_halo2_fq2_to_fp2(fq6_sq.c1);
    let fq6_sqc2 = convert_bn254_halo2_fq2_to_fp2(fq6_sq.c2);
    let fq6_sq = [fq6_sqc0, fq6_sqc1, fq6_sqc2];
    assert_eq!(fp6, fq6_sq);
}

#[test]
fn test_fp2_square() {
    let mut rng = StdRng::seed_from_u64(55);
    let fq2 = Fq2::random(&mut rng);
    let fq2_sq = fq2.square();

    let fp2 = convert_bn254_halo2_fq2_to_fp2(fq2);
    let fp2_sq = &fp2 * &fp2;
    assert_eq!(fp2_sq, convert_bn254_halo2_fq2_to_fp2(fq2_sq));
}

#[test]
fn test_fp_add() {
    let mut rng = StdRng::seed_from_u64(65);
    let fq = Fq::random(&mut rng);
    let fq_res = fq + Fq::one();

    let fp = convert_bn254_halo2_fq_to_fp(fq);
    let fp_res = fp + Fp::ONE;
    assert_eq!(fp_res, convert_bn254_halo2_fq_to_fp(fq_res));
}

#[test]
fn test_fp_one() {
    let fp_one = Fp::ONE;
    let fq_one = Fq::ONE;
    assert_eq!(fp_one, convert_bn254_halo2_fq_to_fp(fq_one));
}

#[test]
fn test_bn254_miller_loop() {
    let mut rng = StdRng::seed_from_u64(53);
    let h2c_p = G1Affine::random(&mut rng);
    let h2c_q = G2Affine::random(&mut rng);

    let p = AffinePoint {
        x: convert_bn254_halo2_fq_to_fp(h2c_p.x),
        y: convert_bn254_halo2_fq_to_fp(h2c_p.y),
    };
    let q = AffinePoint {
        x: convert_bn254_halo2_fq2_to_fp2(h2c_q.x),
        y: convert_bn254_halo2_fq2_to_fp2(h2c_q.y),
    };

    // Compare against halo2curves implementation
    let h2c_q_prepared = G2Prepared::from(h2c_q);
    let compare_miller = halo2curves_axiom::bn256::multi_miller_loop(&[(&h2c_p, &h2c_q_prepared)]);
    let compare_final = compare_miller.final_exponentiation();

    let f = Bn254::multi_miller_loop(&[p], &[q]);
    let f_fq12 = convert_bn254_fp12_to_halo2_fq12(f);
    let wrapped_f = Gt(f_fq12);
    let final_f = wrapped_f.final_exponentiation();

    assert_eq!(final_f, compare_final);
}

// #[test]
// #[allow(non_snake_case)]
// fn test_bn254_final_exp_hint() {
//     let S = G1Affine::generator();
//     let Q = G2Affine::generator();

//     let S_mul = [S * Fr::from(1), S * -Fr::from(2)];
//     let Q_mul = [Q * Fr::from(2), Q * Fr::from(1)];

//     let s = S_mul.map(|s| AffinePoint::new(s.x, s.y));
//     let q = Q_mul.map(|p| AffinePoint::new(p.x, p.y));

//     let ps = [
//         convert_bn254_affine_point_halo2_fq_to_fp(s[0].clone()),
//         convert_bn254_affine_point_halo2_fq_to_fp(s[1].clone()),
//     ];
//     let qs = [
//         convert_bn254_affine_point_halo2_fq2_to_fp2(q[0].clone()),
//         convert_bn254_affine_point_halo2_fq2_to_fp2(q[1].clone()),
//     ];

//     let f = Bn254::multi_miller_loop(&ps, &qs);
//     let (c, u) = Bn254::final_exp_hint(&f);

//     let q = BigUint::from_str_radix(
//         "21888242871839275222246405745257275088696311157297823662689037894645226208583",
//         10,
//     )
//     .unwrap();
//     let six_x_plus_2: BigUint = BigUint::from_str_radix("29793968203157093288", 10).unwrap();
//     let q_pows = q.clone().pow(3) - q.clone().pow(2) + q;
//     let lambda = six_x_plus_2.clone() + q_pows.clone();

//     let c_lambda = c.exp_bytes(true, &lambda.to_bytes_be());
//     assert_eq!(f * u, c_lambda);
// }

// #[test]
// #[allow(non_snake_case)]
// fn test_bn254_pairing_check() {
//     let S = G1Affine::generator();
//     let Q = G2Affine::generator();

//     let S_mul = [S * Fr::from(1), S * -Fr::from(2)];
//     let Q_mul = [Q * Fr::from(2), Q * Fr::from(1)];

//     let s = S_mul.map(|s| AffinePoint::new(s.x, s.y));
//     let q = Q_mul.map(|p| AffinePoint::new(p.x, p.y));

//     let ps = [
//         convert_bn254_affine_point_halo2_fq_to_fp(s[0].clone()),
//         convert_bn254_affine_point_halo2_fq_to_fp(s[1].clone()),
//     ];
//     let qs = [
//         convert_bn254_affine_point_halo2_fq2_to_fp2(q[0].clone()),
//         convert_bn254_affine_point_halo2_fq2_to_fp2(q[1].clone()),
//     ];

//     assert!(Bn254::pairing_check(&ps, &qs).is_ok());
// }
