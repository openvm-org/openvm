use group::{ff::Field, prime::PrimeCurveAffine};
use halo2curves_axiom::bn256::{
    Fq, Fq12, Fq2, Fq6, G1Affine, G2Affine, G2Prepared, Gt, FROBENIUS_COEFF_FQ12_C1,
    FROBENIUS_COEFF_FQ6_C1, XI_TO_Q_MINUS_1_OVER_2,
};
use num_bigint::BigUint;
use num_traits::One;
use openvm_algebra_guest::{field::FieldExtension, IntMod};
use openvm_ecc_guest::{weierstrass::WeierstrassPoint, AffinePoint};
use openvm_pairing_guest::{
    bn254::{BN254_MODULUS, BN254_ORDER},
    pairing::{FinalExp, MultiMillerLoop, PairingCheck, PairingIntrinsics},
};
use rand::{rngs::StdRng, SeedableRng};

use super::{Fp, Fp12, Fp2};
use crate::{
    bn254::{
        utils::{
            convert_bn254_fp12_to_halo2_fq12, convert_bn254_halo2_fq12_to_fp12,
            convert_bn254_halo2_fq2_to_fp2, convert_bn254_halo2_fq_to_fp,
            convert_g2_affine_halo2_to_openvm,
        },
        Bn254, G2Affine as OpenVmG2Affine,
    },
    operations::{fp2_invert_assign, fp6_invert_assign, fp6_square_assign},
};

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

// Gt(Fq12) is not public
fn assert_miller_results_eq(a: Gt, b: Fp12) {
    let b = convert_bn254_fp12_to_halo2_fq12(b);
    openvm_pairing_guest::halo2curves_shims::bn254::test_utils::assert_miller_results_eq(a, b);
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
    let f = Bn254::multi_miller_loop(&[p], &[q]);
    assert_miller_results_eq(compare_miller, f);
}

#[test]
fn test_bn254_miller_loop_identity() {
    let mut rng = StdRng::seed_from_u64(33);
    let h2c_p = G1Affine::identity();
    let h2c_q = G2Affine::random(&mut rng);

    let p = AffinePoint {
        x: convert_bn254_halo2_fq_to_fp(Fq::ZERO),
        y: convert_bn254_halo2_fq_to_fp(Fq::ZERO),
    };
    let q = AffinePoint {
        x: convert_bn254_halo2_fq2_to_fp2(h2c_q.x),
        y: convert_bn254_halo2_fq2_to_fp2(h2c_q.y),
    };

    let f = Bn254::multi_miller_loop(&[p], &[q]);
    // halo2curves implementation
    let h2c_q_prepared = G2Prepared::from(h2c_q);
    let compare_miller = halo2curves_axiom::bn256::multi_miller_loop(&[(&h2c_p, &h2c_q_prepared)]);
    assert_miller_results_eq(compare_miller, f);
}

#[test]
fn test_bn254_miller_loop_identity_2() {
    let mut rng = StdRng::seed_from_u64(33);
    let h2c_p = G1Affine::random(&mut rng);
    let h2c_q = G2Affine::identity();
    let p = AffinePoint {
        x: convert_bn254_halo2_fq_to_fp(h2c_p.x),
        y: convert_bn254_halo2_fq_to_fp(h2c_p.y),
    };
    let q = AffinePoint {
        x: convert_bn254_halo2_fq2_to_fp2(Fq2::ZERO),
        y: convert_bn254_halo2_fq2_to_fp2(Fq2::ZERO),
    };

    let f = Bn254::multi_miller_loop(&[p], &[q]);
    // halo2curves implementation
    let h2c_q_prepared = G2Prepared::from(h2c_q);
    let compare_miller = halo2curves_axiom::bn256::multi_miller_loop(&[(&h2c_p, &h2c_q_prepared)]);
    assert_miller_results_eq(compare_miller, f);
}

// test on host is enough since we are testing the curve formulas and not anything
// about intrinsic functions
#[test]
fn test_bn254_g2_affine() {
    let mut rng = StdRng::seed_from_u64(34);
    for _ in 0..10 {
        let p = G2Affine::random(&mut rng);
        let q = G2Affine::random(&mut rng);
        let expected_add = G2Affine::from(p + q);
        let expected_sub = G2Affine::from(p - q);
        let expected_neg = -p;
        let expected_double = G2Affine::from(p + p);
        let [p, q] = [p, q].map(|p| {
            let x = convert_bn254_halo2_fq2_to_fp2(p.x);
            let y = convert_bn254_halo2_fq2_to_fp2(p.y);
            // check on curve
            OpenVmG2Affine::from_xy(x, y).unwrap()
        });
        let r_add = &p + &q;
        let r_sub = &p - &q;
        let r_neg = -&p;
        let r_double = &p + &p;

        for (expected, actual) in [
            (expected_add, r_add),
            (expected_sub, r_sub),
            (expected_neg, r_neg),
            (expected_double, r_double),
        ] {
            assert_eq!(convert_g2_affine_halo2_to_openvm(expected), actual);
        }
    }
}

#[test]
fn test_bn254_pairing_check_hint_host() {
    let mut rng = StdRng::seed_from_u64(83);
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

    let (c, u) = Bn254::pairing_check_hint(&[p], &[q]);

    let p_cmp = AffinePoint {
        x: h2c_p.x,
        y: h2c_p.y,
    };
    let q_cmp = AffinePoint {
        x: h2c_q.x,
        y: h2c_q.y,
    };

    let f_cmp = openvm_pairing_guest::halo2curves_shims::bn254::Bn254::multi_miller_loop(
        &[p_cmp],
        &[q_cmp],
    );
    let (c_cmp, u_cmp) =
        openvm_pairing_guest::halo2curves_shims::bn254::Bn254::final_exp_hint(&f_cmp);
    let c_cmp = convert_bn254_halo2_fq12_to_fp12(c_cmp);
    let u_cmp = convert_bn254_halo2_fq12_to_fp12(u_cmp);

    assert_eq!(c, c_cmp);
    assert_eq!(u, u_cmp);
}

#[test]
fn test_bn254_final_exponent() {
    let final_exp = (BN254_MODULUS.pow(12) - BigUint::one()) / BN254_ORDER.clone();
    assert_eq!(Bn254::FINAL_EXPONENT.to_vec(), final_exp.to_bytes_be());
}

#[test]
fn test_bn254_frobenius_coeffs_fq6() {
    #[allow(clippy::needless_range_loop)]
    for i in 0..3 {
        assert_eq!(
            Bn254::FROBENIUS_COEFF_FQ6_C1[i],
            convert_bn254_halo2_fq2_to_fp2(FROBENIUS_COEFF_FQ6_C1[i]),
            "FROBENIUS_COEFFS_FQ6_C1[{}] failed",
            i,
        )
    }
}

#[test]
fn test_bn254_xi_to_q_minus_1_over_2() {
    assert_eq!(
        Bn254::XI_TO_Q_MINUS_1_OVER_2,
        convert_bn254_halo2_fq2_to_fp2(XI_TO_Q_MINUS_1_OVER_2),
        "XI_TO_Q_MINUS_1_OVER_2 failed",
    )
}
