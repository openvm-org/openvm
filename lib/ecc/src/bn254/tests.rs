use axvm_algebra::{field::FieldExtension, IntMod};
use group::ff::Field;
use halo2curves_axiom::bn256::{Fq, Fq12, Fq2, Fq6, FROBENIUS_COEFF_FQ12_C1};
use rand::{rngs::StdRng, SeedableRng};

use super::{Fp, Fp12, Fp2};
use crate::{
    bn254::Bn254,
    pairing::{fp2_invert_assign, fp6_invert_assign, fp6_square_assign, PairingIntrinsics},
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
fn test_bn254_frobenius() {
    let mut rng = StdRng::seed_from_u64(15);
    for pow in 0..4 {
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
