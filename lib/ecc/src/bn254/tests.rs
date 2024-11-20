use axvm_algebra::{field::FieldExtension, IntMod};
use group::ff::Field;
use halo2curves_axiom::bn256::{Fq, Fq12, Fq2, Fq6};
use rand::{rngs::StdRng, SeedableRng};

use super::{Fp, Fp12, Fp2};
use crate::{
    bn254::Bn254,
    pairing::{
        fp2_invert_assign, fp6_invert_assign, fp6_square_assign, fp_invert_assign,
        PairingIntrinsics,
    },
};

fn convert_fq_to_bn254fp(x: Fq) -> Fp {
    let bytes = x.to_bytes();
    Fp::from_le_bytes(&bytes)
}

fn convert_fq2_to_bn254fp2(x: Fq2) -> Fp2 {
    Fp2::new(convert_fq_to_bn254fp(x.c0), convert_fq_to_bn254fp(x.c1))
}

fn convert_fq12_to_bn254fp12(x: Fq12) -> Fp12 {
    Fp12 {
        c: [
            convert_fq2_to_bn254fp2(x.c0.c0),
            convert_fq2_to_bn254fp2(x.c0.c1),
            convert_fq2_to_bn254fp2(x.c0.c2),
            convert_fq2_to_bn254fp2(x.c1.c0),
            convert_fq2_to_bn254fp2(x.c1.c1),
            convert_fq2_to_bn254fp2(x.c1.c2),
        ],
    }
}

#[test]
fn test_bn254_frobenius() {
    let mut rng = StdRng::seed_from_u64(15);
    for pow in 0..4 {
        let fq = Fq12::random(&mut rng);
        let fq_frob = fq.frobenius_map(pow);

        let fp = convert_fq12_to_bn254fp12(fq);
        let fp_frob = fp.frobenius_map(pow);

        assert_eq!(fp_frob, convert_fq12_to_bn254fp12(fq_frob));
    }
}

#[test]
fn test_fp12_invert() {
    let mut rng = StdRng::seed_from_u64(15);
    let fq = Fq12::random(&mut rng);
    let fq_inv = fq.invert().unwrap();

    let fp = convert_fq12_to_bn254fp12(fq);
    let fp_inv = fp.invert();
    assert_eq!(fp_inv, convert_fq12_to_bn254fp12(fq_inv));
}

#[test]
fn test_fp6_invert() {
    let mut rng = StdRng::seed_from_u64(20);
    let fq6 = Fq6::random(&mut rng);
    let fq6_inv = fq6.invert().unwrap();

    let fp6c0 = convert_fq2_to_bn254fp2(fq6.c0);
    let fp6c1 = convert_fq2_to_bn254fp2(fq6.c1);
    let fp6c2 = convert_fq2_to_bn254fp2(fq6.c2);
    let mut fp6 = [fp6c0, fp6c1, fp6c2];
    fp6_invert_assign::<Fp, Fp2>(&mut fp6, &Bn254::XI);

    let fq6_invc0 = convert_fq2_to_bn254fp2(fq6_inv.c0);
    let fq6_invc1 = convert_fq2_to_bn254fp2(fq6_inv.c1);
    let fq6_invc2 = convert_fq2_to_bn254fp2(fq6_inv.c2);
    let fq6_inv = [fq6_invc0, fq6_invc1, fq6_invc2];
    assert_eq!(fp6, fq6_inv);
}

#[test]
fn test_fp2_invert() {
    let mut rng = StdRng::seed_from_u64(25);
    let fq2 = Fq2::random(&mut rng);
    let fq2_inv = fq2.invert().unwrap();

    let mut fp2 = convert_fq2_to_bn254fp2(fq2).to_coeffs();
    fp2_invert_assign::<Fp>(&mut fp2);
    assert_eq!(fp2, convert_fq2_to_bn254fp2(fq2_inv).to_coeffs());
}

#[test]
fn test_fp_invert() {
    let mut rng = StdRng::seed_from_u64(35);
    let fq = Fq::random(&mut rng);
    let fq_inv = fq.invert().unwrap();

    let mut fp = convert_fq_to_bn254fp(fq);
    fp_invert_assign::<Fp>(&mut fp);
    assert_eq!(fp, convert_fq_to_bn254fp(fq_inv));
}

#[test]
fn test_fp6_square() {
    let mut rng = StdRng::seed_from_u64(45);
    let fq6 = Fq6::random(&mut rng);
    let fq6_sq = fq6.square();

    let fp6c0 = convert_fq2_to_bn254fp2(fq6.c0);
    let fp6c1 = convert_fq2_to_bn254fp2(fq6.c1);
    let fp6c2 = convert_fq2_to_bn254fp2(fq6.c2);
    let mut fp6 = [fp6c0, fp6c1, fp6c2];
    fp6_square_assign::<Fp, Fp2>(&mut fp6, &Bn254::XI);

    let fq6_sqc0 = convert_fq2_to_bn254fp2(fq6_sq.c0);
    let fq6_sqc1 = convert_fq2_to_bn254fp2(fq6_sq.c1);
    let fq6_sqc2 = convert_fq2_to_bn254fp2(fq6_sq.c2);
    let fq6_sq = [fq6_sqc0, fq6_sqc1, fq6_sqc2];
    assert_eq!(fp6, fq6_sq);
}

#[test]
fn test_fp2_square() {
    let mut rng = StdRng::seed_from_u64(55);
    let fq2 = Fq2::random(&mut rng);
    let fq2_sq = fq2.square();

    let fp2 = convert_fq2_to_bn254fp2(fq2);
    let fp2_sq = &fp2 * &fp2;
    assert_eq!(fp2_sq, convert_fq2_to_bn254fp2(fq2_sq));
}

#[test]
fn test_fp_add() {
    let mut rng = StdRng::seed_from_u64(65);
    let fq = Fq::random(&mut rng);
    let fq_res = fq + Fq::one();

    let fp = convert_fq_to_bn254fp(fq);
    let fp_res = fp + Fp::ONE;
    assert_eq!(fp_res, convert_fq_to_bn254fp(fq_res));
}

#[test]
fn test_fp_one() {
    let fp_one = Fp::ONE;
    let fq_one = Fq::ONE;
    assert_eq!(fp_one, convert_fq_to_bn254fp(fq_one));
}
