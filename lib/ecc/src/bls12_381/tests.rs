use axvm_algebra::{field::FieldExtension, IntMod};
use group::ff::Field;
use halo2curves_axiom::bls12_381::{Fq, Fq12, Fq2, Fq6};
use rand::{rngs::StdRng, SeedableRng};

use super::{Fp, Fp12, Fp2};
use crate::{
    bls12_381::Bls12_381,
    pairing::{fp2_invert_assign, fp6_invert_assign, fp6_square_assign, PairingIntrinsics},
};

fn convert_bls12381_halo2_fq_to_fp(x: Fq) -> Fp {
    let bytes = x.to_bytes();
    Fp::from_le_bytes(&bytes)
}

fn convert_bls12381_halo2_fq2_to_fp2(x: Fq2) -> Fp2 {
    Fp2::new(
        convert_bls12381_halo2_fq_to_fp(x.c0),
        convert_bls12381_halo2_fq_to_fp(x.c1),
    )
}

fn convert_bls12381_halo2_fq12_to_fp12(x: Fq12) -> Fp12 {
    Fp12 {
        c: [
            convert_bls12381_halo2_fq2_to_fp2(x.c0.c0),
            convert_bls12381_halo2_fq2_to_fp2(x.c0.c1),
            convert_bls12381_halo2_fq2_to_fp2(x.c0.c2),
            convert_bls12381_halo2_fq2_to_fp2(x.c1.c0),
            convert_bls12381_halo2_fq2_to_fp2(x.c1.c1),
            convert_bls12381_halo2_fq2_to_fp2(x.c1.c2),
        ],
    }
}

#[test]
fn test_bls12381_frobenius() {
    const MODULUS: [u64; 6] = [
        0xb9fe_ffff_ffff_aaab,
        0x1eab_fffe_b153_ffff,
        0x6730_d2a0_f6b0_f624,
        0x6477_4b84_f385_12bf,
        0x4b1b_a7b6_434b_acd7,
        0x1a01_11ea_397f_e69a,
    ];

    let mut rng = StdRng::seed_from_u64(15);
    let pow = 2;
    let fq = Fq12::random(&mut rng);
    // let fq_frob = fq.frobenius_map().pow([pow as u64]);
    let fq_frob = fq.pow_vartime(MODULUS).pow_vartime([pow as u64]);

    let fp = convert_bls12381_halo2_fq12_to_fp12(fq);
    let fp_frob = fp.frobenius_map(pow);

    assert_eq!(fp_frob, convert_bls12381_halo2_fq12_to_fp12(fq_frob));
}

#[test]
fn test_fp12_invert() {
    let mut rng = StdRng::seed_from_u64(15);
    let fq = Fq12::random(&mut rng);
    let fq_inv = fq.invert().unwrap();

    let fp = convert_bls12381_halo2_fq12_to_fp12(fq);
    let fp_inv = fp.invert();
    assert_eq!(fp_inv, convert_bls12381_halo2_fq12_to_fp12(fq_inv));
}

#[test]
fn test_fp6_invert() {
    let mut rng = StdRng::seed_from_u64(20);
    let fq6 = Fq6 {
        c0: Fq2::random(&mut rng),
        c1: Fq2::random(&mut rng),
        c2: Fq2::random(&mut rng),
    };
    let fq6_inv = fq6.invert().unwrap();

    let fp6c0 = convert_bls12381_halo2_fq2_to_fp2(fq6.c0);
    let fp6c1 = convert_bls12381_halo2_fq2_to_fp2(fq6.c1);
    let fp6c2 = convert_bls12381_halo2_fq2_to_fp2(fq6.c2);
    let mut fp6 = [fp6c0, fp6c1, fp6c2];
    fp6_invert_assign::<Fp, Fp2>(&mut fp6, &Bls12_381::XI);

    let fq6_invc0 = convert_bls12381_halo2_fq2_to_fp2(fq6_inv.c0);
    let fq6_invc1 = convert_bls12381_halo2_fq2_to_fp2(fq6_inv.c1);
    let fq6_invc2 = convert_bls12381_halo2_fq2_to_fp2(fq6_inv.c2);
    let fq6_inv = [fq6_invc0, fq6_invc1, fq6_invc2];
    assert_eq!(fp6, fq6_inv);
}

#[test]
fn test_fp2_invert() {
    let mut rng = StdRng::seed_from_u64(25);
    let fq2 = Fq2::random(&mut rng);
    let fq2_inv = fq2.invert().unwrap();

    let mut fp2 = convert_bls12381_halo2_fq2_to_fp2(fq2).to_coeffs();
    fp2_invert_assign::<Fp>(&mut fp2);
    assert_eq!(fp2, convert_bls12381_halo2_fq2_to_fp2(fq2_inv).to_coeffs());
}

#[test]
fn test_fp6_square() {
    let mut rng = StdRng::seed_from_u64(45);
    let fq6 = Fq6 {
        c0: Fq2::random(&mut rng),
        c1: Fq2::random(&mut rng),
        c2: Fq2::random(&mut rng),
    };
    let fq6_sq = fq6.square();

    let fp6c0 = convert_bls12381_halo2_fq2_to_fp2(fq6.c0);
    let fp6c1 = convert_bls12381_halo2_fq2_to_fp2(fq6.c1);
    let fp6c2 = convert_bls12381_halo2_fq2_to_fp2(fq6.c2);
    let mut fp6 = [fp6c0, fp6c1, fp6c2];
    fp6_square_assign::<Fp, Fp2>(&mut fp6, &Bls12_381::XI);

    let fq6_sqc0 = convert_bls12381_halo2_fq2_to_fp2(fq6_sq.c0);
    let fq6_sqc1 = convert_bls12381_halo2_fq2_to_fp2(fq6_sq.c1);
    let fq6_sqc2 = convert_bls12381_halo2_fq2_to_fp2(fq6_sq.c2);
    let fq6_sq = [fq6_sqc0, fq6_sqc1, fq6_sqc2];
    assert_eq!(fp6, fq6_sq);
}

#[test]
fn test_fp2_square() {
    let mut rng = StdRng::seed_from_u64(55);
    let fq2 = Fq2::random(&mut rng);
    let fq2_sq = fq2.square();

    let fp2 = convert_bls12381_halo2_fq2_to_fp2(fq2);
    let fp2_sq = &fp2 * &fp2;
    assert_eq!(fp2_sq, convert_bls12381_halo2_fq2_to_fp2(fq2_sq));
}

#[test]
fn test_fp_add() {
    let mut rng = StdRng::seed_from_u64(65);
    let fq = Fq::random(&mut rng);
    let fq_res = fq + Fq::one();

    let fp = convert_bls12381_halo2_fq_to_fp(fq);
    let fp_res = fp + Fp::ONE;
    assert_eq!(fp_res, convert_bls12381_halo2_fq_to_fp(fq_res));
}

#[test]
fn test_fp_one() {
    let fp_one = Fp::ONE;
    let fq_one = Fq::ONE;
    assert_eq!(fp_one, convert_bls12381_halo2_fq_to_fp(fq_one));
}
