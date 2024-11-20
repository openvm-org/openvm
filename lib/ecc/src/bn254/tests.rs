use axvm_algebra::{field::FieldExtension, IntMod};
use group::ff::Field;
use halo2curves_axiom::bn256::{Fq, Fq12, Fq2};
use rand::{rngs::StdRng, SeedableRng};

use super::{Fp, Fp12, Fp2};
use crate::bn254::fp2_invert_assign;

fn convert_fq_to_bn254fp(x: Fq) -> Fp {
    let bytes = x.0.map(|y| y.to_le_bytes()).concat();
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
fn test_fp12_invert() {
    let mut rng = StdRng::seed_from_u64(15);
    let fq = Fq12::random(&mut rng);
    let fq_inv = fq.invert().unwrap();

    let fp = convert_fq12_to_bn254fp12(fq);
    let fp_inv = fp.invert();
    assert_eq!(fp_inv, convert_fq12_to_bn254fp12(fq_inv));
}

#[test]
fn test_fp2_invert() {
    let mut rng = StdRng::seed_from_u64(25);
    // let fq2 = Fq2::random(&mut rng);
    let fq2 = Fq2::one();
    let fq2_inv = fq2.invert().unwrap();

    let mut fp2 = convert_fq2_to_bn254fp2(fq2);
    fp2_invert_assign(&mut fp2);
    assert_eq!(fp2, convert_fq2_to_bn254fp2(fq2_inv));
}
