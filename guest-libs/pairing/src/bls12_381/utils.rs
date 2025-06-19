use halo2curves_axiom::bls12_381::{Fq, Fq12, Fq2, G2Affine};
use openvm_algebra_guest::{field::FieldExtension, IntMod};
use openvm_ecc_guest::weierstrass::WeierstrassPoint;

use super::{Fp, Fp12, Fp2};
use crate::bls12_381::G2Affine as OpenVmG2Affine;

pub(crate) fn convert_bls12381_halo2_fq_to_fp(x: Fq) -> Fp {
    let bytes = x.to_bytes();
    Fp::from_le_bytes_unchecked(&bytes)
}

pub(crate) fn convert_bls12381_halo2_fq2_to_fp2(x: Fq2) -> Fp2 {
    Fp2::new(
        convert_bls12381_halo2_fq_to_fp(x.c0),
        convert_bls12381_halo2_fq_to_fp(x.c1),
    )
}

pub(crate) fn convert_bls12381_halo2_fq12_to_fp12(x: Fq12) -> Fp12 {
    Fp12 {
        c: x.to_coeffs().map(convert_bls12381_halo2_fq2_to_fp2),
    }
}

pub(crate) fn convert_bls12381_fp_to_halo2_fq(x: Fp) -> Fq {
    Fq::from_bytes(&x.0).unwrap()
}

pub(crate) fn convert_bls12381_fp2_to_halo2_fq2(x: Fp2) -> Fq2 {
    Fq2 {
        c0: convert_bls12381_fp_to_halo2_fq(x.c0.clone()),
        c1: convert_bls12381_fp_to_halo2_fq(x.c1.clone()),
    }
}

#[allow(unused)]
pub(crate) fn convert_bls12381_fp12_to_halo2_fq12(x: Fp12) -> Fq12 {
    let c = x.to_coeffs();
    Fq12::from_coeffs(c.map(convert_bls12381_fp2_to_halo2_fq2))
}

#[allow(unused)]
pub(crate) fn convert_g2_affine_halo2_to_openvm(p: G2Affine) -> OpenVmG2Affine {
    OpenVmG2Affine::from_xy_unchecked(
        convert_bls12381_halo2_fq2_to_fp2(p.x),
        convert_bls12381_halo2_fq2_to_fp2(p.y),
    )
}
