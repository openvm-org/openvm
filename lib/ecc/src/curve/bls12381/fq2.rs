use alloc::vec::Vec;

use halo2curves::{
    bls12381::{Fq, Fq12, Fq2, Fq6},
    ff::Field,
};

use crate::{
    field::{ExpBigInt, FieldExtension, Fp12Constructor, Fp2Constructor},
    pairing::{EvaluatedLine, LineMType},
};

impl Fp2Constructor<Fq> for Fq2 {
    fn new(c0: Fq, c1: Fq) -> Self {
        let mut b = [0u8; 48 * 2];
        b[..48].copy_from_slice(&c0.to_bytes());
        b[48..].copy_from_slice(&c1.to_bytes());
        Fq2::from_bytes(&b).unwrap()
    }
}

/// FieldExtension for Fq2 with Fq as base field
impl FieldExtension for Fq2 {
    type BaseField = Fq;

    fn from_coeffs(coeffs: &[Self::BaseField]) -> Self {
        assert!(coeffs.len() <= 2, "coeffs must have at most 2 elements");
        let mut coeffs = coeffs.to_vec();
        coeffs.resize(2, Self::BaseField::ZERO);

        Fq2 {
            c0: coeffs[0],
            c1: coeffs[1],
        }
    }

    fn embed(base_elem: &Self::BaseField) -> Self {
        Fq2 {
            c0: *base_elem,
            c1: Fq::ZERO,
        }
    }

    fn conjugate(&self) -> Self {
        let mut res = *self;
        res.conjugate();
        res
    }

    fn frobenius_map(&self, _power: Option<usize>) -> Self {
        Fq2::frobenius_map(&self.0)
    }

    fn mul_base(&self, rhs: &Self::BaseField) -> Self {
        Fq2 {
            c0: self.0.c0 * rhs,
            c1: self.0.c1 * rhs,
        }
    }
}
