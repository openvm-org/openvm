pub use halo2curves_axiom::{
    bn256::{Fq, Fq2},
    ff::Field,
};

use crate::field::{FieldExtension, Fp2Constructor};

impl Fp2Constructor<Fq> for Fq2 {
    fn new(c0: Fq, c1: Fq) -> Self {
        Fq2 { c0, c1 }
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
        let mut s = *self;
        Fq2::conjugate(&mut s);
        s
    }

    fn frobenius_map(&self, power: Option<usize>) -> Self {
        let mut s = *self;
        Fq2::frobenius_map(&mut s, power.unwrap());
        s
    }

    fn mul_base(&self, rhs: &Self::BaseField) -> Self {
        Fq2 {
            c0: self.c0 * rhs,
            c1: self.c1 * rhs,
        }
    }
}
