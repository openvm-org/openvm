pub use halo2curves_axiom::{
    bls12_381::{Fq, Fq2},
    ff::Field,
};

use crate::field::FieldExtension;

/// FieldExtension for Fq2 with Fq as base field
impl FieldExtension for Fq2 {
    type BaseField = Fq;
    type Coeffs = [Self::BaseField; 2];

    fn from_coeffs(coeffs: Self::Coeffs) -> Self {
        // TODO[yj]: conversion for PSE halo2curves implementation
        //     let mut b = [0u8; 48 * 2];
        //     b[..48].copy_from_slice(&c0.to_bytes());
        //     b[48..].copy_from_slice(&c1.to_bytes());
        //     Fq2::from_bytes(&b).unwrap()
        Fq2 {
            c0: coeffs[0],
            c1: coeffs[1],
        }
    }

    fn to_coeffs(self) -> Self::Coeffs {
        [self.c0, self.c1]
    }

    fn embed(base_elem: Self::BaseField) -> Self {
        Fq2 {
            c0: base_elem,
            c1: Fq::ZERO,
        }
    }

    fn conjugate(&self) -> Self {
        // TODO[yj]: conversion for PSE halo2curves implementation
        // let mut res = *self;
        // res.conjugate();
        // res
        Fq2::conjugate(self)
    }

    fn frobenius_map(&self, _power: Option<usize>) -> Self {
        Fq2::frobenius_map(self)
    }

    fn mul_base(&self, rhs: Self::BaseField) -> Self {
        Fq2 {
            c0: self.c0 * rhs,
            c1: self.c1 * rhs,
        }
    }
}
