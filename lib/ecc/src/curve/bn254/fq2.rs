pub use halo2curves_axiom::bn256::{Fq, Fq2};

use crate::field::{Field, FieldExtension};

/// FieldExtension for Fq2 with Fq as base field
impl FieldExtension for Fq2 {
    type BaseField = Fq;
    type Coeffs = [Self::BaseField; 2];
    type SelfRef<'a> = &'a Self;

    fn from_coeffs(coeffs: Self::Coeffs) -> Self {
        Fq2 {
            c0: coeffs[0],
            c1: coeffs[1],
        }
    }

    fn embed(base_elem: Self::BaseField) -> Self {
        Fq2 {
            c0: base_elem,
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

    fn mul_base(&self, rhs: Self::BaseField) -> Self {
        Fq2 {
            c0: self.c0 * rhs,
            c1: self.c1 * rhs,
        }
    }
}
