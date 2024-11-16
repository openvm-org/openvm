mod fq12;
mod fq2;
mod point;

use halo2curves_axiom::bls12_381::Fq6;
pub use halo2curves_axiom::bls12_381::{Fq, Fq12, Fq2};
pub use point::{G1Affine, G2Affine};

use crate::field::Field;

impl Field for Fq {
    type SelfRef<'a> = Self;

    fn zero() -> Self {
        Fq::zero()
    }

    fn one() -> Self {
        Fq::one()
    }

    fn invert(&self) -> Option<Self> {
        self.invert().into()
    }
}

impl Field for Fq2 {
    type SelfRef<'a> = &'a Self;

    fn zero() -> Self {
        Fq2 {
            c0: Fq::zero(),
            c1: Fq::zero(),
        }
    }

    fn one() -> Self {
        Fq2 {
            c0: Fq::one(),
            c1: Fq::zero(),
        }
    }

    fn invert(&self) -> Option<Self> {
        self.invert().into()
    }
}

impl Field for Fq12 {
    type SelfRef<'a> = &'a Self;

    fn zero() -> Self {
        Fq12 {
            c0: Fq6::zero(),
            c1: Fq6::zero(),
        }
    }

    fn one() -> Self {
        Fq12 {
            c0: Fq6::one(),
            c1: Fq6::zero(),
        }
    }

    fn invert(&self) -> Option<Self> {
        self.invert().into()
    }
}
