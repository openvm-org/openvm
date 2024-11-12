mod fq12;
mod fq2;
mod point;

use halo2curves_axiom::bls12_381::Fq6;
pub use halo2curves_axiom::bls12_381::{Fq, Fq12, Fq2};
pub use point::{G1Affine, G2Affine};

use crate::field::Field;

impl Field for Fq {
    type SelfRef<'a> = Self;

    const ZERO: Self = Fq::zero();
    const ONE: Self = Fq::one();

    fn invert(&self) -> Option<Self> {
        self.invert().into()
    }

    fn square(&self) -> Self {
        self.square()
    }
}

impl Field for Fq2 {
    type SelfRef<'a> = &'a Self;

    const ZERO: Self = Fq2 {
        c0: Fq::zero(),
        c1: Fq::zero(),
    };

    const ONE: Self = Fq2 {
        c0: Fq::one(),
        c1: Fq::zero(),
    };

    fn invert(&self) -> Option<Self> {
        self.invert().into()
    }

    fn square(&self) -> Self {
        self.square()
    }
}

impl Field for Fq12 {
    type SelfRef<'a> = &'a Self;

    const ZERO: Self = Fq12 {
        c0: Fq6::zero(),
        c1: Fq6::zero(),
    };

    const ONE: Self = Fq12 {
        c0: Fq6::one(),
        c1: Fq6::zero(),
    };

    fn invert(&self) -> Option<Self> {
        self.invert().into()
    }

    fn square(&self) -> Self {
        self.square()
    }
}
