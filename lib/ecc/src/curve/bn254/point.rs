use core::ops::Mul;

pub use halo2curves_axiom::{
    bn256::{Fq, Fq2, Fr, G1Affine, G2Affine},
    group::prime::PrimeCurveAffine,
};
use rand::Rng;

use crate::point::{AffineCoords, ScalarMul};

impl AffineCoords<Fq> for G1Affine {
    fn x(&self) -> Fq {
        self.x
    }

    fn y(&self) -> Fq {
        self.y
    }

    fn neg(&self) -> Self {
        let mut pt = *self;
        pt.y = -pt.y;
        pt
    }

    fn random(rng: &mut impl Rng) -> Self {
        G1Affine::random(rng)
    }

    fn generator() -> Self {
        G1Affine::generator()
    }
}

impl ScalarMul<Fr> for G1Affine {
    fn scalar_mul(&self, s: Fr) -> Self {
        (self.to_curve().mul(s)).into()
    }
}

impl AffineCoords<Fq2> for G2Affine {
    fn x(&self) -> Fq2 {
        self.x
    }

    fn y(&self) -> Fq2 {
        self.y
    }

    fn neg(&self) -> Self {
        let mut pt = *self;
        pt.y = -pt.y;
        pt
    }

    fn random(rng: &mut impl Rng) -> Self {
        G2Affine::random(rng)
    }

    fn generator() -> Self {
        G2Affine::generator()
    }
}

impl ScalarMul<Fr> for G2Affine {
    fn scalar_mul(&self, s: Fr) -> Self {
        (self.to_curve().mul(s)).into()
    }
}
