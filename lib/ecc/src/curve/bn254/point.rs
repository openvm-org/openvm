use core::ops::{Mul, MulAssign};

pub use halo2curves_axiom::{
    bn256::{Fq, Fq2, Fr, G1Affine as Bn254G1Affine, G2Affine as Bn254G2Affine},
    group::prime::PrimeCurveAffine,
};
use rand::Rng;

use crate::point::AffineCoords;

#[derive(Clone)]
pub struct G1Affine(Bn254G1Affine);

impl G1Affine {
    pub fn inner(&self) -> &Bn254G1Affine {
        &self.0
    }
}

impl AffineCoords<Fq> for G1Affine {
    fn x(&self) -> Fq {
        self.0.x
    }

    fn y(&self) -> Fq {
        self.0.y
    }

    fn neg(&self) -> Self {
        let mut pt = self.0;
        pt.y = -pt.y;
        G1Affine(pt)
    }

    fn random(rng: &mut impl Rng) -> Self {
        G1Affine(Bn254G1Affine::random(rng))
    }

    fn generator() -> Self {
        G1Affine(Bn254G1Affine::generator())
    }
}

impl Mul<Fr> for G1Affine {
    type Output = Self;

    fn mul(self, s: Fr) -> Self::Output {
        G1Affine((self.0.mul(s)).into())
    }
}

impl MulAssign<Fr> for G1Affine {
    fn mul_assign(&mut self, s: Fr) {
        self.0.to_curve().mul_assign(s);
    }
}

#[derive(Clone)]
pub struct G2Affine(Bn254G2Affine);

impl G2Affine {
    pub fn inner(&self) -> &Bn254G2Affine {
        &self.0
    }
}

impl AffineCoords<Fq2> for G2Affine {
    fn x(&self) -> Fq2 {
        self.0.x
    }

    fn y(&self) -> Fq2 {
        self.0.y
    }

    fn neg(&self) -> Self {
        let mut pt = self.0;
        pt.y = -pt.y;
        G2Affine(pt)
    }

    fn random(rng: &mut impl Rng) -> Self {
        G2Affine(Bn254G2Affine::random(rng))
    }

    fn generator() -> Self {
        G2Affine(Bn254G2Affine::generator())
    }
}

impl Mul<Fr> for G2Affine {
    type Output = Self;

    fn mul(self, s: Fr) -> Self::Output {
        G2Affine((self.0.mul(s)).into())
    }
}

impl MulAssign<Fr> for G2Affine {
    fn mul_assign(&mut self, s: Fr) {
        self.0.to_curve().mul_assign(s);
    }
}
