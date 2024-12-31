mod curve;
mod final_exp;
mod line;
mod miller_loop;

pub use curve::*;
pub use line::*;

#[cfg(test)]
pub mod tests;

use halo2curves_axiom::bls12_381::{Fq, Fq12, Fq2, G1Affine, G2Affine};
use openvm_algebra_guest::field::{Field, FieldExtension};
use rand::Rng;

use crate::{
    affine_point::AffineCoords,
    pairing::{Evaluatable, EvaluatedLine, FromLineMType, UnevaluatedLine},
};

impl FromLineMType<Fq2> for Fq12 {
    fn from_evaluated_line_m_type(line: EvaluatedLine<Fq2>) -> Fq12 {
        Fq12::from_coeffs([
            line.c,
            Fq2::zero(),
            line.b,
            Fq2::one(),
            Fq2::zero(),
            Fq2::zero(),
        ])
    }
}

impl Evaluatable<Fq, Fq2> for UnevaluatedLine<Fq2> {
    fn evaluate(&self, xy_frac: &(Fq, Fq)) -> EvaluatedLine<Fq2> {
        let (x_over_y, y_inv) = xy_frac;
        EvaluatedLine {
            b: self.b.mul_base(x_over_y),
            c: self.c.mul_base(y_inv),
        }
    }
}

impl AffineCoords<Fq> for G1Affine {
    fn new(x: Fq, y: Fq) -> Self {
        let mut m = G1Affine::identity();
        m.x = Fq::ONE * x;
        m.y = y;
        m
    }

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

    fn is_infinity(&self) -> bool {
        self.x == Fq::ZERO && self.y == Fq::ZERO
    }
}

impl AffineCoords<Fq2> for G2Affine {
    fn new(x: Fq2, y: Fq2) -> Self {
        let mut m = G2Affine::identity();
        m.x = Fq2::ONE * x;
        m.y = y;
        m
    }

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

    fn is_infinity(&self) -> bool {
        self.x == Fq2::ZERO && self.y == Fq2::ZERO
    }
}
