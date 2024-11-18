use axvm_algebra::field::FieldExtension;
use halo2curves_axiom::bn256::{Fq, Fq12, Fq2, G1Affine, G2Affine};
use rand::Rng;

use crate::{
    pairing::{EvaluatedLine, FromLineDType},
    AffineCoords,
};

impl FromLineDType<Fq2> for Fq12 {
    fn from_evaluated_line_d_type(line: EvaluatedLine<Fq2>) -> Fq12 {
        FieldExtension::<Fq2>::from_coeffs([
            Fq2::one(),
            line.b,
            Fq2::zero(),
            line.c,
            Fq2::zero(),
            Fq2::zero(),
        ])
    }
}

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
