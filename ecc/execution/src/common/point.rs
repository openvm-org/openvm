use halo2curves_axiom::ff::Field;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct EcPoint<F> {
    pub x: F,
    pub y: F,
}

impl<F: Field> EcPoint<F> {
    pub fn new(x: F, y: F) -> Self {
        Self { x, y }
    }

    pub fn neg(&self) -> Self {
        Self {
            x: self.x,
            y: self.y.neg(),
        }
    }
}

pub trait AffineCoords<F: Field>: Clone {
    fn x(&self) -> F;
    fn y(&self) -> F;
    fn random(rng: &mut impl Rng) -> Self;
    fn generator() -> Self;
}

pub trait ScalarMul<Fr: Field> {
    fn scalar_mul(&self, s: Fr) -> Self;
}
