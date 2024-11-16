use rand::Rng;

use crate::field::Field;

#[derive(Debug, Clone)]
#[repr(C)]
pub struct AffinePoint<F> {
    pub x: F,
    pub y: F,
}

impl<F: Field> AffinePoint<F> {
    pub fn new(x: F, y: F) -> Self {
        Self { x, y }
    }

    pub fn neg_assign(self) -> Self {
        Self {
            x: self.x,
            y: self.y.neg(),
        }
    }
}

pub trait AffineCoords<F: Field>: Clone {
    /// Returns the affine representation x-coordinate of the elliptic curve point.
    fn x(&self) -> F;

    /// Returns the affine representation y-coordinate of the elliptic curve point.
    fn y(&self) -> F;

    /// Negates the elliptic curve point (reflection on the x-axis).
    fn neg(&self) -> Self;

    /// Generates a random elliptic curve point.
    fn random(rng: &mut impl Rng) -> Self;

    /// Returns the generator point of the elliptic curve.
    fn generator() -> Self;
}
<<<<<<< HEAD:lib/ecc-execution/src/common/point.rs

pub trait ScalarMul<Fr: Field> {
    /// Scalar multiplication of an elliptic curve point by a scalar.
    fn scalar_mul(&self, s: Fr) -> Self;
}

pub trait EccBinOps<F: Field>: AffineCoords<F> {
    /// Adds two elliptic curve points.
    fn is_eq(&self, other: &Self) -> bool;

    /// Adds two elliptic curve points.
    fn add(&self, other: &Self) -> Self;

    /// Subtracts two elliptic curve points.
    fn sub(&self, other: &Self) -> Self {
        self.add(&other.neg())
    }

    /// Doubles an elliptic curve point.
    fn double(&self, other: &Self) -> Self;

    /// Adds two elliptic curve points.
    fn add_uneq(&self, other: &Self) -> Self;
}
=======
>>>>>>> fb24f40ddede44c91b4012eb499c84d7bbda7acc:lib/ecc/src/point/mod.rs
