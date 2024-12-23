use core::ops::AddAssign;

use openvm_algebra_guest::{DivUnsafe, Field};
use openvm_ecc_guest::{AffinePoint, Group};

use super::{Fp, Fp2};

/// A newtype wrapper for [AffinePoint] that implements elliptic curve operations
/// by using the underlying field operations according to the [formulas](https://www.hyperelliptic.org/EFD/g1p/auto-shortw.html) for short Weierstrass curves.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[repr(transparent)]
pub struct G2Affine(pub AffinePoint<Fp2>);

const THREE: Fp2 = Fp2::new(Fp::from_const_u8(3), Fp::ZERO);

impl G2Affine {
    const IDENTITY: Self = Self(AffinePoint::new(Fp2::ZERO, Fp2::ZERO));

    pub fn into_inner(self) -> AffinePoint<Fp2> {
        self.0
    }

    pub const fn x(&self) -> &Fp2 {
        &self.0.x
    }

    pub const fn y(&self) -> &Fp2 {
        &self.0.y
    }

    fn double_nonidentity(&self) -> Self {
        // lambda = (3*x1^2+a)/(2*y1)
        // for bls12-381, a = 0
        let lambda = (&THREE * self.x() * self.x()).div_unsafe(self.y() + self.y());
        // x3 = lambda^2-x1-x1
        let x3 = &lambda * &lambda - self.x() - self.x();
        // y3 = lambda * (x1-x3) - y1
        let y3 = lambda * (self.x() - &x3) - self.y();
        Self(AffinePoint::new(x3, y3))
    }

    fn double_assign_nonidentity(&mut self) {
        // TODO: revisit if there are possible optimizations
        *self = self.double_nonidentity();
    }

    fn add_ne_nonidentity(&self, p2: &Self) -> Self {
        // lambda = (y2-y1)/(x2-x1)
        // x3 = lambda^2-x1-x2
        // y3 = lambda*(x1-x3)-y1
        let lambda = (p2.y() - self.y()).div_unsafe(p2.x() - self.x());
        let x3 = &lambda * &lambda - self.x() - p2.x();
        let y3 = lambda * (self.x() - &x3) - self.y();
        Self(AffinePoint::new(x3, y3))
    }

    fn add_ne_assign_nonidentity(&mut self, p2: &Self) {
        // TODO: revisit if there are possible optimizations
        *self = self.add_ne_nonidentity(p2);
    }

    fn sub_ne_nonidentity(&self, p2: &Self) -> Self {
        // lambda = (y2+y1)/(x1-x2)
        // x3 = lambda^2-x1-x2
        // y3 = lambda*(x1-x3)-y1
        let lambda = (p2.y() + self.y()).div_unsafe(self.x() - p2.x());
        let x3 = &lambda * &lambda - self.x() - p2.x();
        let y3 = lambda * (self.x() - &x3) - self.y();
        Self(AffinePoint::new(x3, y3))
    }

    fn sub_ne_assign_nonidentity(&mut self, p2: &Self) {
        // TODO: revisit if there are possible optimizations
        *self = self.sub_ne_nonidentity(p2);
    }
}

impl Group for G2Affine {
    type SelfRef<'a> = &'a Self;

    const IDENTITY: Self = Self::IDENTITY;

    fn double(&self) -> Self {
        if self.is_identity() {
            self.clone()
        } else {
            self.double_nonidentity()
        }
    }

    fn double_assign(&mut self) {
        if !self.is_identity() {
            self.double_assign_nonidentity();
        }
    }
}

impl core::ops::Add<&G2Affine> for G2Affine {
    type Output = Self;

    fn add(mut self, p2: &G2Affine) -> Self::Output {
        self.add_assign(p2);
        self
    }
}

impl core::ops::Add for G2Affine {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.add(&rhs)
    }
}

impl core::ops::Add<&G2Affine> for &G2Affine {
    type Output = G2Affine;

    fn add(self, p2: &G2Affine) -> Self::Output {
        if self.is_identity() {
            p2.clone()
        } else if p2.is_identity() {
            self.clone()
        } else if self.x() == p2.x() {
            if self.y() + p2.y() == Fp2::ZERO {
                G2Affine::IDENTITY
            } else {
                self.double_nonidentity()
            }
        } else {
            self.add_ne_nonidentity(p2)
        }
    }
}

impl core::ops::AddAssign<&G2Affine> for G2Affine {
    fn add_assign(&mut self, p2: &G2Affine) {
        if self.is_identity() {
            *self = p2.clone();
        } else if p2.is_identity() {
            // do nothing
        } else if self.x() == p2.x() {
            if self.y() + p2.y() == Fp2::ZERO {
                *self = Self::IDENTITY;
            } else {
                self.double_assign_nonidentity();
            }
        } else {
            self.add_ne_assign_nonidentity(p2);
        }
    }
}

impl core::ops::AddAssign for G2Affine {
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign(&rhs);
    }
}

impl core::ops::Neg for G2Affine {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.0.y.neg_assign();
        self
    }
}

impl core::ops::Sub<&G2Affine> for G2Affine {
    type Output = Self;

    fn sub(self, rhs: &G2Affine) -> Self::Output {
        self.sub(rhs.clone())
    }
}

impl core::ops::Sub for G2Affine {
    type Output = G2Affine;

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(&rhs)
    }
}

impl core::ops::Sub<&G2Affine> for &G2Affine {
    type Output = G2Affine;

    fn sub(self, p2: &G2Affine) -> Self::Output {
        if p2.is_identity() {
            self.clone()
        } else if self.is_identity() {
            G2Affine(p2.0.neg_borrow())
        } else if self.x() == p2.x() {
            if self.y() == p2.y() {
                G2Affine::IDENTITY
            } else {
                self.double_nonidentity()
            }
        } else {
            self.sub_ne_nonidentity(p2)
        }
    }
}

impl core::ops::SubAssign<&G2Affine> for G2Affine {
    fn sub_assign(&mut self, p2: &G2Affine) {
        if p2.is_identity() {
            // do nothing
        } else if self.is_identity() {
            *self = G2Affine(p2.0.neg_borrow());
        } else if self.x() == p2.x() {
            if self.y() == p2.y() {
                *self = Self::IDENTITY;
            } else {
                self.double_assign_nonidentity();
            }
        } else {
            self.sub_ne_assign_nonidentity(p2);
        }
    }
}

impl core::ops::SubAssign for G2Affine {
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_assign(&rhs);
    }
}
