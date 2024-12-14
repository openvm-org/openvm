use core::ops::{Add, AddAssign, Neg, Sub, SubAssign};

use openvm_algebra_guest::Field;

use crate::{weierstrass::IntrinsicCurve, Group};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[repr(C)]
pub struct AffinePoint<F> {
    pub x: F,
    pub y: F,
}

impl<F: Field> AffinePoint<F> {
    pub fn new(x: F, y: F) -> Self {
        Self { x, y }
    }

    pub fn neg_borrow<'a>(&'a self) -> Self
    where
        &'a F: Neg<Output = F>,
    {
        Self {
            x: self.x.clone(),
            y: Neg::neg(&self.y),
        }
    }

    pub fn is_infinity(&self) -> bool {
        self.x == F::ZERO && self.y == F::ZERO
    }

    fn add_impl(&self, rhs: &Self) -> Self {
        if self.is_infinity() {
            return rhs.clone();
        }
        if rhs.is_infinity() {
            return self.clone();
        }

        if self.x == rhs.x {
            if self.y == rhs.y.clone().neg() {
                return Self::IDENTITY;
            }
            if self.y == rhs.y {
                return self.double();
            }
        }

        // let lambda = (rhs.y.clone() - self.y.clone()).div_unsafe(&(rhs.x.clone() - self.x.clone()));
        let mut lambda = rhs.y.clone();
        lambda -= self.y.clone();
        let mut denom = rhs.x.clone();
        denom -= self.x.clone();
        lambda.div_assign_unsafe(&denom);

        // x3 = lambda^2 - x1 - x2
        let mut x3 = lambda.clone();
        x3.square_assign();
        x3 -= self.x.clone();
        x3 -= rhs.x.clone();

        // y3 = lambda * (x1 - x3) - y1
        let x1_minus_x3 = self.x.clone() - x3.clone();
        let mut y3 = lambda;
        y3 *= x1_minus_x3;
        y3 -= self.y.clone();

        Self::new(x3, y3)
    }
}

impl<F> Neg for AffinePoint<F>
where
    F: Neg<Output = F>,
{
    type Output = AffinePoint<F>;

    fn neg(self) -> AffinePoint<F> {
        Self {
            x: self.x,
            y: self.y.neg(),
        }
    }
}

impl<F: Field> Add for AffinePoint<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        self.add_impl(&rhs)
    }
}

impl<'a, F: Field> Add<&'a Self> for AffinePoint<F> {
    type Output = Self;

    fn add(self, rhs: &'a Self) -> Self {
        self.add_impl(rhs)
    }
}

impl<F: Field> Add<&AffinePoint<F>> for &AffinePoint<F> {
    type Output = AffinePoint<F>;

    fn add(self, rhs: &AffinePoint<F>) -> Self::Output {
        self.add_impl(rhs)
    }
}

impl<F: Field> AddAssign for AffinePoint<F> {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.add_impl(&rhs);
    }
}

impl<F: Field> AddAssign<&AffinePoint<F>> for AffinePoint<F> {
    fn add_assign(&mut self, rhs: &AffinePoint<F>) {
        *self = self.add_impl(rhs);
    }
}

impl<F: Field> Sub for AffinePoint<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self.add_impl(&rhs.clone().neg())
    }
}

impl<'a, F: Field> Sub<&'a Self> for AffinePoint<F> {
    type Output = Self;

    fn sub(self, rhs: &'a Self) -> Self {
        self.add_impl(&rhs.clone().neg())
    }
}

impl<F: Field> Sub<&AffinePoint<F>> for &AffinePoint<F> {
    type Output = AffinePoint<F>;

    fn sub(self, rhs: &AffinePoint<F>) -> Self::Output {
        self.add_impl(&rhs.clone().neg())
    }
}

impl<F: Field> SubAssign for AffinePoint<F> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.add_impl(&rhs.clone().neg());
    }
}

impl<F: Field> SubAssign<&AffinePoint<F>> for AffinePoint<F> {
    fn sub_assign(&mut self, rhs: &AffinePoint<F>) {
        *self = self.add_impl(&rhs.clone().neg());
    }
}

impl<F: Field> Group for AffinePoint<F> {
    type SelfRef<'a>
        = &'a Self
    where
        Self: 'a;

    const IDENTITY: Self = Self {
        x: F::ZERO,
        y: F::ZERO,
    };

    fn is_identity(&self) -> bool {
        self.x == F::ZERO && self.y == F::ZERO
    }

    fn double(&self) -> Self {
        if self.is_identity() {
            self.clone()
        } else {
            self.clone() + self
        }
    }

    fn double_assign(&mut self) {
        if self.is_identity() {
            return;
        }
        *self = self.double();
    }
}
