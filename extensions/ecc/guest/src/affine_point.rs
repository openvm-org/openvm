use core::ops::{Add, AddAssign, Neg, Sub, SubAssign};

use openvm_algebra_guest::Field;

use crate::Group;

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
        let x = self.x + rhs.x;
        let y = self.y + rhs.y;
        Self { x, y }
    }
}

impl<'a, F: Field> Add<&'a Self> for AffinePoint<F> {
    type Output = Self;

    fn add(self, rhs: &'a Self) -> Self {
        let x = self.x + rhs.x.clone();
        let y = self.y + rhs.y.clone();
        Self { x, y }
    }
}

impl<F: Field> Add<&AffinePoint<F>> for &AffinePoint<F> {
    type Output = AffinePoint<F>;

    fn add(self, rhs: &AffinePoint<F>) -> Self::Output {
        self.clone() + rhs.clone()
    }
}

impl<F: Field> AddAssign for AffinePoint<F> {
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign(&rhs);
    }
}

impl<F: Field> AddAssign<&AffinePoint<F>> for AffinePoint<F> {
    fn add_assign(&mut self, rhs: &AffinePoint<F>) {
        *self = self.clone() + rhs;
    }
}

impl<F: Field> Sub for AffinePoint<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let x = self.x - rhs.x;
        let y = self.y - rhs.y;
        Self { x, y }
    }
}

impl<'a, F: Field> Sub<&'a Self> for AffinePoint<F> {
    type Output = Self;

    fn sub(self, rhs: &'a Self) -> Self {
        let x = self.x - rhs.x.clone();
        let y = self.y - rhs.y.clone();
        Self { x, y }
    }
}

impl<F: Field> Sub<&AffinePoint<F>> for &AffinePoint<F> {
    type Output = AffinePoint<F>;

    fn sub(self, rhs: &AffinePoint<F>) -> Self::Output {
        self.clone() - rhs.clone()
    }
}

impl<F: Field> SubAssign for AffinePoint<F> {
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_assign(&rhs);
    }
}

impl<F: Field> SubAssign<&AffinePoint<F>> for AffinePoint<F> {
    fn sub_assign(&mut self, rhs: &AffinePoint<F>) {
        *self = self.clone() - rhs;
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
