use core::ops::Neg;

use openvm_algebra_guest::Field;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[repr(C)]
pub struct AffinePoint<F> {
    pub x: F,
    pub y: F,
}

// Note that this AffinePoint is slightly different from the one in the short Weierstrass ec
// extension. In particular, it handles negation and the identity point differently.
impl<F: Field> AffinePoint<F> {
    pub const fn new(x: F, y: F) -> Self {
        Self { x, y }
    }

    pub fn neg_borrow<'a>(&'a self) -> Self
    where
        &'a F: Neg<Output = F>,
    {
        Self {
            x: Neg::neg(&self.x),
            y: self.y.clone(),
        }
    }

    // For twisted Edwards curves, the point at infinity is represented by (0, 1)
    pub fn is_infinity(&self) -> bool {
        self.x == F::ZERO && self.y == F::ONE
    }
}

// Note: this is true for twisted Edwards curves but maybe not in general
impl<F> Neg for AffinePoint<F>
where
    F: Neg<Output = F>,
{
    type Output = AffinePoint<F>;

    fn neg(self) -> AffinePoint<F> {
        Self {
            x: self.x.neg(),
            y: self.y,
        }
    }
}
