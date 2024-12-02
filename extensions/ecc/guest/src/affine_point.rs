use core::ops::Neg;

use axvm_algebra_guest::Field;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
