use core::ops::Neg;

use axvm_algebra::{
    field::{Complex, FieldExtension},
    Field,
};

use super::Fp;

pub type Fp2 = Complex<Fp>;

impl FieldExtension<Fp> for Fp2 {
    const D: usize = 2;
    type Coeffs = [Fp; 2];

    fn from_coeffs([c0, c1]: Self::Coeffs) -> Self {
        Self { c0, c1 }
    }

    fn to_coeffs(self) -> Self::Coeffs {
        [self.c0, self.c1]
    }

    fn embed(c0: Fp) -> Self {
        Self { c0, c1: Fp::ZERO }
    }

    fn frobenius_map(&self, power: usize) -> Self {
        if power % 2 == 0 {
            self.clone()
        } else {
            Self {
                c0: self.c0.clone(),
                c1: (&self.c1).neg(),
            }
        }
    }

    fn mul_base(&self, rhs: &Fp) -> Self {
        Self {
            c0: &self.c0 * rhs,
            c1: &self.c1 * rhs,
        }
    }
}
