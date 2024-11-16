use core::ops::{Add, Mul, Sub};

use halo2curves_axiom::ff;

use crate::{field::Field, DivAssignUnsafe, DivUnsafe};

impl<'a, F: ff::Field> DivUnsafe<&'a F> for F {
    type Output = F;

    fn div_unsafe(self, other: &'a F) -> Self::Output {
        self * other.invert().unwrap()
    }
}

impl<'a, F: ff::Field> DivUnsafe<&'a F> for &'a F {
    type Output = F;

    fn div_unsafe(self, other: &'a F) -> Self::Output {
        *self * other.invert().unwrap()
    }
}

impl<F: ff::Field> DivAssignUnsafe for F {
    fn div_assign_unsafe(&mut self, other: Self) {
        *self *= other.invert().unwrap();
    }
}

impl<'a, F: ff::Field> DivAssignUnsafe<&'a F> for F {
    fn div_assign_unsafe(&mut self, other: &'a F) {
        *self *= other.invert().unwrap();
    }
}

impl<F: ff::Field> Field for F
where
    for<'a> &'a F: Add<&'a F, Output = F> + Sub<&'a F, Output = F> + Mul<&'a F, Output = F>,
{
    const ZERO: Self = <F as ff::Field>::ZERO;
    const ONE: Self = <F as ff::Field>::ONE;

    type SelfRef<'a> = &'a F;
}
