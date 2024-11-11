use core::ops::{Div, DivAssign};

use super::Field;

pub trait DivUnsafe: Field + Div<Output = Self> + for<'a> Div<&'a Self, Output = Self> {
    fn div_refs_impl(&self, other: &Self) -> Self;
}

pub trait DivUnsafeAssign: Field + DivAssign + for<'a> DivAssign<&'a Self> {
    fn div_assign_impl(&mut self, other: &Self);
}
