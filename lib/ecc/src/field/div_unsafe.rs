use core::ops::{Div, DivAssign};

pub trait DivUnsafe: Sized + Div<Output = Self> + for<'a> Div<&'a Self, Output = Self> {
    fn div_refs_impl(&self, other: &Self) -> Self;
}

pub trait DivAssignUnsafe: Sized + DivAssign + for<'a> DivAssign<&'a Self> {
    fn div_assign_impl(&mut self, other: &Self);
}
