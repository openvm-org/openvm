use core::{
    fmt::Debug,
    ops::{Add, AddAssign, Neg, Sub, SubAssign},
};

pub trait Group:
    Clone
    + Debug
    + Eq
    + Sized
    + Add<Output = Self>
    + Sub<Output = Self>
    + Neg<Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + AddAssign
    + SubAssign
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
{
    type SelfRef<'a>: Add<&'a Self, Output = Self> + Sub<&'a Self, Output = Self>
    where
        Self: 'a;

    const IDENTITY: Self;

    fn is_identity(&self) -> bool;

    fn double(&self) -> Self;
    fn double_assign(&mut self);
}

pub trait CyclicGroup: Group {
    const GENERATOR: Self;
    const NEG_GENERATOR: Self;
}
