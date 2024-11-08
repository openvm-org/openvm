use core::{
    fmt::Debug,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

mod field_ext;
pub use field_ext::*;

mod complex;
pub use complex::*;

mod sextic_ext_field;
pub use sextic_ext_field::*;

#[cfg(feature = "halo2curves")]
mod exp_bytes_be;
#[cfg(feature = "halo2curves")]
pub use exp_bytes_be::*;

/// This is a simplified trait for field elements.
pub trait Field:
    Sized
    + Clone
    + Debug
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + for<'a> Div<&'a Self, Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
{
    type SelfRef<'a>: Add<&'a Self, Output = Self>
        + Sub<&'a Self, Output = Self>
        + Mul<&'a Self, Output = Self>
    where
        Self: 'a;

    /// The zero element of the field, the additive identity.
    const ZERO: Self;

    /// The one element of the field, the multiplicative identity.
    const ONE: Self;

    /// Squares this field element.
    fn square(&self) -> Self;

    /// Inverts this element, returning `None` if this element is zero.
    fn invert(&self) -> Option<Self>;
}
