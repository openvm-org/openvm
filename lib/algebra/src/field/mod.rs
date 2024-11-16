use core::{
    fmt::Debug,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// Complex quadratic extension of any ring.
mod complex;
pub use complex::*;

/// This is a simplified trait for field elements.
pub trait Field:
    Sized
    + Clone
    + Debug
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + for<'a, 'b> Mul<&'b Self, Output = Self>
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
        + for<'b> Mul<&'b Self, Output = Self>
    where
        Self: 'a;

    /// The zero element of the field, the additive identity.
    fn zero() -> Self;

    /// The one element of the field, the multiplicative identity.
    fn one() -> Self;

    /// Inverts this element, returning `None` if this element is zero.
    fn invert(&self) -> Option<Self>;
}

/// Field extension trait. BaseField is the base field of the extension field. Coeffs is a fixed size array
/// of coefficients of base field types and how many there are to get to the extension field.
pub trait FieldExtension: Field {
    type BaseField: Field;
    type Coeffs: Sized;
    type SelfRef<'a>: Add<&'a Self, Output = Self>
        + Sub<&'a Self, Output = Self>
        + Mul<&'a Self, Output = Self>
    where
        Self: 'a;

    /// Generate an extension field element from its base field coefficients.
    fn from_coeffs(coeffs: Self::Coeffs) -> Self;

    /// Convert an extension field element to its base field coefficients.
    fn to_coeffs(self) -> Self::Coeffs;

    /// Embed a base field element into an extension field element.
    fn embed(base_elem: Self::BaseField) -> Self;

    /// Conjuagte an extension field element.
    fn conjugate(&self) -> Self;

    /// Frobenius map
    fn frobenius_map(&self, power: Option<usize>) -> Self;

    /// Multiply an extension field element by an element in the base field
    fn mul_base(&self, rhs: Self::BaseField) -> Self;
}
