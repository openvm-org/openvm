use core::{
    fmt::Debug,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

mod complex;
pub use complex::*;

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
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
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

/// Sextic extension field of `F` with irreducible polynomial `X^6 + \xi`.
/// Elements are represented as `c0 + c1 * w` where `w^6 = \xi`, where \xi depends on the twist of the curve.
///
/// Memory alignment follows alignment of `F`.
/// Memory layout is concatenation of `c0` and `c1`.
#[derive(Clone, PartialEq, Eq)]
#[repr(C)]
pub struct SexticExtField<F> {
    pub c: [F; 6],
}

impl<F: Field> SexticExtField<F> {
    pub const fn new(c: [F; 6]) -> Self {
        Self { c }
    }
}
