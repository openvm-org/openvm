use core::{
    fmt::Debug,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// Complex quadratic extension of any ring.
mod complex;
pub use complex::*;

use crate::{DivAssignUnsafe, DivUnsafe};

// TODO: this should extend an IntegralDomain trait
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
    + for<'a> DivUnsafe<&'a Self, Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssignUnsafe
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
    + for<'a> DivAssignUnsafe<&'a Self>
{
    type SelfRef<'a>: Add<&'a Self, Output = Self>
        + Sub<&'a Self, Output = Self>
        + Mul<&'a Self, Output = Self>
        + DivUnsafe<&'a Self, Output = Self>
    where
        Self: 'a;

    /// The zero element of the field, the additive identity.
    const ZERO: Self;

    /// The one element of the field, the multiplicative identity.
    const ONE: Self;
}

/// Field extension trait. BaseField is the base field of the extension field.
pub trait FieldExtension<BaseField: Field>: Field {
    /// Extension field degree.
    const D: usize;
    /// This should be [BaseField; D]. It is an associated type due to rust const generic limitations.
    type Coeffs: Sized;

    /// Generate an extension field element from its base field coefficients.
    fn from_coeffs(coeffs: Self::Coeffs) -> Self;

    /// Convert an extension field element to its base field coefficients.
    fn to_coeffs(self) -> Self::Coeffs;

    /// Embed a base field element into an extension field element.
    fn embed(base_elem: BaseField) -> Self;

    /// Conjuagte an extension field element.
    fn conjugate(&self) -> Self;

    /// Frobenius map
    fn frobenius_map(&self, power: usize) -> Self;

    /// Multiply an extension field element by an element in the base field
    fn mul_base(&self, rhs: BaseField) -> Self;
}
