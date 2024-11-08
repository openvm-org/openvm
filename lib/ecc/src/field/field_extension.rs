use core::ops::{Add, Mul, Sub};

use super::Field;

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

    /// Embed a base field element into an extension field element.
    fn embed(base_elem: Self::BaseField) -> Self;

    /// Conjuagte an extension field element.
    fn conjugate(&self) -> Self;

    /// Frobenius map
    fn frobenius_map(&self, power: Option<usize>) -> Self;

    /// Multiply an extension field element by an element in the base field
    fn mul_base(&self, rhs: Self::BaseField) -> Self;
}
