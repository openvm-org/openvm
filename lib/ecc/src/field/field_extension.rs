use core::ops::{Add, Mul, Sub};

use super::Field;

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

pub struct SexticExtFieldMtype<Fp2>(pub(crate) [Fp2; 5]);

pub struct SexticExtFieldDtype<Fp2>(pub(crate) [Fp2; 5]);
