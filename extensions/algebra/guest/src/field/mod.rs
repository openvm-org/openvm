use alloc::vec::Vec;
use core::{
    fmt::Debug,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::{DivAssignUnsafe, DivUnsafe};

// TODO[jpw]: the shared parts of Field and IntMod should be moved into a new `IntegralDomain`
// trait.
/// This is a simplified trait for field elements.
pub trait Field:
    Sized
    + Eq
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

    /// Doubles `self` in-place.
    fn double_assign(&mut self);

    /// Square `self` in-place
    fn square_assign(&mut self);

    /// Unchecked inversion. See [DivUnsafe].
    ///
    /// ## Panics
    /// If `self` is zero.
    fn invert(&self) -> Self {
        Self::ONE.div_unsafe(self)
    }
}

/// Field extension trait. BaseField is the base field of the extension field.
pub trait FieldExtension<BaseField> {
    /// Extension field degree.
    const D: usize;
    /// This should be [BaseField; D]. It is an associated type due to rust const generic
    /// limitations.
    type Coeffs: Sized;

    /// Create an extension field element from its base field coefficients.
    fn from_coeffs(coeffs: Self::Coeffs) -> Self;

    /// Create an extension field element from little-endian bytes.
    fn from_bytes(bytes: &[u8]) -> Self;

    /// Convert an extension field element to its base field coefficients.
    fn to_coeffs(self) -> Self::Coeffs;

    /// Convert an extension field element to little-endian bytes.
    fn to_bytes(&self) -> Vec<u8>;

    /// Embed a base field element into an extension field element.
    fn embed(base_elem: BaseField) -> Self;

    /// Frobenius map: take `self` to the `p^power`th power, where `p` is the prime characteristic
    /// of the field.
    fn frobenius_map(&self, power: usize) -> Self;

    /// Multiply an extension field element by an element in the base field
    fn mul_base(&self, rhs: &BaseField) -> Self;
}

pub trait ComplexConjugate {
    /// Conjugate an extension field element.
    fn conjugate(self) -> Self;

    /// Replace `self` with its conjugate.
    fn conjugate_assign(&mut self);
}
