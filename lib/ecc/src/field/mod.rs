use ff::Field;

mod complex;
#[cfg(feature = "halo2curves")]
mod exp_bytes_be;

pub use complex::*;
#[cfg(feature = "halo2curves")]
pub use exp_bytes_be::*;

pub trait FieldExtension: Field {
    type BaseField: Field;
    type Coeffs: Sized;

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
