use halo2curves_axiom::ff::Field;
use num::BigInt;

pub trait FieldExtension: Field {
    type BaseField: Field;

    /// Generate an extension field element from its base field coefficients.
    fn from_coeffs(coeffs: &[Self::BaseField]) -> Self;

    /// Embed a base field element into an extension field element.
    fn embed(base_elem: &Self::BaseField) -> Self;

    /// Conjuagte an extension field element.
    fn conjugate(&self) -> Self;

    /// Frobenius map
    fn frobenius_map(&self, power: Option<usize>) -> Self;

    /// Multiply an extension field element by an element in the base field
    fn mul_base(&self, rhs: &Self::BaseField) -> Self;
}

pub trait Fp2Constructor<Fp: Field> {
    /// Constructs a new Fp2 element from 2 Fp coefficients.
    fn new(c0: Fp, c1: Fp) -> Self;
}

pub trait Fp12Constructor<Fp2: FieldExtension> {
    /// Constructs a new Fp12 element from 6 Fp2 coefficients.
    fn new(c00: Fp2, c01: Fp2, c02: Fp2, c10: Fp2, c11: Fp2, c12: Fp2) -> Self;
}

pub trait ExpBigInt<Fp: Field> {
    /// Exponentiates a field element by a BigInt
    fn exp_bigint(&self, k: BigInt) -> Fp;
}

#[cfg(test)]
pub trait FeltPrint<Fp: Field> {
    fn felt_print(&self, label: &str);
}

pub fn fp12_square<Fp12: Field>(x: Fp12) -> Fp12 {
    fp12_multiply(x, x)
}

pub fn fp12_multiply<Fp12: Field>(x: Fp12, y: Fp12) -> Fp12 {
    x * y
}
