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

pub trait ExpBigInt<F: Field>: Field {
    /// Exponentiates a field element by a BigInt
    fn exp_bigint(&self, k: BigInt) -> Self {
        if k == BigInt::from(0) {
            return Self::ONE;
        }

        let mut e = k.clone();
        let mut x = *self;

        if k < BigInt::from(0) {
            x = x.invert().unwrap();
            e = -k;
        }

        let mut res = Self::ONE;

        let x_sq = x.square();
        let ops = [x, x_sq, x_sq * x];

        let bytes = e.to_bytes_be();
        for &b in bytes.1.iter() {
            let mut mask = 0xc0;
            for j in 0..4 {
                res = res.square().square();
                let c = (b & mask) >> (6 - 2 * j);
                if c != 0 {
                    res *= &ops[(c - 1) as usize];
                }
                mask >>= 2;
            }
        }

        res
    }
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
