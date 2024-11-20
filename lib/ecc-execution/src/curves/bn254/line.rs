use std::ops::{Add, Mul, Neg, Sub};

use axvm_ecc::{
    algebra::{field::FieldExtension, Field},
    pairing::{EvaluatedLine, LineMulDType},
    AffinePoint,
};
use halo2curves_axiom::bn256::{Fq12, Fq2};

use super::Bn254;

impl LineMulDType<Fq2, Fq12> for Bn254 {
    /// Multiplies two lines in 013-form to get an element in 01234-form
    fn mul_013_by_013(l0: &EvaluatedLine<Fq2>, l1: &EvaluatedLine<Fq2>) -> [Fq2; 5] {
        let b0 = &l0.b;
        let c0 = &l0.c;
        let b1 = &l1.b;
        let c1 = &l1.c;

        // where w⁶ = xi
        // l0 * l1 = 1 + (b0 + b1)w + (b0b1)w² + (c0 + c1)w³ + (b0c1 + b1c0)w⁴ + (c0c1)w⁶
        //         = (1 + c0c1 * xi) + (b0 + b1)w + (b0b1)w² + (c0 + c1)w³ + (b0c1 + b1c0)w⁴
        let x0 = Fq2::ONE + c0 * c1 * *BN254_XI;
        let x1 = b0 + b1;
        let x2 = b0 * b1;
        let x3 = c0 + c1;
        let x4 = b0 * c1 + b1 * c0;

        [x0, x1, x2, x3, x4]
    }

    /// Multiplies a line in 013-form with a Fp12 element to get an Fp12 element
    fn mul_by_013(f: &Fq12, l: &EvaluatedLine<Fq2>) -> Fq12 {
        Self::mul_by_01234(f, &[Fq2::ONE, l.b, Fq2::ZERO, l.c, Fq2::ZERO])
    }

    /// Multiplies a line in 01234-form with a Fp12 element to get an Fp12 element
    fn mul_by_01234(f: &Fq12, x: &[Fq2; 5]) -> Fq12 {
        let fx = Fq12::from_coeffs([x[0], x[1], x[2], x[3], x[4], Fq2::ZERO]);
        f * fx
    }
}

/// Returns a line function for a tangent line at the point P
#[allow(non_snake_case)]
pub fn tangent_line_013<Fp, Fp2>(P: AffinePoint<Fp>) -> EvaluatedLine<Fp2>
where
    Fp: Field,
    Fp2: Field + FieldExtension<Fp>,
    for<'a> &'a Fp: Add<&'a Fp, Output = Fp>,
    for<'a> &'a Fp: Sub<&'a Fp, Output = Fp>,
    for<'a> &'a Fp: Mul<&'a Fp, Output = Fp>,
    for<'a> &'a Fp2: Add<&'a Fp2, Output = Fp2>,
    for<'a> &'a Fp2: Sub<&'a Fp2, Output = Fp2>,
    for<'a> &'a Fp2: Mul<&'a Fp2, Output = Fp2>,
    for<'a> &'a Fp2: Neg<Output = Fp2>,
{
    let one = Fp2::ONE;
    let two = &one + &one;
    let three = &one + &two;
    let x = Fp2::embed(P.x);
    let y = Fp2::embed(P.y);

    // λ = (3x^2) / (2y)
    // 1 - λ(x/y)w + (λx - y)(1/y)w^3
    // b = -(λ * x / y)
    //   = -3x^3 / 2y^2
    // c = (λ * x - y) / y
    //   = 3x^3/2y^2 - 1
    let x_squared = &x * &x;
    let x_cubed = x_squared * &x;
    let y_squared = &y * &y;
    let three_x_cubed = &three * &x_cubed;
    let over_two_y_squared = Fp2::ONE.div_unsafe(&(&two * &y_squared));

    let b = (&three_x_cubed).neg() * &over_two_y_squared;
    let c = &three_x_cubed * &over_two_y_squared - Fp2::ONE;

    EvaluatedLine { b, c }
}
