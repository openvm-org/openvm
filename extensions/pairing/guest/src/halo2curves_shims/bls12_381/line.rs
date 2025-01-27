extern crate std;

use std::ops::{Add, Mul, Neg, Sub};

use halo2curves_axiom::bls12_381::{Fq12, Fq2};
use openvm_ecc_guest::{
    algebra::{field::FieldExtension, Field},
    AffinePoint,
};

use super::{Bls12_381, BLS12_381_XI};
use crate::pairing::{EvaluatedLine, LineMulMType};

impl LineMulMType<Fq2, Fq12> for Bls12_381 {
    fn mul_023_by_023(l0: &EvaluatedLine<Fq2>, l1: &EvaluatedLine<Fq2>) -> [Fq2; 5] {
        let b0 = &l0.b;
        let c0 = &l0.c;
        let b1 = &l1.b;
        let c1 = &l1.c;

        // where w⁶ = xi
        // l0 * l1 = c0c1 + (c0b1 + c1b0)w² + (c0 + c1)w³ + (b0b1)w⁴ + (b0 +b1)w⁵ + w⁶
        //         = (c0c1 + xi) + (c0b1 + c1b0)w² + (c0 + c1)w³ + (b0b1)w⁴ + (b0 + b1)w⁵
        let x0 = c0 * c1 + *BLS12_381_XI;
        let x2 = c0 * b1 + c1 * b0;
        let x3 = c0 + c1;
        let x4 = b0 * b1;
        let x5 = b0 + b1;

        [x0, x2, x3, x4, x5]
    }

    /// Multiplies a line in 023-form with a Fp12 element to get an Fp12 element
    fn mul_by_023(f: &Fq12, l: &EvaluatedLine<Fq2>) -> Fq12 {
        Self::mul_by_02345(f, &[l.c, l.b, Fq2::ONE, Fq2::ZERO, Fq2::ZERO])
    }

    /// Multiplies a line in 02345-form with a Fp12 element to get an Fp12 element
    fn mul_by_02345(f: &Fq12, x: &[Fq2; 5]) -> Fq12 {
        let fx = Fq12::from_coeffs([x[0], Fq2::ZERO, x[1], x[2], x[3], x[4]]);
        f * fx
    }
}

/// Returns a line function for a tangent line at the point P
#[allow(non_snake_case)]
pub fn tangent_line_023<Fp, Fp2>(P: AffinePoint<Fp>) -> EvaluatedLine<Fp2>
where
    Fp: Field,
    Fp2: FieldExtension<Fp> + Field,
    for<'a> &'a Fp: Add<&'a Fp, Output = Fp>,
    for<'a> &'a Fp: Sub<&'a Fp, Output = Fp>,
    for<'a> &'a Fp: Mul<&'a Fp, Output = Fp>,
    for<'a> &'a Fp2: Add<&'a Fp2, Output = Fp2>,
    for<'a> &'a Fp2: Sub<&'a Fp2, Output = Fp2>,
    for<'a> &'a Fp2: Mul<&'a Fp2, Output = Fp2>,
    for<'a> &'a Fp2: Neg<Output = Fp2>,
{
    let one = &Fp2::ONE;
    let two = &(one + one);
    let three = &(one + two);
    let x = &Fp2::embed(P.x);
    let y = &Fp2::embed(P.y);

    // λ = (3x^2) / (2y)
    // 1 - λ(x/y)w^-1 + (λx - y)(1/y)w^-3
    // = (λx - y)(1/y) - λ(x/y)w^2 + w^3
    //
    // b = -(λ * x / y)
    //   = -3x^3 / 2y^2
    // c = (λ * x - y) / y
    //   = 3x^3/2y^2 - 1
    let x_squared = &(x * x);
    let x_cubed = &(x_squared * x);
    let y_squared = &(y * y);
    let three_x_cubed = three * x_cubed;
    let two_y_squared = two * y_squared;

    let b = three_x_cubed.clone().neg().div_unsafe(&two_y_squared);
    let c = three_x_cubed.div_unsafe(&two_y_squared) - &Fp2::ONE;

    EvaluatedLine { b, c }
}
