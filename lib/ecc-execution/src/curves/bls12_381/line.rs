use std::ops::{Add, Mul, Neg, Sub};

use axvm_ecc::{
    curve::bls12381::{Fq, Fq12, Fq2},
    field::{Field, FieldExtension},
    pairing::{EvaluatedLine, LineMulMType},
    point::AffinePoint,
};

use super::{Bls12_381, BLS12381_XI};

impl LineMulMType<Fq, Fq2, Fq12> for Bls12_381 {
    fn mul_023_by_023(line_0: EvaluatedLine<Fq, Fq2>, line_1: EvaluatedLine<Fq, Fq2>) -> [Fq2; 5] {
        let b0 = line_0.b;
        let c0 = line_0.c;
        let b1 = line_1.b;
        let c1 = line_1.c;

        // where w⁶ = xi
        // l0 * l1 = c0c1 + (c0b1 + c1b0)w² + (c0 + c1)w³ + (b0b1)w⁴ + (b0 +b1)w⁵ + w⁶
        //         = (c0c1 + xi) + (c0b1 + c1b0)w² + (c0 + c1)w³ + (b0b1)w⁴ + (b0 + b1)w⁵
        let x0 = c0 * c1 + *BLS12381_XI;
        let x2 = c0 * b1 + c1 * b0;
        let x3 = c0 + c1;
        let x4 = b0 * b1;
        let x5 = b0 + b1;

        [x0, x2, x3, x4, x5]
    }

    fn mul_by_023(f: Fq12, l: EvaluatedLine<Fq, Fq2>) -> Fq12 {
        Self::mul_by_02345(f, [l.c, l.b, Fq2::ONE, Fq2::ZERO, Fq2::ZERO])
    }

    fn mul_by_02345(f: Fq12, x: [Fq2; 5]) -> Fq12 {
        let x_fp12 = Fq12::from_coeffs([x[0], Fq2::ZERO, x[1], x[2], x[3], x[4]]);
        f * x_fp12
    }
}

/// Returns a line function for a tangent line at the point P
#[allow(non_snake_case)]
pub fn tangent_line_023<Fp, Fp2>(P: AffinePoint<Fp>) -> EvaluatedLine<Fp, Fp2>
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
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
    let three_x_cubed = &(three * x_cubed);
    let over_two_y_squared = &(two * y_squared).invert().unwrap();

    let b = three_x_cubed.neg() * over_two_y_squared;
    let c = three_x_cubed * over_two_y_squared - &Fp2::ONE;

    EvaluatedLine { b, c }
}
