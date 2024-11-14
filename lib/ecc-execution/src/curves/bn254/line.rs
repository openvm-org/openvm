use std::ops::{Add, Mul, Neg, Sub};

use axvm_ecc::{
    curve::bn254::{Fq, Fq12, Fq2},
    field::{Field, FieldExtension},
    pairing::{EvaluatedLine, LineMulDType},
    point::AffinePoint,
};

use super::{Bn254, BN254_XI};

impl LineMulDType<Fq, Fq2, Fq12> for Bn254 {
    fn mul_013_by_013(line_0: EvaluatedLine<Fq, Fq2>, line_1: EvaluatedLine<Fq, Fq2>) -> [Fq2; 5] {
        let b0 = line_0.b;
        let c0 = line_0.c;
        let b1 = line_1.b;
        let c1 = line_1.c;

        // where w⁶ = xi
        // l0 * l1 = 1 + (b0 + b1)w + (b0b1)w² + (c0 + c1)w³ + (b0c1 + b1c0)w⁴ + (c0c1)w⁶
        //         = (1 + c0c1 * xi) + (b0 + b1)w + (b0b1)w² + (c0 + c1)w³ + (b0c1 + b1c0)w⁴
        let l0 = Fq2::one() + c0 * c1 * *BN254_XI;
        let l1 = b0 + b1;
        let l2 = b0 * b1;
        let l3 = c0 + c1;
        let l4 = b0 * c1 + b1 * c0;

        [l0, l1, l2, l3, l4]
    }

    fn mul_by_013(f: Fq12, l: EvaluatedLine<Fq, Fq2>) -> Fq12 {
        Self::mul_by_01234(f, [Fq2::one(), l.b, Fq2::zero(), l.c, Fq2::zero()])
    }

    fn mul_by_01234(f: Fq12, x: [Fq2; 5]) -> Fq12 {
        let x_fp12 = Fq12::from_coeffs([x[0], x[1], x[2], x[3], x[4], Fq2::zero()]);
        f * x_fp12
    }
}

/// Returns a line function for a tangent line at the point P
#[allow(non_snake_case)]
pub fn tangent_line_013<Fp, Fp2>(P: AffinePoint<Fp>) -> EvaluatedLine<Fp, Fp2>
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
    let one = &Fp2::one();
    let two = &(one + one);
    let three = &(one + two);
    let x = &Fp2::embed(P.x);
    let y = &Fp2::embed(P.y);

    // λ = (3x^2) / (2y)
    // 1 - λ(x/y)w + (λx - y)(1/y)w^3
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
    let c = three_x_cubed * over_two_y_squared - &Fp2::one();

    EvaluatedLine { b, c }
}
