use std::ops::{Add, Mul, Neg, Sub};

use axvm_ecc::{
    curve::bn254::{Fq, Fq12, Fq2},
    field::{Field, FieldExt, SexticExtField},
    pairing::{EvaluatedLine, LineMulDType, UnevaluatedLine},
    point::AffinePoint,
};

use super::Bn254;

impl LineMulDType<Fq, Fq2, Fq12> for Bn254 {
    fn mul_013_by_013(
        line_0: EvaluatedLine<Fq, Fq2>,
        line_1: EvaluatedLine<Fq, Fq2>,
    ) -> SexticExtField<Fq2> {
        let b0 = line_0.b;
        let c0 = line_0.c;
        let b1 = line_1.b;
        let c1 = line_1.c;

        // where w⁶ = xi
        // l0 * l1 = 1 + (b0 + b1)w + (b0b1)w² + (c0 + c1)w³ + (b0c1 + b1c0)w⁴ + (c0c1)w⁶
        //         = (1 + c0c1 * xi) + (b0 + b1)w + (b0b1)w² + (c0 + c1)w³ + (b0c1 + b1c0)w⁴
        let l0 = Fq2::ONE + c0 * c1 * Bn254::xi();
        let l1 = b0 + b1;
        let l2 = b0 * b1;
        let l3 = c0 + c1;
        let l4 = b0 * c1 + b1 * c0;

        SexticExtField::new([l0, l1, l2, l3, l4, Fq2::ZERO])
    }

    fn mul_by_013(f: Fq12, l: EvaluatedLine<Fq, Fq2>) -> Fq12 {
        Self::mul_by_01234(
            f,
            SexticExtField::new([Fq2::ONE, l.b, Fq2::ZERO, l.c, Fq2::ZERO, Fq2::ZERO]),
        )
    }

    fn mul_by_01234(f: Fq12, x: SexticExtField<Fq2>) -> Fq12 {
        let x_fp12 = Fq12::from_coeffs([x.c[0], x.c[1], x.c[2], x.c[3], x.c[4], Fq2::ZERO]);
        f * x_fp12
    }

    fn evaluate_line(
        l: UnevaluatedLine<Fq, Fq2>,
        x_over_y: Fq,
        y_inv: Fq,
    ) -> EvaluatedLine<Fq, Fq2> {
        EvaluatedLine {
            b: l.b.mul_base(x_over_y),
            c: l.c.mul_base(y_inv),
        }
    }
}

/// Returns a line function for a tangent line at the point P
#[allow(non_snake_case)]
pub fn tangent_line_013<Fp, Fp2>(P: AffinePoint<Fp>) -> EvaluatedLine<Fp, Fp2>
where
    Fp: Field,
    Fp2: FieldExt<BaseField = Fp>,
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
    let c = three_x_cubed * over_two_y_squared - &Fp2::ONE;

    EvaluatedLine { b, c }
}
