use halo2curves_axiom::ff::Field;

use crate::common::{EcPoint, EvaluatedLine, FieldExtension};

/// Multiplies two line functions in 023 form and outputs the product in 02345 form
pub fn mul_023_by_023<Fp, Fp2>(
    line_0: EvaluatedLine<Fp, Fp2>,
    line_1: EvaluatedLine<Fp, Fp2>,
    // TODO[yj]: once this function is moved into a chip, we can use the xi property instead of passing in this argument
    xi: Fp2,
) -> [Fp2; 6]
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
{
    let b0 = line_0.b;
    let c0 = line_0.c;
    let b1 = line_1.b;
    let c1 = line_1.c;

    // where w⁶ = xi
    // l0 * l1 = c0c1 + (c0b1 + c1b0)w² + (c0 + c1)w³ + (b0b1)w⁴ + (b0 +b1)w⁵ + w⁶
    //         = (c0c1 + xi) + (c0b1 + c1b0)w² + (c0 + c1)w³ + (b0b1)w⁴ + (b0 + b1)w⁵
    let x0 = c0 * c1 + xi;
    let x2 = c0 * b1 + c1 * b0;
    let x3 = c0 + c1;
    let x4 = b0 * b1;
    let x5 = b0 + b1;

    [x0, Fp2::ZERO, x2, x3, x4, x5]
}

pub fn mul_by_023<Fp, Fp2, Fp12>(f: Fp12, line: EvaluatedLine<Fp, Fp2>) -> Fp12
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
    Fp12: FieldExtension<BaseField = Fp2>,
{
    mul_by_02345(
        f,
        [line.c, Fp2::ZERO, line.b, Fp2::ONE, Fp2::ZERO, Fp2::ZERO],
    )
}

pub fn mul_by_02345<Fp, Fp2, Fp12>(f: Fp12, x: [Fp2; 6]) -> Fp12
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
    Fp12: FieldExtension<BaseField = Fp2>,
{
    let x_fp12 = Fp12::from_coeffs(&x);
    f * x_fp12
}

/// Returns a line function for a tangent line at the point P
#[allow(non_snake_case)]
pub fn tangent_line_023<Fp, Fp2>(P: EcPoint<Fp>) -> EvaluatedLine<Fp, Fp2>
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
{
    let one = Fp2::ONE;
    let two = one + one;
    let three = one + two;
    let x = Fp2::embed(&P.x);
    let y = Fp2::embed(&P.y);

    // λ = (3x^2) / (2y)
    // 1 - λ(x/y)w^-1 + (λx - y)(1/y)w^-3
    // = (λx - y)(1/y) - λ(x/y)w^2 + w^3
    //
    // b = -(λ * x / y)
    //   = -3x^3 / 2y^2
    // c = (λ * x - y) / y
    //   = 3x^3/2y^2 - 1
    let x_squared = x.square();
    let x_cubed = x_squared * x;
    let y_squared = y.square();
    let three_x_cubed = three * x_cubed;
    let over_two_y_squared = (two * y_squared).invert().unwrap();

    let b = three_x_cubed.neg() * over_two_y_squared;
    let c = three_x_cubed * over_two_y_squared - Fp2::ONE;

    EvaluatedLine { b, c }
}
