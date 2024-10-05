use halo2curves_axiom::ff::Field;

use crate::common::{EvaluatedLine, FieldExtension};

/// Multiplies two line functions in 023 form and outputs the product in 012345 form
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
    let x1 = Fp2::ZERO;
    let x2 = c0 * b1 + c1 * b0;
    let x3 = c0 + c1;
    let x4 = b0 * b1;
    let x5 = b0 + b1;

    [x0, x1, x2, x3, x4, x5]
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
