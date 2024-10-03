use halo2curves_axiom::ff::Field;

use crate::common::FieldExtension;

/// Multiplies two line functions in 023 form and outputs the product in 012345 form
pub fn mul_023_by_023<Fp, Fp2>(
    line_0: [Fp2; 2],
    line_1: [Fp2; 2],
    // TODO[yj]: once this function is moved into a chip, we can use the xi property instead of passing in this argument
    xi: Fp2,
) -> [Fp2; 6]
where
    Fp: Field,
    Fp2: FieldExtension<2, BaseField = Fp>,
{
    let c0 = line_0[0];
    let b0 = line_0[1];
    let c1 = line_1[0];
    let b1 = line_1[1];

    // where w⁶ = xi
    // l0 * l1 = b0b1 + (b0c1 + b1c0)w² + (b0 + b1)w³ + (c0c1)w⁴ + (c0 +c1)w⁵ + w⁶
    //         = (b0b1 + xi) + (b0c1 + b1c0)w² + (b0 + b1)w³ + (c0c1)w⁴ + (c0 + c1)w⁵
    let x0 = b0 * b1 + xi;
    let x1 = Fp2::ZERO;
    let x2 = b0 * c1 + b1 * c0;
    let x3 = b0 + b1;
    let x4 = c0 * c1;
    let x5 = c0 + c1;

    [x0, x1, x2, x3, x4, x5]
}

pub fn mul_by_023<Fp, Fp2, Fp12>(f: Fp12, line: [Fp2; 2]) -> Fp12
where
    Fp: Field,
    Fp2: FieldExtension<2, BaseField = Fp>,
    Fp12: FieldExtension<6, BaseField = Fp2>,
{
    mul_by_02345(
        f,
        [line[1], Fp2::ZERO, line[0], Fp2::ONE, Fp2::ZERO, Fp2::ZERO],
    )
}

pub fn mul_by_02345<Fp, Fp2, Fp12>(f: Fp12, x: [Fp2; 6]) -> Fp12
where
    Fp: Field,
    Fp2: FieldExtension<2, BaseField = Fp>,
    Fp12: FieldExtension<6, BaseField = Fp2>,
{
    let x_fp12 = Fp12::from_coeffs(x);
    f * x_fp12
}

pub fn evaluate_lines_vec<Fp, Fp2, Fp12>(mut f: Fp12, mut lines: Vec<[Fp2; 2]>, xi: Fp2) -> Fp12
where
    Fp: Field,
    Fp2: FieldExtension<2, BaseField = Fp>,
    Fp12: FieldExtension<6, BaseField = Fp2>,
{
    if lines.len() % 2 == 1 {
        f = mul_by_023::<Fp, Fp2, Fp12>(f, lines.pop().unwrap());
    }
    for chunk in lines.chunks(2) {
        if let [line0, line1] = chunk {
            let prod = mul_023_by_023(*line0, *line1, xi);
            f = mul_by_02345(f, prod);
        } else {
            panic!("lines.len() % 2 should be 0 at this point");
        }
    }
    f
}
