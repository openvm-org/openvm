use axvm_ecc::{field::FieldExtension, pairing::EvaluatedLine, point::EcPoint};
use halo2curves_axiom::ff::Field;

/// Multiplies two elements in 013 form and outputs the product in 01234 form
pub fn mul_013_by_013<Fp, Fp2>(
    line_0: EvaluatedLine<Fp, Fp2>,
    line_1: EvaluatedLine<Fp, Fp2>,
    // TODO[yj]: once this function is moved into a chip, we can use the xi property instead of passing in this argument
    xi: Fp2,
) -> [Fp2; 5]
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
{
    let b0 = line_0.b;
    let c0 = line_0.c;
    let b1 = line_1.b;
    let c1 = line_1.c;

    // where w⁶ = xi
    // l0 * l1 = 1 + (b0 + b1)w + (b0b1)w² + (c0 + c1)w³ + (b0c1 + b1c0)w⁴ + (c0c1)w⁶
    //         = (1 + c0c1 * xi) + (b0 + b1)w + (b0b1)w² + (c0 + c1)w³ + (b0c1 + b1c0)w⁴
    let l0 = Fp2::ONE + c0 * c1 * xi;
    let l1 = b0 + b1;
    let l2 = b0 * b1;
    let l3 = c0 + c1;
    let l4 = b0 * c1 + b1 * c0;

    [l0, l1, l2, l3, l4]
}

pub fn mul_by_013<Fp, Fp2, Fp12>(f: Fp12, line: EvaluatedLine<Fp, Fp2>) -> Fp12
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
    Fp12: FieldExtension<BaseField = Fp2>,
{
    mul_by_01234(f, [Fp2::ONE, line.b, Fp2::ZERO, line.c, Fp2::ZERO])
}

pub fn mul_by_01234<Fp, Fp2, Fp12>(f: Fp12, x: [Fp2; 5]) -> Fp12
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
pub fn tangent_line_013<Fp, Fp2>(P: EcPoint<Fp>) -> EvaluatedLine<Fp, Fp2>
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
    // 1 - λ(x/y)w + (λx - y)(1/y)w^3
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
