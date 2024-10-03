use halo2curves_axiom::ff::Field;

use crate::common::{EcPoint, FieldExtension, Fp12Constructor};

pub fn conv_013_to_fp12<Fp, Fp2, Fp12>(line: [Fp2; 2]) -> Fp12
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
    Fp12: FieldExtension<BaseField = Fp2> + Fp12Constructor<Fp2>,
{
    // x0 + x1*w + x2*w^2 + x3*w^3 + x4*w^4 + x5*w^5
    // (x0 + x2*w^2 + x4*w^4) + (x1 + x3*w^2 + x5*w^4)*w
    let x0 = Fp2::ONE;
    let x1 = line[0];
    let x2 = Fp2::ZERO;
    let x3 = line[1];
    let x4 = Fp2::ZERO;
    let x5 = Fp2::ZERO;

    Fp12::new(x0, x2, x4, x1, x3, x5)
}

pub fn conv_fp2_coeffs_to_fp12<Fp, Fp2, Fp12>(fp2_coeffs: &[Fp2]) -> Fp12
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
    Fp12: FieldExtension<BaseField = Fp2> + Fp12Constructor<Fp2>,
{
    assert!(
        fp2_coeffs.len() <= 6,
        "fp2_coeffs must have at most 6 elements"
    );
    let mut coeffs = fp2_coeffs.to_vec();
    coeffs.resize(6, Fp2::ZERO);

    let x0 = coeffs[0];
    let x1 = coeffs[1];
    let x2 = coeffs[2];
    let x3 = coeffs[3];
    let x4 = coeffs[4];
    let x5 = coeffs[5];

    Fp12::new(x0, x2, x4, x1, x3, x5)
}

/// Returns a line function for a tangent line at the point P
#[allow(non_snake_case)]
pub fn point_to_013<Fp, Fp2>(P: EcPoint<Fp>) -> [Fp2; 2]
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

    [b, c]
}

#[allow(non_snake_case)]
pub fn q_signed<Fp, Fp2>(Q: &[EcPoint<Fp2>], sigma_i: i32) -> Vec<EcPoint<Fp2>>
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
{
    Q.iter()
        .map(|q| match sigma_i {
            1 => q.clone(),
            -1 => q.neg(),
            _ => panic!("Invalid sigma_i"),
        })
        .collect()
}

pub fn fp12_square<Fp12: Field>(x: Fp12) -> Fp12 {
    fp12_multiply(x, x)
}

pub fn fp12_multiply<Fp12: Field>(x: Fp12, y: Fp12) -> Fp12 {
    x * y
}
