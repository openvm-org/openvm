use std::ops::{Add, Mul, Neg, Sub};

use axvm_ecc::{
    curve::bls12381::{Fq, Fq12, Fq2},
    field::{Field, FieldExtension},
    pairing::{EvaluatedLine, LineMulMType},
    point::AffinePoint,
};

use super::Bls12_381;

impl LineMulMType<Fq, Fq2, Fq12> for Bls12_381 {}

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
    let one = &Fp2::one();
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
    let c = three_x_cubed * over_two_y_squared - &Fp2::one();

    EvaluatedLine { b, c }
}
