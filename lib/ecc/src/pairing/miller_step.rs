use ff::Field;

use super::UnevaluatedLine;
use crate::{field::FieldExtension, point::EcPoint};

#[allow(non_snake_case)]
pub fn miller_double_step<Fp, Fp2>(S: EcPoint<Fp2>) -> (EcPoint<Fp2>, UnevaluatedLine<Fp, Fp2>)
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
{
    let one = Fp2::ONE;
    let two = one + one;
    let three = one + two;

    let x = S.x;
    let y = S.y;
    // λ = (3x^2) / (2y)
    let two_y_inv = (two * y).invert().unwrap();
    let lambda = (three * x.square()) * two_y_inv;
    // x_2S = λ^2 - 2x
    let x_2S = lambda.square() - two * x;
    // y_2S = λ(x - x_2S) - y
    let y_2S = lambda * (x - x_2S) - y;
    let two_s = EcPoint { x: x_2S, y: y_2S };

    // Tangent line
    //   1 + b' (x_P / y_P) w^-1 + c' (1 / y_P) w^-3
    // where
    //   l_{\Psi(S),\Psi(S)}(P) = (λ * x_S - y_S) (1 / y_P)  - λ (x_P / y_P) w^2 + w^3
    // x0 = λ * x_S - y_S
    // x2 = - λ
    let b = -lambda;
    let c = lambda * x - y;

    (two_s, UnevaluatedLine { b, c })
}

#[allow(non_snake_case)]
pub fn miller_add_step<Fp, Fp2>(
    S: EcPoint<Fp2>,
    Q: EcPoint<Fp2>,
) -> (EcPoint<Fp2>, UnevaluatedLine<Fp, Fp2>)
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
{
    let x_s = S.x;
    let y_s = S.y;
    let x_q = Q.x;
    let y_q = Q.y;

    // λ1 = (y_s - y_q) / (x_s - x_q)
    let x_s_minus_x_q_inv = (x_s - x_q).invert().unwrap();
    let lambda = (y_s - y_q) * x_s_minus_x_q_inv;
    let x_s_plus_q = lambda.square() - x_s - x_q;
    let y_s_plus_q = lambda * (x_q - x_s_plus_q) - y_q;

    let s_plus_q = EcPoint {
        x: x_s_plus_q,
        y: y_s_plus_q,
    };

    // l_{\Psi(S),\Psi(Q)}(P) = (λ_1 * x_S - y_S) (1 / y_P) - λ_1 (x_P / y_P) w^2 + w^3
    let b = -lambda;
    let c = lambda * x_s - y_s;

    (s_plus_q, UnevaluatedLine { b, c })
}

#[allow(non_snake_case)]
pub fn miller_double_and_add_step<Fp, Fp2>(
    S: EcPoint<Fp2>,
    Q: EcPoint<Fp2>,
) -> (
    EcPoint<Fp2>,
    UnevaluatedLine<Fp, Fp2>,
    UnevaluatedLine<Fp, Fp2>,
)
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
{
    let one = Fp2::ONE;
    let two = one + one;

    let x_s = S.x;
    let y_s = S.y;
    let x_q = Q.x;
    let y_q = Q.y;

    // λ1 = (y_s - y_q) / (x_s - x_q)
    let lambda1 = (y_s - y_q) * (x_s - x_q).invert().unwrap();
    let x_s_plus_q = lambda1.square() - x_s - x_q;

    // λ2 = -λ1 - 2y_s / (x_{s+q} - x_s)
    let lambda2 = lambda1.neg() - two * y_s * (x_s_plus_q - x_s).invert().unwrap();
    let x_s_plus_q_plus_s = lambda2.square() - x_s - x_s_plus_q;
    let y_s_plus_q_plus_s = lambda2 * (x_s - x_s_plus_q_plus_s) - y_s;

    let s_plus_q_plus_s = EcPoint {
        x: x_s_plus_q_plus_s,
        y: y_s_plus_q_plus_s,
    };

    // l_{\Psi(S),\Psi(Q)}(P) = (λ_1 * x_S - y_S) (1 / y_P) - λ_1 (x_P / y_P) w^2 + w^3
    let b0 = -lambda1;
    let c0 = lambda1 * x_s - y_s;

    // l_{\Psi(S+Q),\Psi(S)}(P) = (λ_2 * x_S - y_S) (1 / y_P) - λ_2 (x_P / y_P) w^2 + w^3
    let b1 = -lambda2;
    let c1 = lambda2 * x_s - y_s;

    (
        s_plus_q_plus_s,
        UnevaluatedLine { b: b0, c: c0 },
        UnevaluatedLine { b: b1, c: c1 },
    )
}
