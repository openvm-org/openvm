use core::ops::{Add, Mul, Neg, Sub};

use super::UnevaluatedLine;
use crate::{
    field::{Field, FieldExtension},
    point::AffinePoint,
};

/// Trait definition for Miller step opcodes
pub trait MillerStep
where
    for<'a> &'a Self::Fp2: Add<&'a Self::Fp2, Output = Self::Fp2>,
    for<'a> &'a Self::Fp2: Sub<&'a Self::Fp2, Output = Self::Fp2>,
    for<'a> &'a Self::Fp2: Mul<&'a Self::Fp2, Output = Self::Fp2>,
    for<'a> &'a Self::Fp2: Neg<Output = Self::Fp2>,
{
    type Fp: Field;
    type Fp2: FieldExtension<BaseField = Self::Fp>;

    /// Miller double step
    #[allow(clippy::type_complexity)]
    fn miller_double_step(
        s: AffinePoint<Self::Fp2>,
    ) -> (AffinePoint<Self::Fp2>, UnevaluatedLine<Self::Fp, Self::Fp2>) {
        #[cfg(not(target_os = "zkvm"))]
        {
            let one = &Self::Fp2::ONE;
            let two = &(one + one);
            let three = &(one + two);

            let x = &s.x;
            let y = &s.y;
            // λ = (3x^2) / (2y)
            let two_y_inv = &(two * y).invert().unwrap();
            let lambda = &((three * x * x) * two_y_inv);
            // x_2s = λ^2 - 2x
            let x_2s = lambda * lambda - two * x;
            // y_2s = λ(x - x_2s) - y
            let y_2s = lambda * &(x - &x_2s) - y;
            let two_s = AffinePoint { x: x_2s, y: y_2s };

            // Tangent line
            //   1 + b' (x_P / y_P) w^-1 + c' (1 / y_P) w^-3
            // where
            //   l_{\Psi(S),\Psi(S)}(P) = (λ * x_S - y_S) (1 / y_P)  - λ (x_P / y_P) w^2 + w^3
            // x0 = λ * x_S - y_S
            // x2 = - λ
            let b = lambda.clone().neg();
            let c = lambda * x - y;

            (two_s, UnevaluatedLine { b, c })
        }
        #[cfg(target_os = "zkvm")]
        {
            todo!()
        }
    }

    /// Miller add step
    #[allow(clippy::type_complexity)]
    fn miller_add_step(
        s: AffinePoint<Self::Fp2>,
        q: AffinePoint<Self::Fp2>,
    ) -> (AffinePoint<Self::Fp2>, UnevaluatedLine<Self::Fp, Self::Fp2>) {
        let x_s = &s.x;
        let y_s = &s.y;
        let x_q = &q.x;
        let y_q = &q.y;

        // λ1 = (y_s - y_q) / (x_s - x_q)
        let x_s_minus_x_q_inv = &(x_s - x_q).invert().unwrap();
        let lambda = &((y_s - y_q) * x_s_minus_x_q_inv);
        let x_s_plus_q = lambda * lambda - x_s - x_q;
        let y_s_plus_q = lambda * &(x_q - &x_s_plus_q) - y_q;

        let s_plus_q = AffinePoint {
            x: x_s_plus_q,
            y: y_s_plus_q,
        };

        // l_{\Psi(S),\Psi(Q)}(P) = (λ_1 * x_S - y_S) (1 / y_P) - λ_1 (x_P / y_P) w^2 + w^3
        let b = lambda.clone().neg();
        let c = lambda * x_s - y_s;

        (s_plus_q, UnevaluatedLine { b, c })
    }

    /// Miller double and add step (2S + Q implemented as S + Q + S for efficiency)
    #[allow(clippy::type_complexity)]
    fn miller_double_and_add_step(
        s: AffinePoint<Self::Fp2>,
        q: AffinePoint<Self::Fp2>,
    ) -> (
        AffinePoint<Self::Fp2>,
        UnevaluatedLine<Self::Fp, Self::Fp2>,
        UnevaluatedLine<Self::Fp, Self::Fp2>,
    ) {
        #[cfg(not(target_os = "zkvm"))]
        {
            let one = &Self::Fp2::ONE;
            let two = &(one + one);

            let x_s = &s.x;
            let y_s = &s.y;
            let x_q = &q.x;
            let y_q = &q.y;

            // λ1 = (y_s - y_q) / (x_s - x_q)
            let lambda1 = &((y_s - y_q) * (x_s - x_q).invert().unwrap());
            let x_s_plus_q = lambda1 * lambda1 - x_s - x_q;

            // λ2 = -λ1 - 2y_s / (x_{s+q} - x_s)
            let lambda2 =
                &(lambda1.clone().neg() - two * y_s * (&x_s_plus_q - x_s).invert().unwrap());
            let x_s_plus_q_plus_s = lambda2 * lambda2 - x_s - &x_s_plus_q;
            let y_s_plus_q_plus_s = lambda2 * &(x_s - &x_s_plus_q_plus_s) - y_s;

            let s_plus_q_plus_s = AffinePoint {
                x: x_s_plus_q_plus_s,
                y: y_s_plus_q_plus_s,
            };

            // l_{\Psi(S),\Psi(Q)}(P) = (λ_1 * x_S - y_S) (1 / y_P) - λ_1 (x_P / y_P) w^2 + w^3
            let b0 = lambda1.clone().neg();
            let c0 = lambda1 * x_s - y_s;

            // l_{\Psi(S+Q),\Psi(S)}(P) = (λ_2 * x_S - y_S) (1 / y_P) - λ_2 (x_P / y_P) w^2 + w^3
            let b1 = lambda2.clone().neg();
            let c1 = lambda2 * x_s - y_s;

            (
                s_plus_q_plus_s,
                UnevaluatedLine { b: b0, c: c0 },
                UnevaluatedLine { b: b1, c: c1 },
            )
        }
        #[cfg(target_os = "zkvm")]
        {
            todo!()
        }
    }
}
