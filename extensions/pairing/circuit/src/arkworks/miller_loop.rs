use ark_ec::short_weierstrass::{Affine, SWCurveConfig};
use ark_ff::Field;

/// Line function used in the Miller loop.
/// Represents coefficients b and c where the line is b * (x_p/y_p) + c * (1/y_p)
#[derive(Clone, Debug)]
pub struct UnevaluatedLine<F> {
    pub b: F,
    pub c: F,
}

/// Miller double step for short Weierstrass curves with a = 0.
/// Returns 2S and the line function tangent to S.
///
/// Assumptions:
/// - S is not the point at infinity
/// - The curve equation is y^2 = x^3 + b (a = 0)
pub fn miller_double_step<P: SWCurveConfig>(
    s: &Affine<P>,
) -> (Affine<P>, UnevaluatedLine<P::BaseField>) {
    let two = P::BaseField::from(2u64);
    let three = P::BaseField::from(3u64);

    let x = &s.x;
    let y = &s.y;

    // λ = (3x^2) / (2y)
    let lambda = (three * x.square()) / (two * y);

    // x_2s = λ^2 - 2x
    let x_2s = lambda.square() - two * x;

    // y_2s = λ(x - x_2s) - y
    let y_2s = lambda * (*x - x_2s) - y;

    let two_s = Affine::<P>::new_unchecked(x_2s, y_2s);

    // Line function: l(P) = b * (x_p/y_p) + c * (1/y_p)
    // where b = -λ and c = λ * x_s - y_s
    let b = -lambda;
    let c = lambda * x - y;

    (two_s, UnevaluatedLine { b, c })
}

/// Miller add step for short Weierstrass curves.
/// Returns S+Q and the line function passing through S and Q.
///
/// Assumptions:
/// - S and Q are not the point at infinity
/// - S != Q and S != -Q (to avoid division by zero)
pub fn miller_add_step<P: SWCurveConfig>(
    s: &Affine<P>,
    q: &Affine<P>,
) -> (Affine<P>, UnevaluatedLine<P::BaseField>) {
    let x_s = &s.x;
    let y_s = &s.y;
    let x_q = &q.x;
    let y_q = &q.y;

    // λ = (y_s - y_q) / (x_s - x_q)
    let lambda = (*y_s - y_q) / (*x_s - x_q);
    let x_s_plus_q = lambda.square() - x_s - x_q;
    let y_s_plus_q = lambda * (*x_q - x_s_plus_q) - y_q;

    let s_plus_q = Affine::<P>::new_unchecked(x_s_plus_q, y_s_plus_q);

    // Line function: l(P) = b * (x_p/y_p) + c * (1/y_p)
    let b = -lambda;
    let c = lambda * x_s - y_s;

    (s_plus_q, UnevaluatedLine { b, c })
}

/// Miller double and add step (2S + Q implemented as S + Q + S for efficiency).
/// Returns 2S+Q and two line functions: one through S and Q, another through S+Q and S.
///
/// Assumptions:
/// - S and Q are not the point at infinity
/// - Q != ±S and (S+Q) != ±S (to avoid division by zero)
///
/// This is more efficient than calling miller_add_step twice because it reuses
/// intermediate computations.
#[allow(clippy::type_complexity)]
pub fn miller_double_and_add_step<P: SWCurveConfig>(
    s: &Affine<P>,
    q: &Affine<P>,
) -> (
    Affine<P>,
    UnevaluatedLine<P::BaseField>,
    UnevaluatedLine<P::BaseField>,
) {
    let two = P::BaseField::from(2u64);

    let x_s = &s.x;
    let y_s = &s.y;
    let x_q = &q.x;
    let y_q = &q.y;

    // First add: S + Q
    // λ1 = (y_s - y_q) / (x_s - x_q)
    let lambda1 = (*y_s - y_q) / (*x_s - x_q);
    let x_s_plus_q = lambda1.square() - x_s - x_q;

    // Second add: (S + Q) + S = 2S + Q
    // λ2 = -λ1 - 2y_s / (x_{s+q} - x_s)
    let lambda2 = -lambda1 - (two * y_s) / (x_s_plus_q - x_s);
    let x_s_plus_q_plus_s = lambda2.square() - x_s - x_s_plus_q;
    let y_s_plus_q_plus_s = lambda2 * (*x_s - x_s_plus_q_plus_s) - y_s;

    let s_plus_q_plus_s = Affine::<P>::new_unchecked(x_s_plus_q_plus_s, y_s_plus_q_plus_s);

    // Line function for S + Q
    let b0 = -lambda1;
    let c0 = lambda1 * x_s - y_s;

    // Line function for (S+Q) + S
    let b1 = -lambda2;
    let c1 = lambda2 * x_s - y_s;

    (
        s_plus_q_plus_s,
        UnevaluatedLine { b: b0, c: c0 },
        UnevaluatedLine { b: b1, c: c1 },
    )
}
