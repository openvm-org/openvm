use core::ops::{Add, Mul, Neg, Sub};

use openvm_algebra_guest::{DivUnsafe, Field};
use openvm_ecc_guest::AffinePoint;

use super::{PairingIntrinsics, UnevaluatedLine};

/// Trait definition for Miller step opcodes
pub trait MillerStep {
    type Fp2;

    /// Miller double step
    fn miller_double_step(
        s: &AffinePoint<Self::Fp2>,
    ) -> (AffinePoint<Self::Fp2>, UnevaluatedLine<Self::Fp2>);

    /// Miller add step
    fn miller_add_step(
        s: &AffinePoint<Self::Fp2>,
        q: &AffinePoint<Self::Fp2>,
    ) -> (AffinePoint<Self::Fp2>, UnevaluatedLine<Self::Fp2>);

    /// Miller double and add step (2S + Q implemented as S + Q + S for efficiency)
    #[allow(clippy::type_complexity)]
    fn miller_double_and_add_step(
        s: &AffinePoint<Self::Fp2>,
        q: &AffinePoint<Self::Fp2>,
    ) -> (
        AffinePoint<Self::Fp2>,
        UnevaluatedLine<Self::Fp2>,
        UnevaluatedLine<Self::Fp2>,
    );
}

impl<P> MillerStep for P
where
    P: PairingIntrinsics,
    for<'a> &'a P::Fp2: Add<&'a P::Fp2, Output = P::Fp2>,
    for<'a> &'a P::Fp2: Sub<&'a P::Fp2, Output = P::Fp2>,
    for<'a> &'a P::Fp2: Mul<&'a P::Fp2, Output = P::Fp2>,
    for<'a> &'a P::Fp2: Neg<Output = P::Fp2>,
{
    type Fp2 = <P as PairingIntrinsics>::Fp2;

    /// Miller double step.
    /// Returns 2S and a line in Fp12 tangent to \Psi(S).
    /// Assumptions:
    ///     - s is not point at infinity.
    ///     - a in the curve equation is 0.
    /// The case y = 0 does not happen as long as the curve satisfies that 0 = X^3 + b has no
    /// solutions in Fp2. The curve G1Affine and twist G2Affine are both chosen for bn254,
    /// bls12_381 so that this never happens.
    fn miller_double_step(
        s: &AffinePoint<Self::Fp2>,
    ) -> (AffinePoint<Self::Fp2>, UnevaluatedLine<Self::Fp2>) {
        let two: &Self::Fp2 = &<P as PairingIntrinsics>::FP2_TWO;
        let three: &Self::Fp2 = &<P as PairingIntrinsics>::FP2_THREE;

        let x = &s.x;
        let y = &s.y;
        // λ = (3x^2) / (2y)
        let lambda = &((three * x * x).div_unsafe(&(two * y)));
        // x_2s = λ^2 - 2x
        let x_2s = lambda * lambda - two * x;
        // y_2s = λ(x - x_2s) - y
        let y_2s = lambda * &(x - &x_2s) - y;
        let two_s = AffinePoint { x: x_2s, y: y_2s };

        // l_{\Psi(S),\Psi(S)}(P)
        let b = Self::Fp2::ZERO - lambda;
        let c = lambda * x - y;

        (two_s, UnevaluatedLine { b, c })
    }

    /// Miller add step.
    /// Returns S+Q and a line in Fp12 passing through \Psi(S) and \Psi(Q).
    fn miller_add_step(
        s: &AffinePoint<Self::Fp2>,
        q: &AffinePoint<Self::Fp2>,
    ) -> (AffinePoint<Self::Fp2>, UnevaluatedLine<Self::Fp2>) {
        let x_s = &s.x;
        let y_s = &s.y;
        let x_q = &q.x;
        let y_q = &q.y;

        // λ1 = (y_s - y_q) / (x_s - x_q)
        let x_delta = x_s - x_q;
        let lambda = &((y_s - y_q).div_unsafe(&x_delta));
        let x_s_plus_q = lambda * lambda - x_s - x_q;
        let y_s_plus_q = lambda * &(x_q - &x_s_plus_q) - y_q;

        let s_plus_q = AffinePoint {
            x: x_s_plus_q,
            y: y_s_plus_q,
        };

        // l_{\Psi(S),\Psi(Q)}(P)
        let b = Self::Fp2::ZERO - lambda;
        let c = lambda * x_s - y_s;

        (s_plus_q, UnevaluatedLine { b, c })
    }

    /// Miller double and add step (2S + Q implemented as S + Q + S for efficiency).
    /// Returns 2S+Q, a line in Fp12 passing through S and Q, and a line in Fp12 passing through S+Q
    /// and S Assumption: Q != +- S && (S+Q) != +-S, so that there is no division by zero.
    /// The way this is used in miller loop, this is always satisfied.
    fn miller_double_and_add_step(
        s: &AffinePoint<Self::Fp2>,
        q: &AffinePoint<Self::Fp2>,
    ) -> (
        AffinePoint<Self::Fp2>,
        UnevaluatedLine<Self::Fp2>,
        UnevaluatedLine<Self::Fp2>,
    ) {
        let two = &Self::FP2_TWO;

        let x_s = &s.x;
        let y_s = &s.y;
        let x_q = &q.x;
        let y_q = &q.y;

        // λ1 = (y_s - y_q) / (x_s - x_q)
        let lambda1 = &((y_s - y_q).div_unsafe(&(x_s - x_q)));
        let x_s_plus_q = lambda1 * lambda1 - x_s - x_q;

        // λ2 = -λ1 - 2y_s / (x_{s+q} - x_s)
        let lambda2 =
            &(Self::Fp2::ZERO - lambda1.clone() - (two * y_s).div_unsafe(&(&x_s_plus_q - x_s)));
        let x_s_plus_q_plus_s = lambda2 * lambda2 - x_s - &x_s_plus_q;
        let y_s_plus_q_plus_s = lambda2 * &(x_s - &x_s_plus_q_plus_s) - y_s;

        let s_plus_q_plus_s = AffinePoint {
            x: x_s_plus_q_plus_s,
            y: y_s_plus_q_plus_s,
        };

        // l_{\Psi(S),\Psi(Q)}(P)
        let b0 = Self::Fp2::ZERO - lambda1;
        let c0 = lambda1 * x_s - y_s;

        // l_{\Psi(S+Q),\Psi(S)}(P)
        let b1 = Self::Fp2::ZERO - lambda2;
        let c1 = lambda2 * x_s - y_s;

        (
            s_plus_q_plus_s,
            UnevaluatedLine { b: b0, c: c0 },
            UnevaluatedLine { b: b1, c: c1 },
        )
    }
}
