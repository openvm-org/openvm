use alloc::vec::Vec;
use core::{
    iter::zip,
    ops::{Mul, Neg},
};

use openvm_algebra_guest::{field::FieldExtension, DivUnsafe, Field};
use openvm_ecc_guest::AffinePoint;

use super::{Evaluatable, EvaluatedLine, MillerStep, UnevaluatedLine};

#[allow(non_snake_case)]
pub trait MultiMillerLoop: MillerStep
where
    <Self as MillerStep>::Fp2: Field + FieldExtension<Self::Fp>,
    // these trait bounds are needed for `multi_miller_loop_embedded_exp`. It would be better to
    // move into a macro so the trait stays clean
    UnevaluatedLine<Self::Fp2>: Evaluatable<Self::Fp, Self::Fp2>,
    for<'a> &'a Self::Fp: DivUnsafe<&'a Self::Fp, Output = Self::Fp>,
    for<'a> &'a Self::Fp2: Neg<Output = Self::Fp2>,
    for<'a> &'a Self::Fp12: Mul<&'a Self::Fp12, Output = Self::Fp12>,
    for<'a> &'a Self::Fp12: DivUnsafe<&'a Self::Fp12, Output = Self::Fp12>,
{
    type Fp: Field;
    type Fp12: Field + FieldExtension<Self::Fp2>;

    const SEED_ABS: u64;
    const PSEUDO_BINARY_ENCODING: &[i8];

    /// Function to evaluate the line functions of the Miller loop
    fn evaluate_lines_vec(f: Self::Fp12, lines: &[EvaluatedLine<Self::Fp2>]) -> Self::Fp12;

    /// Runs before the main loop in the Miller loop function
    ///
    /// xy_fracs consists of (x/y, 1/y) pairs for each point P
    fn pre_loop(
        Q_acc: Vec<AffinePoint<Self::Fp2>>,
        Q: &[AffinePoint<Self::Fp2>],
        c: Option<Self::Fp12>,
        xy_fracs: &[(Self::Fp, Self::Fp)],
    ) -> (Self::Fp12, Vec<AffinePoint<Self::Fp2>>);

    /// Runs after the main loop in the Miller loop function
    fn post_loop(
        f: &Self::Fp12,
        Q_acc: Vec<AffinePoint<Self::Fp2>>,
        Q: &[AffinePoint<Self::Fp2>],
        c: Option<Self::Fp12>,
        xy_fracs: &[(Self::Fp, Self::Fp)],
    ) -> (Self::Fp12, Vec<AffinePoint<Self::Fp2>>);

    /// Runs the multi-Miller loop with no embedded exponent
    #[allow(non_snake_case)]
    fn multi_miller_loop(P: &[AffinePoint<Self::Fp>], Q: &[AffinePoint<Self::Fp2>]) -> Self::Fp12 {
        Self::multi_miller_loop_embedded_exp(P, Q, None)
    }

    /// Runs the multi-Miller loop with an embedded exponent, removing the need to calculate the
    /// residue witness in the final exponentiation step.
    ///
    /// `c` is assumed nonzero.
    ///
    /// **Note:** All points in `P` must have non-zero `y` coordinate. Points with `y=0`
    /// are not on BN254/BLS12-381, but `AffinePoint` does not enforce on-curve membership.
    /// Passing such points would cause `div_unsafe` on zero, producing unconstrained values.
    fn multi_miller_loop_embedded_exp(
        P: &[AffinePoint<Self::Fp>],
        Q: &[AffinePoint<Self::Fp2>],
        c: Option<Self::Fp12>,
    ) -> Self::Fp12 {
        assert!(!P.is_empty());
        assert_eq!(P.len(), Q.len());

        // Filter out the pair with infinity points
        let (P, Q): (Vec<_>, Vec<_>) = zip(P, Q)
            .filter(|(p, q)| !p.is_infinity() && !q.is_infinity())
            .map(|(p, q)| (p.clone(), q.clone()))
            .unzip();

        let xy_fracs = P
            .iter()
            .map(|P| ((&P.x).div_unsafe(&P.y), (&Self::Fp::ONE).div_unsafe(&P.y)))
            .collect::<Vec<(Self::Fp, Self::Fp)>>();
        let c_inv = if let Some(c) = c.as_ref() {
            (&Self::Fp12::ONE).div_unsafe(c)
        } else {
            Self::Fp12::ONE
        };

        let mut Q_acc = Q.to_vec();

        let (f_out, Q_acc_out) = Self::pre_loop(Q_acc, &Q, c.clone(), &xy_fracs);
        let mut f = f_out;
        Q_acc = Q_acc_out;

        // One buffer holds every iteration's lines (two per pair on a double-and-add step, one
        // on a double step), and `Q_acc` is advanced in place, so an iteration allocates
        // nothing. Fresh vectors per iteration are freed right away by a reclaiming allocator,
        // but a guest on a bump allocator never gets them back: its heap would grow with every
        // iteration of every pairing.
        let mut lines = Vec::with_capacity(2 * xy_fracs.len());

        for i in (0..Self::PSEUDO_BINARY_ENCODING.len() - 2).rev() {
            f.square_assign();
            lines.clear();

            if Self::PSEUDO_BINARY_ENCODING[i] == 0 {
                // Run miller double step if \sigma_i == 0
                for (Q_acc, xy_frac) in Q_acc.iter_mut().zip(&xy_fracs) {
                    let (Q_out, line_2S) = Self::miller_double_step(Q_acc);
                    *Q_acc = Q_out;
                    lines.push(line_2S.evaluate(xy_frac));
                }
            } else {
                // use embedded exponent technique if c is provided
                f = if let Some(c) = c.as_ref() {
                    match Self::PSEUDO_BINARY_ENCODING[i] {
                        1 => &f * c,
                        -1 => &f * &c_inv,
                        _ => panic!("Invalid sigma_i"),
                    }
                } else {
                    f
                };

                // Run miller double and add if \sigma_i != 0
                for ((Q_acc, q), xy_frac) in Q_acc.iter_mut().zip(&Q).zip(&xy_fracs) {
                    // OPT[jpw]: cache the neg q outside of the loop
                    let q_signed = match Self::PSEUDO_BINARY_ENCODING[i] {
                        1 => q,
                        -1 => &q.neg_borrow(),
                        _ => panic!("Invalid sigma_i"),
                    };
                    let (Q_out, line_S_plus_Q, line_S_plus_Q_plus_S) =
                        Self::miller_double_and_add_step(Q_acc, q_signed);
                    *Q_acc = Q_out;
                    lines.push(line_S_plus_Q.evaluate(xy_frac));
                    lines.push(line_S_plus_Q_plus_S.evaluate(xy_frac));
                }
            };

            f = Self::evaluate_lines_vec(f, &lines);
        }

        let (f_out, _) = Self::post_loop(&f, Q_acc, &Q, c, &xy_fracs);
        f = f_out;

        f
    }
}
