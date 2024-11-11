use alloc::vec::Vec;
use core::ops::{Add, Mul, Neg, Sub};

use itertools::{izip, Itertools};

use super::{EvaluatedLine, MillerStep};
use crate::{
    field::{Field, FieldExtension},
    point::AffinePoint,
};

#[allow(non_snake_case)]
pub trait MultiMillerLoop: MillerStep
where
    for<'a> &'a Self::Fp: Add<&'a Self::Fp, Output = Self::Fp>,
    for<'a> &'a Self::Fp: Sub<&'a Self::Fp, Output = Self::Fp>,
    for<'a> &'a Self::Fp: Mul<&'a Self::Fp, Output = Self::Fp>,
    for<'a> &'a Self::Fp2: Add<&'a Self::Fp2, Output = Self::Fp2>,
    for<'a> &'a Self::Fp2: Sub<&'a Self::Fp2, Output = Self::Fp2>,
    for<'a> &'a Self::Fp2: Mul<&'a Self::Fp2, Output = Self::Fp2>,
    for<'a> &'a Self::Fp2: Neg<Output = Self::Fp2>,
    for<'a> &'a Self::Fp12: Mul<&'a Self::Fp12, Output = Self::Fp12>,
{
    type Fp12: FieldExtension<BaseField = Self::Fp2>;

    const SEED_ABS: u64;
    const PSEUDO_BINARY_ENCODING: &[i8];

    /// Function to evaluate the line functions of the Miller loop
    fn evaluate_lines_vec(
        &self,
        f: Self::Fp12,
        lines: Vec<EvaluatedLine<Self::Fp, Self::Fp2>>,
    ) -> Self::Fp12;

    /// Runs before the main loop in the Miller loop function
    fn pre_loop(
        &self,
        f: &Self::Fp12,
        Q_acc: Vec<AffinePoint<Self::Fp2>>,
        Q: &[AffinePoint<Self::Fp2>],
        c: Option<Self::Fp12>,
        x_over_ys: Vec<Self::Fp>,
        y_invs: Vec<Self::Fp>,
    ) -> (Self::Fp12, Vec<AffinePoint<Self::Fp2>>);

    /// Runs after the main loop in the Miller loop function
    fn post_loop(
        &self,
        f: &Self::Fp12,
        Q_acc: Vec<AffinePoint<Self::Fp2>>,
        Q: &[AffinePoint<Self::Fp2>],
        c: Option<Self::Fp12>,
        x_over_ys: Vec<Self::Fp>,
        y_invs: Vec<Self::Fp>,
    ) -> (Self::Fp12, Vec<AffinePoint<Self::Fp2>>);

    /// Runs the multi-Miller loop with no embedded exponent
    #[allow(non_snake_case)]
    fn multi_miller_loop<const PBE_LEN: usize>(
        &self,
        P: &[AffinePoint<Self::Fp>],
        Q: &[AffinePoint<Self::Fp2>],
    ) -> Self::Fp12 {
        self.multi_miller_loop_embedded_exp::<PBE_LEN>(P, Q, None)
    }

    /// Runs the multi-Miller loop with an embedded exponent, removing the need to calculate the residue witness
    /// in the final exponentiation step
    #[allow(non_snake_case)]
    fn multi_miller_loop_embedded_exp<const PBE_LEN: usize>(
        &self,
        P: &[AffinePoint<Self::Fp>],
        Q: &[AffinePoint<Self::Fp2>],
        c: Option<Self::Fp12>,
    ) -> Self::Fp12 {
        assert!(!P.is_empty());
        assert_eq!(P.len(), Q.len());

        let y_invs = P
            .iter()
            .map(|P| P.y.invert().unwrap())
            .collect::<Vec<Self::Fp>>();
        let x_over_ys = P
            .iter()
            .zip(y_invs.iter())
            .map(|(P, y_inv)| &P.x * y_inv)
            .collect::<Vec<Self::Fp>>();
        let c_inv = if let Some(c) = c.clone() {
            c.invert().unwrap()
        } else {
            Self::Fp12::ONE
        };

        let mut f = if let Some(c) = c.clone() {
            c
        } else {
            Self::Fp12::ONE
        };
        let mut Q_acc = Q.to_vec();

        let (f_out, Q_acc_out) =
            self.pre_loop(&f, Q_acc, Q, c.clone(), x_over_ys.clone(), y_invs.clone());
        f = f_out;
        Q_acc = Q_acc_out;

        fn q_signed<Fp, Fp2>(Q: &[AffinePoint<Fp2>], sigma_i: i8) -> Vec<AffinePoint<Fp2>>
        where
            Fp: Field,
            Fp2: FieldExtension<BaseField = Fp>,
        {
            Q.iter()
                .map(|q| match sigma_i {
                    1 => q.clone(),
                    -1 => q.clone().neg_assign(),
                    _ => panic!("Invalid sigma_i"),
                })
                .collect()
        }

        for i in (0..PBE_LEN - 2).rev() {
            f = &f * &f;

            let mut lines = Vec::<EvaluatedLine<Self::Fp, Self::Fp2>>::new();

            if Self::PSEUDO_BINARY_ENCODING[i] == 0 {
                // Run miller double step if \sigma_i == 0
                let (Q_out, lines_2S) = Q_acc
                    .into_iter()
                    .map(Self::miller_double_step)
                    .unzip::<_, _, Vec<_>, Vec<_>>();
                Q_acc = Q_out;

                let lines_iter = izip!(lines_2S.iter(), x_over_ys.iter(), y_invs.iter());
                for (line_2S, x_over_y, y_inv) in lines_iter {
                    let line = line_2S.evaluate(&(x_over_y.clone(), y_inv.clone()));
                    lines.push(line);
                }
            } else {
                // use embedded exponent technique if c is provided
                f = if let Some(c) = c.clone() {
                    match Self::PSEUDO_BINARY_ENCODING[i] {
                        1 => &f * &c,
                        -1 => &f * &c_inv,
                        _ => panic!("Invalid sigma_i"),
                    }
                } else {
                    f
                };

                // Run miller double and add if \sigma_i != 0
                let Q_signed = q_signed(Q, Self::PSEUDO_BINARY_ENCODING[i]);
                let (Q_out, lines_S_plus_Q, lines_S_plus_Q_plus_S): (Vec<_>, Vec<_>, Vec<_>) =
                    Q_acc
                        .iter()
                        .zip(Q_signed.iter())
                        .map(|(Q_acc, Q_signed)| {
                            Self::miller_double_and_add_step(Q_acc.clone(), Q_signed.clone())
                        })
                        .multiunzip();
                Q_acc = Q_out;

                let lines_iter = izip!(
                    lines_S_plus_Q.iter(),
                    lines_S_plus_Q_plus_S.iter(),
                    x_over_ys.iter(),
                    y_invs.iter()
                );
                for (line_S_plus_Q, line_S_plus_Q_plus_S, x_over_y, y_inv) in lines_iter {
                    let bc_vals = (x_over_y.clone(), y_inv.clone());
                    let line0 = line_S_plus_Q.evaluate(&bc_vals);
                    let line1 = line_S_plus_Q_plus_S.evaluate(&bc_vals);
                    lines.push(line0);
                    lines.push(line1);
                }
            };

            f = self.evaluate_lines_vec(f, lines);
        }

        let (f_out, _) = self.post_loop(&f, Q_acc, Q, c, x_over_ys, y_invs);
        f = f_out;

        f
    }
}
