use halo2curves_axiom::ff::Field;
use itertools::{izip, Itertools};

use crate::common::{
    fp12_multiply, fp12_square, miller_double_and_add_step, miller_double_step, q_signed, EcPoint,
    EvaluatedLine, FieldExtension,
};

#[allow(non_snake_case)]
pub trait MultiMillerLoop<Fp, Fp2, Fp12, const BITS: usize>
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
    Fp12: FieldExtension<BaseField = Fp2>,
{
    /// We use the field extension tower `Fp12 = Fp2[w]/(w^6 - xi)`.
    fn xi() -> Fp2;

    /// Seed value for the curve
    fn seed() -> u64;

    /// Pseudo-binary used for the loop counter of the curve
    fn pseudo_binary_encoding() -> [i8; BITS];

    /// Function to evaluate the line functions of the Miller loop
    fn evaluate_lines_vec(&self, f: Fp12, lines: Vec<EvaluatedLine<Fp, Fp2>>) -> Fp12;

    /// Runs before the main loop in the Miller loop function
    fn pre_loop(
        &self,
        f: Fp12,
        Q_acc: Vec<EcPoint<Fp2>>,
        Q: &[EcPoint<Fp2>],
        x_over_ys: Vec<Fp>,
        y_invs: Vec<Fp>,
    ) -> (Fp12, Vec<EcPoint<Fp2>>);

    /// Runs after the main loop in the Miller loop function
    fn post_loop(
        &self,
        f: Fp12,
        Q_acc: Vec<EcPoint<Fp2>>,
        Q: &[EcPoint<Fp2>],
        x_over_ys: Vec<Fp>,
        y_invs: Vec<Fp>,
    ) -> (Fp12, Vec<EcPoint<Fp2>>);

    /// Runs the multi-Miller loop with no embedded exponent
    #[allow(non_snake_case)]
    fn multi_miller_loop(&self, P: &[EcPoint<Fp>], Q: &[EcPoint<Fp2>]) -> Fp12 {
        self.multi_miller_loop_embedded_exp(P, Q, None)
    }

    /// Runs the multi-Miller loop with an embedded exponent, removing the need to calculate the residue witness
    /// in the final exponentiation step
    #[allow(non_snake_case)]
    fn multi_miller_loop_embedded_exp(
        &self,
        P: &[EcPoint<Fp>],
        Q: &[EcPoint<Fp2>],
        c: Option<Fp12>,
    ) -> Fp12 {
        assert!(!P.is_empty());
        assert_eq!(P.len(), Q.len());

        let y_invs = P.iter().map(|P| P.y.invert().unwrap()).collect::<Vec<Fp>>();
        let x_over_ys = P
            .iter()
            .zip(y_invs.iter())
            .map(|(P, y_inv)| P.x * y_inv)
            .collect::<Vec<Fp>>();
        let c_inv = if let Some(c) = c {
            c.invert().unwrap()
        } else {
            Fp12::ONE
        };

        let mut f = if let Some(c) = c { c } else { Fp12::ONE };
        let mut Q_acc = Q.to_vec();

        let (f_out, Q_acc_out) = self.pre_loop(f, Q_acc, Q, x_over_ys.clone(), y_invs.clone());
        f = f_out;
        Q_acc = Q_acc_out;

        let pseudo_binary_encoding = Self::pseudo_binary_encoding();
        for i in (0..pseudo_binary_encoding.len() - 2).rev() {
            f = fp12_square::<Fp12>(f);

            let mut lines = Vec::<EvaluatedLine<Fp, Fp2>>::new();

            if pseudo_binary_encoding[i] == 0 {
                // Run miller double step if \sigma_i == 0
                let (Q_out, lines_2S) = Q_acc
                    .into_iter()
                    .map(miller_double_step::<Fp, Fp2>)
                    .unzip::<_, _, Vec<_>, Vec<_>>();
                Q_acc = Q_out;

                let lines_iter = izip!(lines_2S.iter(), x_over_ys.iter(), y_invs.iter());
                for (line_2S, x_over_y, y_inv) in lines_iter {
                    let line = line_2S.evaluate(*x_over_y, *y_inv);
                    lines.push(line);
                }
            } else {
                // use embedded exponent technique if c is provided
                f = if let Some(c) = c {
                    match pseudo_binary_encoding[i] {
                        1 => fp12_multiply(f, c),
                        -1 => fp12_multiply(f, c_inv),
                        _ => panic!("Invalid sigma_i"),
                    }
                } else {
                    f
                };

                // Run miller double and add if \sigma_i != 0
                let Q_signed = q_signed(Q, pseudo_binary_encoding[i]);
                let (Q_out, lines_S_plus_Q, lines_S_plus_Q_plus_S): (Vec<_>, Vec<_>, Vec<_>) =
                    Q_acc
                        .iter()
                        .zip(Q_signed.iter())
                        .map(|(Q_acc, Q_signed)| {
                            miller_double_and_add_step::<Fp, Fp2>(Q_acc.clone(), Q_signed.clone())
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
                    let line0 = line_S_plus_Q.evaluate(*x_over_y, *y_inv);
                    let line1 = line_S_plus_Q_plus_S.evaluate(*x_over_y, *y_inv);
                    lines.push(line0);
                    lines.push(line1);
                }
            };

            f = self.evaluate_lines_vec(f, lines);
        }

        let (f_out, _) = self.post_loop(f, Q_acc, Q, x_over_ys, y_invs);
        f = f_out;

        f
    }
}
