use halo2curves_axiom::ff::Field;
use itertools::{izip, Itertools};

use crate::common::{
    fp12_multiply, fp12_square, miller_add_step, miller_double_and_add_step, miller_double_step,
    q_signed, EcPoint, EvaluatedLine, FieldExtension,
};

pub trait MultiMillerLoop<Fp, Fp2, Fp12>
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
    Fp12: FieldExtension<BaseField = Fp2>,
{
    fn negative_x(&self) -> bool;

    fn evaluate_lines_vec(&self, f: Fp12, lines: Vec<EvaluatedLine<Fp, Fp2>>, xi: Fp2) -> Fp12;

    #[allow(non_snake_case)]
    fn multi_miller_loop(
        &self,
        P: &[EcPoint<Fp>],
        Q: &[EcPoint<Fp2>],
        pseudo_binary_encoding: &[i32],
        xi: Fp2,
    ) -> Fp12 {
        self.multi_miller_loop_embedded_exp(P, Q, None, pseudo_binary_encoding, xi)
    }

    #[allow(non_snake_case)]
    fn multi_miller_loop_embedded_exp(
        &self,
        P: &[EcPoint<Fp>],
        Q: &[EcPoint<Fp2>],
        c: Option<Fp12>,
        pseudo_binary_encoding: &[i32],
        xi: Fp2,
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

        let mut f = Fp12::ONE;
        let mut Q_acc = Q.to_vec();

        // Special case the first iteration of the miller loop with pseudo_binary_encoding = 1:
        // this means that the first step is a double and add, but we need to separate the two steps since the optimized
        // `miller_double_and_add_step` will fail because Q_acc is equal to Q_signed on the first iteration
        let (Q_out_double, lines_2S) = Q_acc
            .into_iter()
            .map(miller_double_step::<Fp, Fp2>)
            .unzip::<_, _, Vec<_>, Vec<_>>();
        Q_acc = Q_out_double;

        let mut initial_lines = Vec::<EvaluatedLine<Fp, Fp2>>::new();

        let lines_iter = izip!(lines_2S.iter(), x_over_ys.iter(), y_invs.iter());
        for (line_2S, x_over_y, y_inv) in lines_iter {
            let line = line_2S.evaluate(*x_over_y, *y_inv);
            initial_lines.push(line);
        }

        let (Q_out_add, lines_S_plus_Q) = Q_acc
            .iter()
            .zip(Q.iter())
            .map(|(Q_acc, Q)| miller_add_step::<Fp, Fp2>(Q_acc.clone(), Q.clone()))
            .unzip::<_, _, Vec<_>, Vec<_>>();
        Q_acc = Q_out_add;

        let lines_iter = izip!(lines_S_plus_Q.iter(), x_over_ys.iter(), y_invs.iter());
        for (lines_S_plus_Q, x_over_y, y_inv) in lines_iter {
            let line = lines_S_plus_Q.evaluate(*x_over_y, *y_inv);
            initial_lines.push(line);
        }

        f = self.evaluate_lines_vec(f, initial_lines, xi);

        for i in (0..pseudo_binary_encoding.len() - 2).rev() {
            println!(
                "miller i: {} = {}; Q_acc.x: {:?}",
                i, pseudo_binary_encoding[i], Q_acc[0].x
            );

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

            f = self.evaluate_lines_vec(f, lines, xi);
        }

        if self.negative_x() {
            f = f.conjugate();
        }

        f
    }
}

// #[allow(non_snake_case)]
// pub fn multi_miller_loop<Fp, Fp2, Fp12>(
//     P: &[EcPoint<Fp>],
//     Q: &[EcPoint<Fp2>],
//     pseudo_binary_encoding: &[i32],
//     xi: Fp2,
// ) -> Fp12
// where
//     Fp: Field,
//     Fp2: FieldExtension<BaseField = Fp>,
//     Fp12: FieldExtension<BaseField = Fp2>,
// {
//     multi_miller_loop_embedded_exp::<Fp, Fp2, Fp12>(P, Q, None, pseudo_binary_encoding, xi)
// }

// #[allow(non_snake_case)]
// pub fn multi_miller_loop_embedded_exp<Fp, Fp2, Fp12>(
//     P: &[EcPoint<Fp>],
//     Q: &[EcPoint<Fp2>],
//     c: Option<Fp12>,
//     pseudo_binary_encoding: &[i32],
//     xi: Fp2,
// ) -> Fp12
// where
//     Fp: Field,
//     Fp2: FieldExtension<BaseField = Fp>,
//     Fp12: FieldExtension<BaseField = Fp2>,
// {
//     assert!(!P.is_empty());
//     assert_eq!(P.len(), Q.len());

//     let y_invs = P.iter().map(|P| P.y.invert().unwrap()).collect::<Vec<Fp>>();
//     let x_over_ys = P
//         .iter()
//         .zip(y_invs.iter())
//         .map(|(P, y_inv)| P.x * y_inv)
//         .collect::<Vec<Fp>>();
//     let c_inv = if let Some(c) = c {
//         c.invert().unwrap()
//     } else {
//         Fp12::ONE
//     };

//     let mut f = Fp12::ONE;
//     let mut Q_acc = Q.to_vec();

//     // Special case the first iteration of the miller loop with pseudo_binary_encoding = 1:
//     // this means that the first step is a double and add, but we need to separate the two steps since the optimized
//     // `miller_double_and_add_step` will fail because Q_acc is equal to Q_signed on the first iteration
//     let (Q_out_double, lines_2S) = Q_acc
//         .into_iter()
//         .map(miller_double_step::<Fp, Fp2>)
//         .unzip::<_, _, Vec<_>, Vec<_>>();
//     Q_acc = Q_out_double;

//     let mut initial_lines = Vec::<EvaluatedLine<Fp, Fp2>>::new();

//     let lines_iter = izip!(lines_2S.iter(), x_over_ys.iter(), y_invs.iter());
//     for (line_2S, x_over_y, y_inv) in lines_iter {
//         let line = line_2S.evaluate(*x_over_y, *y_inv);
//         initial_lines.push(line);
//     }

//     let (Q_out_add, lines_S_plus_Q) = Q_acc
//         .iter()
//         .zip(Q.iter())
//         .map(|(Q_acc, Q)| miller_add_step::<Fp, Fp2>(Q_acc.clone(), Q.clone()))
//         .unzip::<_, _, Vec<_>, Vec<_>>();
//     Q_acc = Q_out_add;

//     let lines_iter = izip!(lines_S_plus_Q.iter(), x_over_ys.iter(), y_invs.iter());
//     for (lines_S_plus_Q, x_over_y, y_inv) in lines_iter {
//         let line = lines_S_plus_Q.evaluate(*x_over_y, *y_inv);
//         initial_lines.push(line);
//     }

//     f = evaluate_lines_vec::<Fp, Fp2, Fp12>(f, initial_lines, xi);

//     for i in (0..pseudo_binary_encoding.len() - 2).rev() {
//         println!(
//             "miller i: {} = {}; Q_acc.x: {:?}",
//             i, pseudo_binary_encoding[i], Q_acc[0].x
//         );

//         f = fp12_square::<Fp12>(f);

//         let mut lines = Vec::<EvaluatedLine<Fp, Fp2>>::new();

//         if pseudo_binary_encoding[i] == 0 {
//             // Run miller double step if \sigma_i == 0
//             let (Q_out, lines_2S) = Q_acc
//                 .into_iter()
//                 .map(miller_double_step::<Fp, Fp2>)
//                 .unzip::<_, _, Vec<_>, Vec<_>>();
//             Q_acc = Q_out;

//             let lines_iter = izip!(lines_2S.iter(), x_over_ys.iter(), y_invs.iter());
//             for (line_2S, x_over_y, y_inv) in lines_iter {
//                 let line = line_2S.evaluate(*x_over_y, *y_inv);
//                 lines.push(line);
//             }
//         } else {
//             // use embedded exponent technique if c is provided
//             f = if let Some(c) = c {
//                 match pseudo_binary_encoding[i] {
//                     1 => fp12_multiply(f, c),
//                     -1 => fp12_multiply(f, c_inv),
//                     _ => panic!("Invalid sigma_i"),
//                 }
//             } else {
//                 f
//             };

//             // Run miller double and add if \sigma_i != 0
//             let Q_signed = q_signed(Q, pseudo_binary_encoding[i]);
//             let (Q_out, lines_S_plus_Q, lines_S_plus_Q_plus_S): (Vec<_>, Vec<_>, Vec<_>) = Q_acc
//                 .iter()
//                 .zip(Q_signed.iter())
//                 .map(|(Q_acc, Q_signed)| {
//                     miller_double_and_add_step::<Fp, Fp2>(Q_acc.clone(), Q_signed.clone())
//                 })
//                 .multiunzip();
//             Q_acc = Q_out;

//             let lines_iter = izip!(
//                 lines_S_plus_Q.iter(),
//                 lines_S_plus_Q_plus_S.iter(),
//                 x_over_ys.iter(),
//                 y_invs.iter()
//             );
//             for (line_S_plus_Q, line_S_plus_Q_plus_S, x_over_y, y_inv) in lines_iter {
//                 let line0 = line_S_plus_Q.evaluate(*x_over_y, *y_inv);
//                 let line1 = line_S_plus_Q_plus_S.evaluate(*x_over_y, *y_inv);
//                 lines.push(line0);
//                 lines.push(line1);
//             }
//         };

//         // TODO[yj]: in order to make this miller loop more general, we can either create a new trait that will be applied to
//         // different curves or we can pass in this evaluation function as a parameter
//         f = evaluate_lines_vec::<Fp, Fp2, Fp12>(f, lines, xi);
//     }

//     // We conjugate here f since the x value of BLS12-381 is *negative* 0xd201000000010000
//     // TODO[yj]: we will need to make this more general to support other curves
//     f = f.conjugate();

//     f
// }
