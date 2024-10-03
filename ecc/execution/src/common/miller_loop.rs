use halo2curves_axiom::ff::Field;
use itertools::{izip, Itertools};

use super::UnevaluatedLine;
use crate::{
    common::{fp12_multiply, fp12_square, EcPoint, EvaluatedLine, FieldExtension},
    curves::bls12_381::evaluate_lines_vec,
};

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

#[allow(non_snake_case)]
pub fn multi_miller_loop<Fp, Fp2, Fp12>(
    P: &[EcPoint<Fp>],
    Q: &[EcPoint<Fp2>],
    pseudo_binary_encoding: &[i32],
    xi: Fp2,
) -> Fp12
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
    Fp12: FieldExtension<BaseField = Fp2>,
{
    multi_miller_loop_embedded_exp::<Fp, Fp2, Fp12>(P, Q, None, pseudo_binary_encoding, xi)
}

#[allow(non_snake_case)]
pub fn multi_miller_loop_embedded_exp<Fp, Fp2, Fp12>(
    P: &[EcPoint<Fp>],
    Q: &[EcPoint<Fp2>],
    c: Option<Fp12>,
    pseudo_binary_encoding: &[i32],
    xi: Fp2,
) -> Fp12
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
    Fp12: FieldExtension<BaseField = Fp2>,
{
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
        // let line = evaluate_line::<Fp, Fp2>(*line_2S, *x_over_y, *y_inv);
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
        // let line = evaluate_line::<Fp, Fp2>(*lines_S_plus_Q, *x_over_y, *y_inv);
        let line = lines_S_plus_Q.evaluate(*x_over_y, *y_inv);
        initial_lines.push(line);
    }

    f = evaluate_lines_vec::<Fp, Fp2, Fp12>(f, initial_lines, xi);

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
                // let line = evaluate_line::<Fp, Fp2>(*line_2S, *x_over_y, *y_inv);
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
            let (Q_out, lines_S_plus_Q, lines_S_plus_Q_plus_S): (Vec<_>, Vec<_>, Vec<_>) = Q_acc
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
                // let line0 = evaluate_line::<Fp, Fp2>(*line_S_plus_Q, *x_over_y, *y_inv);
                // let line1 = evaluate_line::<Fp, Fp2>(*line_S_plus_Q_plus_S, *x_over_y, *y_inv);
                let line0 = line_S_plus_Q.evaluate(*x_over_y, *y_inv);
                let line1 = line_S_plus_Q_plus_S.evaluate(*x_over_y, *y_inv);
                lines.push(line0);
                lines.push(line1);
            }
        };

        // TODO[yj]: in order to make this miller loop more general, we can either create a new trait that will be applied to
        // different curves or we can pass in this evaluation function as a parameter
        f = evaluate_lines_vec::<Fp, Fp2, Fp12>(f, lines, xi);
    }

    // We conjugate here f since the x value of BLS12-381 is *negative* 0xd201000000010000
    // TODO[yj]: we will need to make this more general to support other curves
    f = f.conjugate();

    f
}
