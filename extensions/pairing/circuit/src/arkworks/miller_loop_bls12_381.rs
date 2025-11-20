use ark_bls12_381::{Fq, Fq12, Fq2, Fq6, G1Affine, G2Affine};
use ark_ec::AffineRepr;
use ark_ff::{Field, Zero};

use crate::arkworks::miller_loop::{
    miller_add_step, miller_double_and_add_step, miller_double_step,
};
use openvm_pairing_guest::halo2curves_shims::bls12_381::miller_loop::BLS12_381_PSEUDO_BINARY_ENCODING;

#[derive(Clone, Debug)]
struct EvaluatedLine {
    b: Fq2,
    c: Fq2,
}

fn mul_fp2_by_fp(value: &Fq2, scalar: &Fq) -> Fq2 {
    Fq2::new(value.c0 * scalar, value.c1 * scalar)
}

fn xi() -> Fq2 {
    Fq2::new(Fq::ONE, Fq::ONE)
}

fn fq12_from_coeffs(coeffs: [Fq2; 6]) -> Fq12 {
    Fq12::new(
        Fq6::new(coeffs[0], coeffs[2], coeffs[4]),
        Fq6::new(coeffs[1], coeffs[3], coeffs[5]),
    )
}

fn evaluate_lines_vec(mut f: Fq12, mut lines: Vec<EvaluatedLine>) -> Fq12 {
    fn mul_by_023(f: &Fq12, line: &EvaluatedLine) -> Fq12 {
        let coeffs = [
            line.c,
            Fq2::zero(),
            line.b,
            Fq2::ONE,
            Fq2::zero(),
            Fq2::zero(),
        ];
        *f * fq12_from_coeffs(coeffs)
    }

    fn mul_023_by_023(line0: &EvaluatedLine, line1: &EvaluatedLine) -> [Fq2; 5] {
        let b0 = &line0.b;
        let c0 = &line0.c;
        let b1 = &line1.b;
        let c1 = &line1.c;

        let x0 = c0 * c1 + xi();
        let x2 = c0 * b1 + c1 * b0;
        let x3 = c0 + c1;
        let x4 = b0 * b1;
        let x5 = b0 + b1;

        [x0, x2, x3, x4, x5]
    }

    fn mul_by_02345(f: &Fq12, prod: &[Fq2; 5]) -> Fq12 {
        let coeffs = [prod[0], Fq2::zero(), prod[1], prod[2], prod[3], prod[4]];
        *f * fq12_from_coeffs(coeffs)
    }

    if lines.len() % 2 == 1 {
        f = mul_by_023(&f, &lines.pop().unwrap());
    }
    for chunk in lines.chunks(2) {
        if let [line0, line1] = chunk {
            let prod = mul_023_by_023(line0, line1);
            f = mul_by_02345(&f, &prod);
        }
    }
    f
}

fn pre_loop(
    mut q_acc: Vec<G2Affine>,
    q: &[G2Affine],
    c: Option<Fq12>,
    xy_fracs: &[(Fq, Fq)],
) -> (Fq12, Vec<G2Affine>) {
    let mut f = if let Some(embedded) = c {
        let embedded_sq = embedded.square();
        embedded * embedded_sq
    } else {
        Fq12::ONE
    };

    let (q_out_double, lines_2s): (Vec<_>, Vec<_>) = q_acc
        .into_iter()
        .map(|acc| miller_double_step(&acc))
        .unzip();
    q_acc = q_out_double;

    let mut initial_lines = Vec::with_capacity(2 * xy_fracs.len());
    for (line, xy_frac) in lines_2s.iter().zip(xy_fracs.iter()) {
        initial_lines.push(EvaluatedLine {
            b: mul_fp2_by_fp(&line.b, &xy_frac.0),
            c: mul_fp2_by_fp(&line.c, &xy_frac.1),
        });
    }

    let (q_out_add, lines_s_plus_q): (Vec<_>, Vec<_>) = q_acc
        .iter()
        .zip(q.iter())
        .map(|(acc, q)| miller_add_step(acc, q))
        .unzip();
    q_acc = q_out_add;

    for (line, xy_frac) in lines_s_plus_q.iter().zip(xy_fracs.iter()) {
        initial_lines.push(EvaluatedLine {
            b: mul_fp2_by_fp(&line.b, &xy_frac.0),
            c: mul_fp2_by_fp(&line.c, &xy_frac.1),
        });
    }

    f = evaluate_lines_vec(f, initial_lines);

    (f, q_acc)
}

fn post_loop(f: Fq12, q_acc: Vec<G2Affine>) -> (Fq12, Vec<G2Affine>) {
    let mut result = f;
    result.conjugate_in_place();
    (result, q_acc)
}

pub fn multi_miller_loop(P: &[G1Affine], Q: &[G2Affine]) -> Fq12 {
    multi_miller_loop_embedded_exp(P, Q, None)
}

pub fn multi_miller_loop_embedded_exp(P: &[G1Affine], Q: &[G2Affine], c: Option<Fq12>) -> Fq12 {
    assert!(!P.is_empty());
    assert_eq!(P.len(), Q.len());

    let c_inv_value = c.as_ref().map(|value| {
        value
            .inverse()
            .expect("attempted to invert zero element for embedded exponent")
    });

    let filtered: Vec<_> = P
        .iter()
        .zip(Q.iter())
        .filter(|(p, q)| !p.is_zero() && !q.is_zero())
        .map(|(p, q)| (*p, *q))
        .collect();
    let (P, Q): (Vec<_>, Vec<_>) = filtered.into_iter().unzip();

    if P.is_empty() {
        return Fq12::ONE;
    }

    let xy_fracs: Vec<(Fq, Fq)> = P
        .iter()
        .map(|p| {
            let (x, y) = p.xy().unwrap();
            let y_inv = y.inverse().unwrap();
            (x * y_inv, y_inv)
        })
        .collect();

    let mut Q_acc = Q.clone();
    let (mut f, new_q_acc) = pre_loop(Q_acc, &Q, c.clone(), &xy_fracs);
    Q_acc = new_q_acc;

    for i in (0..BLS12_381_PSEUDO_BINARY_ENCODING.len() - 2).rev() {
        f = f.square();
        let mut lines = Vec::new();

        if BLS12_381_PSEUDO_BINARY_ENCODING[i] == 0 {
            let (Q_out, lines_2S): (Vec<_>, Vec<_>) =
                Q_acc.iter().map(|q| miller_double_step(q)).unzip();
            Q_acc = Q_out;
            for (line, xy_frac) in lines_2S.iter().zip(xy_fracs.iter()) {
                lines.push(EvaluatedLine {
                    b: mul_fp2_by_fp(&line.b, &xy_frac.0),
                    c: mul_fp2_by_fp(&line.c, &xy_frac.1),
                });
            }
        } else {
            if let Some(c_val) = c.as_ref() {
                match BLS12_381_PSEUDO_BINARY_ENCODING[i] {
                    1 => {
                        f *= c_val;
                    }
                    -1 => {
                        let c_inv = c_inv_value
                            .as_ref()
                            .expect("missing inverse for embedded exponent input");
                        f *= c_inv;
                    }
                    _ => panic!("Invalid sigma_i"),
                }
            }

            let results: Vec<_> = Q_acc
                .iter()
                .zip(&Q)
                .map(|(q_acc, q)| {
                    let q_signed = if BLS12_381_PSEUDO_BINARY_ENCODING[i] == 1 {
                        *q
                    } else {
                        -(*q)
                    };
                    miller_double_and_add_step(q_acc, &q_signed)
                })
                .collect();
            let (Q_out, lines_S_plus_Q, lines_S_plus_Q_plus_S): (Vec<_>, Vec<_>, Vec<_>) =
                results.into_iter().fold(
                    (Vec::new(), Vec::new(), Vec::new()),
                    |(mut qs, mut ls1, mut ls2), (q, l1, l2)| {
                        qs.push(q);
                        ls1.push(l1);
                        ls2.push(l2);
                        (qs, ls1, ls2)
                    },
                );
            Q_acc = Q_out;
            for ((line0, line1), xy_frac) in lines_S_plus_Q
                .iter()
                .zip(lines_S_plus_Q_plus_S.iter())
                .zip(xy_fracs.iter())
            {
                lines.push(EvaluatedLine {
                    b: mul_fp2_by_fp(&line0.b, &xy_frac.0),
                    c: mul_fp2_by_fp(&line0.c, &xy_frac.1),
                });
                lines.push(EvaluatedLine {
                    b: mul_fp2_by_fp(&line1.b, &xy_frac.0),
                    c: mul_fp2_by_fp(&line1.c, &xy_frac.1),
                });
            }
        }

        f = evaluate_lines_vec(f, lines);
    }

    let (f, _) = post_loop(f, Q_acc);

    f
}
