use ark_bn254::{Fq, Fq12, Fq2, Fq6, G1Affine, G2Affine};
use ark_ec::AffineRepr;
use ark_ff::Field;

use crate::{
    arkworks::miller_loop::{miller_add_step, miller_double_and_add_step, miller_double_step},
    bn254::arkworks::{FROBENIUS_COEFF_FQ6_C1, XI_TO_Q_MINUS_1_OVER_2},
};

const BN254_PSEUDO_BINARY_ENCODING: [i8; 66] = [
    0, 0, 0, 1, 0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, -1, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0,
    -1, 0, 0, 1, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 1, 0, -1, 0, 0, 0, -1, 0, -1, 0,
    0, 0, 1, 0, -1, 0, 1,
];

#[derive(Clone, Debug)]
struct EvaluatedLine {
    b: Fq2,
    c: Fq2,
}

fn evaluate_lines_vec(mut f: Fq12, mut lines: Vec<EvaluatedLine>) -> Fq12 {
    fn mul_by_013(f: &Fq12, line: &EvaluatedLine) -> Fq12 {
        trait MulByNonresidue {
            fn mul_by_nonresidue(&self) -> Self;
        }
        impl MulByNonresidue for Fq2 {
            fn mul_by_nonresidue(&self) -> Self {
                let c0 = self.c1 * Fq::from(9u64);
                let c1 = self.c0 + self.c1;
                Fq2::new(c0, c1)
            }
        }

        let c0 = &line.c;
        let c1 = &line.b;
        let a_a = f.c0.c0 * c0;
        let a_b = f.c0.c1 * c0;
        let a_c = f.c0.c2 * c0;
        let b_a = (f.c1.c0 * c1).mul_by_nonresidue();
        let b_c = f.c1.c2 * c1;
        let c0_new = Fq6::new(a_a + b_a, a_b, a_c + b_c);
        let t0 = c0 + c1;
        let t1 = (f.c0.c0 + f.c1.c0) * t0 - a_a - b_a;
        let t2 = (f.c0.c1 + f.c1.c1) * t0 - a_b;
        let t3 = (f.c0.c2 + f.c1.c2) * t0 - a_c - b_c;
        let c1_new = Fq6::new(t1, t2, t3);
        Fq12::new(c0_new, c1_new)
    }

    fn mul_013_by_013(line0: &EvaluatedLine, line1: &EvaluatedLine) -> [Fq2; 5] {
        trait MulByNonresidue {
            fn mul_by_nonresidue(&self) -> Self;
        }
        impl MulByNonresidue for Fq2 {
            fn mul_by_nonresidue(&self) -> Self {
                let c0 = self.c1 * Fq::from(9u64);
                let c1 = self.c0 + self.c1;
                Fq2::new(c0, c1)
            }
        }

        let a0 = &line0.c;
        let a1 = &line0.b;
        let b0 = &line1.c;
        let b1 = &line1.b;
        let c0 = a0 * b0;
        let c1 = a0 * b1 + a1 * b0;
        let c2 = a1 * b1;
        let c3 = c2.mul_by_nonresidue();
        let c4 = c3.mul_by_nonresidue();
        [c0, c1, c2, c3, c4]
    }

    fn mul_by_01234(f: &Fq12, prod: &[Fq2; 5]) -> Fq12 {
        trait MulByNonresidue {
            fn mul_by_nonresidue(&self) -> Self;
        }
        impl MulByNonresidue for Fq2 {
            fn mul_by_nonresidue(&self) -> Self {
                let c0 = self.c1 * Fq::from(9u64);
                let c1 = self.c0 + self.c1;
                Fq2::new(c0, c1)
            }
        }

        let [c0, c1, c2, c3, c4] = prod;
        let a0 = f.c0.c0 * c0;
        let a1 = f.c0.c1 * c1;
        let a2 = f.c0.c2 * c2;
        let b0 = f.c1.c0 * c3;
        let b1 = f.c1.c1 * c4;
        let t0 = (f.c0.c0 + f.c1.c0) * (c0 + c3) - a0 - b0;
        let t1 = (f.c0.c1 + f.c1.c1) * (c1 + c4) - a1 - b1;
        let t2 = (f.c0.c2 + f.c1.c2) * c2 - a2;
        let c0_new = Fq6::new(a0 + b1.mul_by_nonresidue(), a1, a2 + b0);
        let c1_new = Fq6::new(t0, t1, t2);
        Fq12::new(c0_new, c1_new)
    }

    if lines.len() % 2 == 1 {
        f = mul_by_013(&f, &lines.pop().unwrap());
    }
    for chunk in lines.chunks(2) {
        if let [line0, line1] = chunk {
            let prod = mul_013_by_013(line0, line1);
            f = mul_by_01234(&f, &prod);
        }
    }
    f
}

fn post_loop(
    f: Fq12,
    mut q_acc: Vec<G2Affine>,
    q: &[G2Affine],
    xy_fracs: &[(Fq, Fq)],
) -> (Fq12, Vec<G2Affine>) {
    let mut lines = Vec::new();
    let x_to_q_minus_1_over_3 = FROBENIUS_COEFF_FQ6_C1[1];
    let x_to_q_sq_minus_1_over_3 = FROBENIUS_COEFF_FQ6_C1[2];
    let xi_to_q_minus_1_over_2 = *XI_TO_Q_MINUS_1_OVER_2;
    let q1_vec: Vec<G2Affine> = q
        .iter()
        .map(|q| {
            let (x, y) = q.xy().unwrap();
            let x = x.frobenius_map(1) * x_to_q_minus_1_over_3;
            let y = -(y.frobenius_map(1) * xi_to_q_minus_1_over_2);
            G2Affine::new_unchecked(x, y)
        })
        .collect();

    let (q_out_add, lines_s_plus_q1): (Vec<_>, Vec<_>) = q_acc
        .iter()
        .zip(q1_vec.iter())
        .map(|(q_acc, q1)| miller_add_step(q_acc, q1))
        .unzip();
    q_acc = q_out_add;

    for (line, xy_frac) in lines_s_plus_q1.iter().zip(xy_fracs.iter()) {
        lines.push(EvaluatedLine {
            b: line.b * Fq2::new(xy_frac.0, Fq::from(0u64)),
            c: line.c * Fq2::new(xy_frac.1, Fq::from(0u64)),
        });
    }

    let q2_vec: Vec<G2Affine> = q
        .iter()
        .map(|q| {
            let (x, y) = q.xy().unwrap();
            let x = x * x_to_q_sq_minus_1_over_3;
            G2Affine::new_unchecked(x, y)
        })
        .collect();

    let (_q_out_add, lines_s_plus_q2): (Vec<_>, Vec<_>) = q_acc
        .iter()
        .zip(q2_vec.iter())
        .map(|(q_acc, q2)| miller_add_step(q_acc, q2))
        .unzip();

    for (line, xy_frac) in lines_s_plus_q2.iter().zip(xy_fracs.iter()) {
        lines.push(EvaluatedLine {
            b: line.b * Fq2::new(xy_frac.0, Fq::from(0u64)),
            c: line.c * Fq2::new(xy_frac.1, Fq::from(0u64)),
        });
    }

    let mut f = f;
    f = evaluate_lines_vec(f, lines);

    (f, q_acc)
}

pub fn multi_miller_loop(P: &[G1Affine], Q: &[G2Affine]) -> Fq12 {
    assert!(!P.is_empty());
    assert_eq!(P.len(), Q.len());

    let (P, Q): (Vec<_>, Vec<_>) = P
        .iter()
        .zip(Q.iter())
        .filter(|(p, q)| !p.is_zero() && !q.is_zero())
        .map(|(p, q)| (*p, *q))
        .collect::<Vec<_>>()
        .into_iter()
        .unzip();

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
    let mut f = Fq12::ONE;

    let (Q_out_double, lines_2S): (Vec<_>, Vec<_>) =
        Q_acc.iter().map(|q| miller_double_step(q)).unzip();
    Q_acc = Q_out_double;

    let mut initial_lines = Vec::new();
    for (line, xy_frac) in lines_2S.iter().zip(xy_fracs.iter()) {
        initial_lines.push(EvaluatedLine {
            b: line.b * Fq2::new(xy_frac.0, Fq::from(0u64)),
            c: line.c * Fq2::new(xy_frac.1, Fq::from(0u64)),
        });
    }
    f = evaluate_lines_vec(f, initial_lines);

    for i in (0..BN254_PSEUDO_BINARY_ENCODING.len() - 2).rev() {
        f = f.square();
        let mut lines = Vec::new();

        if BN254_PSEUDO_BINARY_ENCODING[i] == 0 {
            let (Q_out, lines_2S): (Vec<_>, Vec<_>) =
                Q_acc.iter().map(|q| miller_double_step(q)).unzip();
            Q_acc = Q_out;
            for (line, xy_frac) in lines_2S.iter().zip(xy_fracs.iter()) {
                lines.push(EvaluatedLine {
                    b: line.b * Fq2::new(xy_frac.0, Fq::from(0u64)),
                    c: line.c * Fq2::new(xy_frac.1, Fq::from(0u64)),
                });
            }
        } else {
            let results: Vec<_> = Q_acc
                .iter()
                .zip(&Q)
                .map(|(q_acc, q)| {
                    let q_signed = if BN254_PSEUDO_BINARY_ENCODING[i] == 1 {
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
                    b: line0.b * Fq2::new(xy_frac.0, Fq::from(0u64)),
                    c: line0.c * Fq2::new(xy_frac.1, Fq::from(0u64)),
                });
                lines.push(EvaluatedLine {
                    b: line1.b * Fq2::new(xy_frac.0, Fq::from(0u64)),
                    c: line1.c * Fq2::new(xy_frac.1, Fq::from(0u64)),
                });
            }
        }
        f = evaluate_lines_vec(f, lines);
    }

    let (f, _) = post_loop(f, Q_acc, &Q, &xy_fracs);

    f
}
