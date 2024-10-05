use halo2curves_axiom::bn256::{Fq, Fq12, Fq2, FROBENIUS_COEFF_FQ6_C1, XI_TO_Q_MINUS_1_OVER_2};
use itertools::izip;

use super::{mul_013_by_013, mul_by_01234, mul_by_013, Bn254, BN254_PBE_BITS};
use crate::common::{
    miller_add_step, miller_double_step, EcPoint, EvaluatedLine, FieldExtension, MultiMillerLoop,
};

#[allow(non_snake_case)]
impl MultiMillerLoop<Fq, Fq2, Fq12, BN254_PBE_BITS> for Bn254 {
    fn xi() -> Fq2 {
        Self::xi()
    }

    fn seed() -> u64 {
        Self::seed()
    }

    fn pseudo_binary_encoding() -> [i8; BN254_PBE_BITS] {
        Self::pseudo_binary_encoding()
    }

    fn evaluate_lines_vec(&self, f: Fq12, lines: Vec<EvaluatedLine<Fq, Fq2>>) -> Fq12 {
        let mut f = f;
        let mut lines = lines;
        if lines.len() % 2 == 1 {
            f = mul_by_013(f, lines.pop().unwrap());
        }
        for chunk in lines.chunks(2) {
            if let [line0, line1] = chunk {
                let prod = mul_013_by_013(*line0, *line1, Self::xi());
                f = mul_by_01234(f, prod);
            } else {
                panic!("lines.len() % 2 should be 0 at this point");
            }
        }
        f
    }

    fn pre_loop(
        &self,
        f: Fq12,
        Q_acc: Vec<EcPoint<Fq2>>,
        _Q: &[EcPoint<Fq2>],
        x_over_ys: Vec<Fq>,
        y_invs: Vec<Fq>,
    ) -> (Fq12, Vec<EcPoint<Fq2>>) {
        let mut f = f;
        let mut Q_acc = Q_acc;
        let mut initial_lines = Vec::<EvaluatedLine<Fq, Fq2>>::new();

        let (Q_out_double, lines_2S) = Q_acc
            .into_iter()
            .map(|Q| miller_double_step::<Fq, Fq2>(Q.clone()))
            .unzip::<_, _, Vec<_>, Vec<_>>();
        Q_acc = Q_out_double;

        let lines_iter = izip!(lines_2S.iter(), x_over_ys.iter(), y_invs.iter());
        for (line_2S, x_over_y, y_inv) in lines_iter {
            let line = line_2S.evaluate(*x_over_y, *y_inv);
            initial_lines.push(line);
        }

        f = self.evaluate_lines_vec(f, initial_lines);

        (f, Q_acc)
    }

    fn post_loop(
        &self,
        f: Fq12,
        Q_acc: Vec<EcPoint<Fq2>>,
        Q: &[EcPoint<Fq2>],
        x_over_ys: Vec<Fq>,
        y_invs: Vec<Fq>,
    ) -> (Fq12, Vec<EcPoint<Fq2>>) {
        let mut f = f;
        let mut Q_acc = Q_acc;
        let mut lines = Vec::<EvaluatedLine<Fq, Fq2>>::new();

        let x_to_q_minus_1_over_3 = FROBENIUS_COEFF_FQ6_C1[1];
        let x_to_q_sq_minus_1_over_3 = FROBENIUS_COEFF_FQ6_C1[2];
        let q1_vec = Q
            .iter()
            .map(|Q| {
                let x = Q.x.frobenius_map(Some(1));
                let x = x * x_to_q_minus_1_over_3;
                let y = Q.y.frobenius_map(Some(1));
                let y = y * XI_TO_Q_MINUS_1_OVER_2;
                EcPoint { x, y }
            })
            .collect::<Vec<_>>();

        let (Q_out_add, lines_S_plus_Q) = Q_acc
            .iter()
            .zip(q1_vec.iter())
            .map(|(Q_acc, q1)| miller_add_step::<Fq, Fq2>(Q_acc.clone(), q1.clone()))
            .unzip::<_, _, Vec<_>, Vec<_>>();
        Q_acc = Q_out_add;

        let lines_iter = izip!(lines_S_plus_Q.iter(), x_over_ys.iter(), y_invs.iter());
        for (lines_S_plus_Q, x_over_y, y_inv) in lines_iter {
            let line = lines_S_plus_Q.evaluate(*x_over_y, *y_inv);
            lines.push(line);
        }

        let q2_vec = Q
            .iter()
            .map(|Q| {
                // There is a frobenius mapping π²(Q) that we skip here since it is equivalent to the identity mapping
                let x = Q.x * x_to_q_sq_minus_1_over_3;
                EcPoint { x, y: Q.y }
            })
            .collect::<Vec<_>>();

        let (Q_out_add, lines_S_plus_Q) = Q_acc
            .iter()
            .zip(q2_vec.iter())
            .map(|(Q_acc, q2)| miller_add_step::<Fq, Fq2>(Q_acc.clone(), q2.clone()))
            .unzip::<_, _, Vec<_>, Vec<_>>();
        Q_acc = Q_out_add;

        let lines_iter = izip!(lines_S_plus_Q.iter(), x_over_ys.iter(), y_invs.iter());
        for (lines_S_plus_Q, x_over_y, y_inv) in lines_iter {
            let line = lines_S_plus_Q.evaluate(*x_over_y, *y_inv);
            lines.push(line);
        }

        f = self.evaluate_lines_vec(f, lines);

        (f, Q_acc)
    }
}
