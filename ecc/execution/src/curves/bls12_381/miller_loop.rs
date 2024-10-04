use halo2curves_axiom::ff::Field;
use itertools::izip;

use super::{mul_023_by_023, mul_by_023, mul_by_02345, BLS12_381, BLS12_381_PBE_BITS};
use crate::common::{
    miller_add_step, miller_double_step, EcPoint, EvaluatedLine, FieldExtension, MultiMillerLoop,
};

#[allow(non_snake_case)]
impl<Fp, Fp2, Fp12> MultiMillerLoop<Fp, Fp2, Fp12, BLS12_381_PBE_BITS>
    for BLS12_381<Fp, Fp2, BLS12_381_PBE_BITS>
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
    Fp12: FieldExtension<BaseField = Fp2>,
{
    fn xi(&self) -> Fp2 {
        BLS12_381::xi()
    }

    fn seed(&self) -> u64 {
        self.seed
    }

    fn pseudo_binary_encoding(&self) -> [i32; BLS12_381_PBE_BITS] {
        self.pseudo_binary_encoding
    }

    fn evaluate_lines_vec(&self, f: Fp12, lines: Vec<EvaluatedLine<Fp, Fp2>>) -> Fp12 {
        let mut f = f;
        let mut lines = lines;
        if lines.len() % 2 == 1 {
            f = mul_by_023(f, lines.pop().unwrap());
        }
        for chunk in lines.chunks(2) {
            if let [line0, line1] = chunk {
                let prod = mul_023_by_023(*line0, *line1, BLS12_381::xi());
                f = mul_by_02345(f, prod);
            } else {
                panic!("lines.len() % 2 should be 0 at this point");
            }
        }
        f
    }

    fn pre_loop(
        &self,
        f: Fp12,
        Q_acc: Vec<EcPoint<Fp2>>,
        Q: &[EcPoint<Fp2>],
        x_over_ys: Vec<Fp>,
        y_invs: Vec<Fp>,
    ) -> (Fp12, Vec<EcPoint<Fp2>>) {
        let mut f = f;
        let mut Q_acc = Q_acc;

        // Special case the first iteration of the miller loop with pseudo_binary_encoding = 1:
        // this means that the first step is a double and add, but we need to separate the two steps since the optimized
        // `miller_double_and_add_step` will fail because Q_acc is equal to Q_signed on the first iteration
        let (Q_out_double, lines_2S) = Q_acc
            .into_iter()
            .map(|Q| miller_double_step::<Fp, Fp2>(Q.clone()))
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

        f = self.evaluate_lines_vec(f, initial_lines);

        (f, Q_acc)
    }

    fn post_loop(&self, f: Fp12, Q_acc: Vec<EcPoint<Fp2>>) -> (Fp12, Vec<EcPoint<Fp2>>) {
        let res = f.conjugate();
        (res, Q_acc)
    }
}
