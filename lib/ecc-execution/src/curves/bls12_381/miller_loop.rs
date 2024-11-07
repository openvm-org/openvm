use axvm_ecc::{
    curve::bls12381::{Fq, Fq12, Fq2},
    pairing::{miller_add_step, miller_double_step, EvaluatedLine, MultiMillerLoop},
    point::EcPoint,
};
use itertools::izip;

use super::{mul_023_by_023, mul_by_023, mul_by_02345, Bls12_381, BLS12_381_PBE_BITS};

#[allow(non_snake_case)]
impl MultiMillerLoop<Fq, Fq2, Fq12, BLS12_381_PBE_BITS> for Bls12_381 {
    fn xi() -> Fq2 {
        Self::xi()
    }

    fn seed() -> u64 {
        Self::seed()
    }

    fn pseudo_binary_encoding() -> [i8; BLS12_381_PBE_BITS] {
        Self::pseudo_binary_encoding()
    }

    fn evaluate_lines_vec(&self, f: Fq12, lines: Vec<EvaluatedLine<Fq, Fq2>>) -> Fq12 {
        let mut f = f;
        let mut lines = lines;
        if lines.len() % 2 == 1 {
            f = mul_by_023(f, lines.pop().unwrap());
        }
        for chunk in lines.chunks(2) {
            if let [line0, line1] = chunk {
                let prod = mul_023_by_023(*line0, *line1, Self::xi());
                f = mul_by_02345(f, prod);
            } else {
                panic!("lines.len() % 2 should be 0 at this point");
            }
        }
        f
    }

    /// The expected output of this function when running the Miller loop with embedded exponent is c^3 * l_{3Q}
    fn pre_loop(
        &self,
        f: Fq12,
        Q_acc: Vec<EcPoint<Fq2>>,
        Q: &[EcPoint<Fq2>],
        c: Option<Fq12>,
        x_over_ys: Vec<Fq>,
        y_invs: Vec<Fq>,
    ) -> (Fq12, Vec<EcPoint<Fq2>>) {
        let mut f = f;

        if c.is_some() {
            // for the miller loop with embedded exponent, f will be set to c at the beginning of the function, and we
            // will multiply by c again due to the last two values of the pseudo-binary encoding (BN12_381_PBE) being 1.
            // Therefore, the final value of f at the end of this block is c^3.
            f = f.square() * c.unwrap();
        }

        let mut Q_acc = Q_acc;

        // Special case the first iteration of the Miller loop with pseudo_binary_encoding = 1:
        // this means that the first step is a double and add, but we need to separate the two steps since the optimized
        // `miller_double_and_add_step` will fail because Q_acc is equal to Q_signed on the first iteration
        let (Q_out_double, lines_2S) = Q_acc
            .into_iter()
            .map(|Q| miller_double_step::<Fq, Fq2>(Q.clone()))
            .unzip::<_, _, Vec<_>, Vec<_>>();
        Q_acc = Q_out_double;

        let mut initial_lines = Vec::<EvaluatedLine<Fq, Fq2>>::new();

        let lines_iter = izip!(lines_2S.iter(), x_over_ys.iter(), y_invs.iter());
        for (line_2S, x_over_y, y_inv) in lines_iter {
            let line = line_2S.evaluate(*x_over_y, *y_inv);
            initial_lines.push(line);
        }

        let (Q_out_add, lines_S_plus_Q) = Q_acc
            .iter()
            .zip(Q.iter())
            .map(|(Q_acc, Q)| miller_add_step::<Fq, Fq2>(Q_acc.clone(), Q.clone()))
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

    /// After running the main body of the Miller loop, we conjugate f due to the curve seed x being negative.
    fn post_loop(
        &self,
        f: Fq12,
        Q_acc: Vec<EcPoint<Fq2>>,
        _Q: &[EcPoint<Fq2>],
        _c: Option<Fq12>,
        _x_over_ys: Vec<Fq>,
        _y_invs: Vec<Fq>,
    ) -> (Fq12, Vec<EcPoint<Fq2>>) {
        // Conjugate for negative component of the seed
        let f = f.conjugate();
        (f, Q_acc)
    }
}
