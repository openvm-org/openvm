use alloc::vec::Vec;

use itertools::izip;

use super::{Bls12381Fp, Bls12381Fp12, Bls12381Fp2, Bls12381Intrinsic};
use crate::{
    field::{FieldExtension, Fp12Mul},
    pairing::{EvaluatedLine, LineMulMType, MillerStep, MultiMillerLoop},
    point::AffinePoint,
};

impl MillerStep for Bls12381Intrinsic {
    type Fp = Bls12381Fp;
    type Fp2 = Bls12381Fp2;
}

#[allow(non_snake_case)]
impl MultiMillerLoop for Bls12381Intrinsic {
    type Fp12 = Bls12381Fp12;

    const SEED_ABS: u64 = 0xd201000000010000;
    const PSEUDO_BINARY_ENCODING: &[i8] = &[
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        1, 0, 1, 1,
    ];

    fn evaluate_lines_vec(
        &self,
        f: Self::Fp12,
        lines: Vec<EvaluatedLine<Self::Fp, Self::Fp2>>,
    ) -> Self::Fp12 {
        let mut f = f;
        let mut lines = lines;
        if lines.len() % 2 == 1 {
            f = Self::mul_by_023(f, lines.pop().unwrap());
        }
        for chunk in lines.chunks(2) {
            if let [line0, line1] = chunk {
                let prod = Self::mul_023_by_023(line0.clone(), line1.clone());
                f = Self::mul_by_02345(f, prod);
            } else {
                panic!("lines.len() % 2 should be 0 at this point");
            }
        }
        f
    }

    /// The expected output of this function when running the Miller loop with embedded exponent is c^3 * l_{3Q}
    fn pre_loop(
        &self,
        f: &Self::Fp12,
        Q_acc: Vec<AffinePoint<Self::Fp2>>,
        Q: &[AffinePoint<Self::Fp2>],
        c: Option<Self::Fp12>,
        x_over_ys: Vec<Self::Fp>,
        y_invs: Vec<Self::Fp>,
    ) -> (Self::Fp12, Vec<AffinePoint<Self::Fp2>>) {
        let mut f = f.clone();

        if c.is_some() {
            // for the miller loop with embedded exponent, f will be set to c at the beginning of the function, and we
            // will multiply by c again due to the last two values of the pseudo-binary encoding (BN12_381_PBE) being 1.
            // Therefore, the final value of f at the end of this block is c^3.
            f = f.fp12_mul_refs(&f).fp12_mul_refs(&c.unwrap());
        }

        let mut Q_acc = Q_acc;

        // Special case the first iteration of the Miller loop with pseudo_binary_encoding = 1:
        // this means that the first step is a double and add, but we need to separate the two steps since the optimized
        // `miller_double_and_add_step` will fail because Q_acc is equal to Q_signed on the first iteration
        let (Q_out_double, lines_2S) = Q_acc
            .into_iter()
            .map(|Q| Self::miller_double_step(Q.clone()))
            .unzip::<_, _, Vec<_>, Vec<_>>();
        Q_acc = Q_out_double;

        let mut initial_lines = Vec::<EvaluatedLine<Self::Fp, Self::Fp2>>::new();

        let lines_iter = izip!(lines_2S.iter(), x_over_ys.iter(), y_invs.iter());
        for (line_2S, x_over_y, y_inv) in lines_iter {
            let line = line_2S.evaluate(&(x_over_y.clone(), y_inv.clone()));
            initial_lines.push(line);
        }

        let (Q_out_add, lines_S_plus_Q) = Q_acc
            .iter()
            .zip(Q.iter())
            .map(|(Q_acc, Q)| Self::miller_add_step(Q_acc.clone(), Q.clone()))
            .unzip::<_, _, Vec<_>, Vec<_>>();
        Q_acc = Q_out_add;

        let lines_iter = izip!(lines_S_plus_Q.iter(), x_over_ys.iter(), y_invs.iter());
        for (lines_S_plus_Q, x_over_y, y_inv) in lines_iter {
            let line = lines_S_plus_Q.evaluate(&(x_over_y.clone(), y_inv.clone()));
            initial_lines.push(line);
        }

        f = self.evaluate_lines_vec(f, initial_lines);

        (f, Q_acc)
    }

    /// After running the main body of the Miller loop, we conjugate f due to the curve seed x being negative.
    fn post_loop(
        &self,
        f: &Self::Fp12,
        Q_acc: Vec<AffinePoint<Self::Fp2>>,
        _Q: &[AffinePoint<Self::Fp2>],
        _c: Option<Self::Fp12>,
        _x_over_ys: Vec<Self::Fp>,
        _y_invs: Vec<Self::Fp>,
    ) -> (Self::Fp12, Vec<AffinePoint<Self::Fp2>>) {
        // Conjugate for negative component of the seed
        let f = f.conjugate();
        (f, Q_acc)
    }
}
