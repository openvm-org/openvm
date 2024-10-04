use halo2curves_axiom::ff::Field;

use super::{mul_013_by_013, mul_by_01234, mul_by_013, BN254, BN254_PBE_BITS};
use crate::common::{EcPoint, EvaluatedLine, FieldExtension, MultiMillerLoop};

#[allow(non_snake_case)]
impl<Fp, Fp2, Fp12> MultiMillerLoop<Fp, Fp2, Fp12, BN254_PBE_BITS>
    for BN254<Fp, Fp2, BN254_PBE_BITS>
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
    Fp12: FieldExtension<BaseField = Fp2>,
{
    fn xi(&self) -> Fp2 {
        self.xi
    }

    fn seed(&self) -> u64 {
        self.seed
    }

    fn pseudo_binary_encoding(&self) -> [i32; BN254_PBE_BITS] {
        self.pseudo_binary_encoding
    }

    fn evaluate_lines_vec(&self, f: Fp12, lines: Vec<EvaluatedLine<Fp, Fp2>>) -> Fp12 {
        let mut f = f;
        let mut lines = lines;
        if lines.len() % 2 == 1 {
            f = mul_by_013(f, lines.pop().unwrap());
        }
        for chunk in lines.chunks(2) {
            if let [line0, line1] = chunk {
                let prod = mul_013_by_013(*line0, *line1, BN254::xi());
                f = mul_by_01234(f, prod);
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

        (f, Q_acc)
    }

    fn post_loop(&self, f: Fp12, Q_acc: Vec<EcPoint<Fp2>>) -> (Fp12, Vec<EcPoint<Fp2>>) {
        (f, Q_acc)
    }
}
