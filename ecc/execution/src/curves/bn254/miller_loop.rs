use halo2curves_axiom::ff::Field;

use super::{mul_013_by_013, mul_by_01234, mul_by_013, BN254};
use crate::common::{EvaluatedLine, FieldExtension, MultiMillerLoop};

impl<Fp, Fp2, Fp12> MultiMillerLoop<Fp, Fp2, Fp12> for BN254
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
    Fp12: FieldExtension<BaseField = Fp2>,
{
    fn negative_x(&self) -> bool {
        true
    }

    fn evaluate_lines_vec(&self, f: Fp12, lines: Vec<EvaluatedLine<Fp, Fp2>>, xi: Fp2) -> Fp12 {
        let mut f = f;
        let mut lines = lines;
        if lines.len() % 2 == 1 {
            f = mul_by_013(f, lines.pop().unwrap());
        }
        for chunk in lines.chunks(2) {
            if let [line0, line1] = chunk {
                let prod = mul_013_by_013(*line0, *line1, xi);
                f = mul_by_01234(f, prod);
            } else {
                panic!("lines.len() % 2 should be 0 at this point");
            }
        }
        f
    }
}
