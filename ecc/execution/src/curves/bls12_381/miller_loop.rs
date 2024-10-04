use halo2curves_axiom::ff::Field;

use super::{mul_023_by_023, mul_by_023, mul_by_02345, BLS12_381, BLS12_381_PBE_BITS};
use crate::common::{EvaluatedLine, FieldExtension, MultiMillerLoop};

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

    fn negative_x(&self) -> bool {
        true
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
}
