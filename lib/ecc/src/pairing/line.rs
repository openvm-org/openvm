use ff::Field;

use crate::field::FieldExtension;

#[derive(Clone, Copy, Debug)]
pub struct UnevaluatedLine<Fp, Fp2>
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
{
    pub b: Fp2,
    pub c: Fp2,
}

impl<Fp, Fp2> UnevaluatedLine<Fp, Fp2>
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
{
    pub fn evaluate(self, x_over_y: Fp, y_inv: Fp) -> EvaluatedLine<Fp, Fp2> {
        EvaluatedLine {
            b: self.b.mul_base(&x_over_y),
            c: self.c.mul_base(&y_inv),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EvaluatedLine<Fp, Fp2>
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
{
    pub b: Fp2,
    pub c: Fp2,
}

pub trait LineMType<Fp, Fp2, Fp12>
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
    Fp12: FieldExtension<BaseField = Fp2>,
{
    fn from_evaluated_line_m_type(line: EvaluatedLine<Fp, Fp2>) -> Fp12;
}

pub trait LineDType<Fp, Fp2, Fp12>
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
    Fp12: FieldExtension<BaseField = Fp2>,
{
    fn from_evaluated_line_d_type(line: EvaluatedLine<Fp, Fp2>) -> Fp12;
}
