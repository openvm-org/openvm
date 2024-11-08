use crate::field::{Field, FieldExtension, SexticExtField};

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
    pub fn evaluate(&self, x_over_y: &Fp, y_inv: &Fp) -> EvaluatedLine<Fp, Fp2> {
        EvaluatedLine {
            b: self.b.mul_base(x_over_y.clone()),
            c: self.c.mul_base(y_inv.clone()),
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

/// Convert M-type lines into Fp12 elements
pub trait LineMType<Fp, Fp2, Fp12>
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
    Fp12: FieldExtension<BaseField = Fp2>,
{
    fn from_evaluated_line_m_type(line: EvaluatedLine<Fp, Fp2>) -> Fp12;
}

/// Trait definition for line multiplication opcodes for M-type lines
pub trait LineMulMType<Fp, Fp2, Fp12>
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
    Fp12: FieldExtension<BaseField = Fp2>,
{
    fn mul_023_by_023(
        l0: EvaluatedLine<Fp, Fp2>,
        l1: EvaluatedLine<Fp, Fp2>,
    ) -> SexticExtField<Fp2>;

    fn mul_by_023(f: Fp12, l: EvaluatedLine<Fp, Fp2>) -> Fp12;

    fn mul_by_02345(f: Fp12, x: SexticExtField<Fp2>) -> Fp12;

    fn evaluate_line(
        l: UnevaluatedLine<Fp, Fp2>,
        x_over_y: Fp,
        y_inv: Fp,
    ) -> EvaluatedLine<Fp, Fp2>;
}

/// Convert D-type lines into Fp12 elements
pub trait LineDType<Fp, Fp2, Fp12>
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
    Fp12: FieldExtension<BaseField = Fp2>,
{
    fn from_evaluated_line_d_type(line: EvaluatedLine<Fp, Fp2>) -> Fp12;
}

/// Trait definition for line multiplication opcodes for D-type lines
pub trait LineMulDType<Fp, Fp2, Fp12>
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
    Fp12: FieldExtension<BaseField = Fp2>,
{
    fn mul_013_by_013(
        l0: EvaluatedLine<Fp, Fp2>,
        l1: EvaluatedLine<Fp, Fp2>,
    ) -> SexticExtField<Fp2>;

    fn mul_by_013(f: Fp12, l: EvaluatedLine<Fp, Fp2>) -> Fp12;

    fn mul_by_01234(f: Fp12, x: SexticExtField<Fp2>) -> Fp12;

    fn evaluate_line(
        l: UnevaluatedLine<Fp, Fp2>,
        x_over_y: Fp,
        y_inv: Fp,
    ) -> EvaluatedLine<Fp, Fp2>;
}
