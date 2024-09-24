use halo2curves_axiom::ff::Field;

use crate::common::field::FieldExtension;

pub fn final_exp_hint<Fp, Fp2, Fp6, Fp12>(f: Fp12) -> (Fp12, Fp12)
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
    Fp6: FieldExtension<BaseField = Fp2>,
    Fp12: FieldExtension<BaseField = Fp6>,
{
    unimplemented!("final_exp_hint is not implemented");
}

pub fn final_exponentiation<Fp, Fp2, Fp6, Fp12>(x: Fp12) -> Fp12
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
    Fp6: FieldExtension<BaseField = Fp2>,
    Fp12: FieldExtension<BaseField = Fp6>,
{
    unimplemented!("final_exponentiation is not implemented");
}
