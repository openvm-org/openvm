use halo2curves_axiom::ff::Field;

use crate::common::FieldExtension;

pub fn final_exponentiation<Fp, Fp2, Fp12>(_f: Fp12) -> Fp12
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
    Fp12: FieldExtension<BaseField = Fp2>,
{
    unimplemented!("final_exponentiation is not implemented");
}
