use halo2curves_axiom::ff::Field;

use super::FieldExtension;

#[allow(non_snake_case)]
pub trait PairingCheck<Fp, Fp2>
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
{
    fn pairing_check(&self, P: Fp, Q: Fp2);
}
