use std::marker::PhantomData;

use halo2curves_axiom::ff::Field;

use crate::common::FieldExtension;

// BLS12-381 pseudo-binary encoding
// from gnark implementation: https://github.com/Consensys/gnark/blob/42dcb0c3673b2394bf1fd82f5128f7a121d7d48e/std/algebra/emulated/sw_bls12381/pairing.go#L322
pub const BLS12_381_SEED: u64 = 0xd201000000010000;
pub const BLS12_381_SEED_NEG: bool = true;
pub const BLS12_381_PBE_BITS: usize = 64;
pub const BLS12_381_PBE: [i32; BLS12_381_PBE_BITS] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1,
];

pub struct BLS12_381<Fp, Fp2, const BITS: usize> {
    pub xi: Fp2,
    pub seed: u64,
    pub pseudo_binary_encoding: [i32; BITS],
    _marker: PhantomData<Fp>,
}

impl<Fp, Fp2> BLS12_381<Fp, Fp2, BLS12_381_PBE_BITS>
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
{
    pub fn new() -> Self {
        Self {
            xi: Self::xi(),
            seed: Self::seed(),
            pseudo_binary_encoding: Self::pseudo_binary_encoding(),
            _marker: PhantomData::<Fp>,
        }
    }

    pub fn xi() -> Fp2 {
        Fp2::from_coeffs(&[Fp::ONE, Fp::ONE])
    }

    pub fn seed() -> u64 {
        BLS12_381_SEED
    }

    pub fn pseudo_binary_encoding() -> [i32; BLS12_381_PBE_BITS] {
        BLS12_381_PBE
    }
}
