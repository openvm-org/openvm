use std::marker::PhantomData;

use halo2curves_axiom::ff::Field;

use crate::common::FieldExtension;

// from gnark implementation: https://github.com/Consensys/gnark/blob/42dcb0c3673b2394bf1fd82f5128f7a121d7d48e/std/algebra/emulated/sw_bn254/pairing.go#L356
// loopCounter = 6x₀+2 = 29793968203157093288 in 2-NAF (nonadjacent form)
// where curve seed x = 0x44e992b44a6909f1
pub const BN254_SEED: u64 = 0x44e992b44a6909f1;
pub const BN254_SEED_NEG: bool = false;
pub const BN254_PBE_BITS: usize = 66;
pub const GNARK_BN254_PBE_NAF: [i32; BN254_PBE_BITS] = [
    0, 0, 0, 1, 0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, -1, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0,
    -1, 0, 0, 1, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 1, 0, -1, 0, 0, 0, -1, 0, -1, 0,
    0, 0, 1, 0, -1, 0, 1,
];

pub struct BN254<Fp, Fp2, const BITS: usize> {
    pub xi: Fp2,
    pub seed: u64,
    pub pseudo_binary_encoding: [i32; BITS],
    _marker: PhantomData<Fp>,
}

impl<Fp, Fp2> BN254<Fp, Fp2, BN254_PBE_BITS>
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
        let one = Fp::ONE;
        let two = one + one;
        let three = one + two;
        let nine = three * three;
        Fp2::from_coeffs(&[nine, one])
    }

    pub fn seed() -> u64 {
        BN254_SEED
    }

    pub fn pseudo_binary_encoding() -> [i32; BN254_PBE_BITS] {
        GNARK_BN254_PBE_NAF
    }
}
