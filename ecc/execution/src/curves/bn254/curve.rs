use halo2curves_axiom::bn256::{Fq, Fq2};

use crate::common::FieldExtension;

// from gnark implementation: https://github.com/Consensys/gnark/blob/42dcb0c3673b2394bf1fd82f5128f7a121d7d48e/std/algebra/emulated/sw_bn254/pairing.go#L356
// loopCounter = 6x₀+2 = 29793968203157093288 in 2-NAF (nonadjacent form)
// where curve seed x = 0x44e992b44a6909f1
pub const BN254_SEED: u64 = 0x44e992b44a6909f1;
pub const BN254_SEED_NEG: bool = false;
pub const BN254_PBE_BITS: usize = 66;
pub const GNARK_BN254_PBE_NAF: [i8; BN254_PBE_BITS] = [
    0, 0, 0, 1, 0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, -1, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0,
    -1, 0, 0, 1, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 1, 0, -1, 0, 0, 0, -1, 0, -1, 0,
    0, 0, 1, 0, -1, 0, 1,
];

pub struct Bn254;

impl Bn254 {
    pub fn xi() -> Fq2 {
        Fq2::from_coeffs(&[Fq::from_raw([9, 0, 0, 0]), Fq::one()])
    }

    pub fn seed() -> u64 {
        BN254_SEED
    }

    pub fn pseudo_binary_encoding() -> [i8; BN254_PBE_BITS] {
        GNARK_BN254_PBE_NAF
    }
}
