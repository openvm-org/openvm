use halo2curves_axiom::bls12_381::{Fq, Fq2};

use crate::common::FieldExtension;

// BLS12-381 pseudo-binary encoding
// from gnark implementation: https://github.com/Consensys/gnark/blob/42dcb0c3673b2394bf1fd82f5128f7a121d7d48e/std/algebra/emulated/sw_bls12381/pairing.go#L322
pub const BLS12_381_SEED: u64 = 0xd201000000010000;
pub const BLS12_381_SEED_NEG: bool = true;
pub const BLS12_381_PBE_BITS: usize = 64;
pub const BLS12_381_PBE: [i8; BLS12_381_PBE_BITS] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1,
];

pub struct Bls12_381;

impl Bls12_381 {
    pub fn xi() -> Fq2 {
        Fq2::from_coeffs(&[Fq::one(), Fq::one()])
    }

    pub fn seed() -> u64 {
        BLS12_381_SEED
    }

    pub fn pseudo_binary_encoding() -> [i8; BLS12_381_PBE_BITS] {
        BLS12_381_PBE
    }
}
