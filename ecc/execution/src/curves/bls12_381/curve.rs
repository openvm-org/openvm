use halo2curves_axiom::bls12_381::{Fq, Fq2, G1Affine, G2Affine};
use rand::Rng;

use crate::common::{AffineCoords, FieldExtension};

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

impl AffineCoords<Fq> for G1Affine {
    fn x(&self) -> Fq {
        self.x
    }

    fn y(&self) -> Fq {
        self.y
    }

    fn neg(&self) -> Self {
        let mut pt = *self;
        pt.y = -pt.y;
        pt
    }

    fn random(rng: &mut impl Rng) -> Self {
        G1Affine::random(rng)
    }

    fn generator() -> Self {
        G1Affine::generator()
    }
}

impl AffineCoords<Fq2> for G2Affine {
    fn x(&self) -> Fq2 {
        self.x
    }

    fn y(&self) -> Fq2 {
        self.y
    }

    fn neg(&self) -> Self {
        let mut pt = *self;
        pt.y = -pt.y;
        pt
    }

    fn random(rng: &mut impl Rng) -> Self {
        G2Affine::random(rng)
    }

    fn generator() -> Self {
        G2Affine::generator()
    }
}
