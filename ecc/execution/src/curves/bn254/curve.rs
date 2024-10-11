use std::ops::Mul;

use halo2curves_axiom::{
    bn256::{Fq, Fq2, Fr, G1Affine, G2Affine},
    group::prime::PrimeCurveAffine,
};
use rand::Rng;

use crate::common::{AffineCoords, FieldExtension, ScalarMul};

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

impl AffineCoords<Fq> for G1Affine {
    fn x(&self) -> Fq {
        self.x
    }

    fn y(&self) -> Fq {
        self.y
    }

    fn random(rng: &mut impl Rng) -> Self {
        G1Affine::random(rng)
    }

    fn generator() -> Self {
        G1Affine::generator()
    }
}

impl ScalarMul<Fr> for G1Affine {
    fn scalar_mul(&self, s: Fr) -> Self {
        (self.to_curve().mul(s)).into()
    }
}

impl AffineCoords<Fq2> for G2Affine {
    fn x(&self) -> Fq2 {
        self.x
    }

    fn y(&self) -> Fq2 {
        self.y
    }

    fn random(rng: &mut impl Rng) -> Self {
        G2Affine::random(rng)
    }

    fn generator() -> Self {
        G2Affine::generator()
    }
}

impl ScalarMul<Fr> for G2Affine {
    fn scalar_mul(&self, s: Fr) -> Self {
        (self.to_curve().mul(s)).into()
    }
}
