use halo2curves_axiom::bls12_381::{Fq, Fq2};

pub const BLS12_381_XI: Fq2 = Fq2 {
    c0: Fq::one(),
    c1: Fq::one(),
};

// BLS12-381 pseudo-binary encoding
// from gnark implementation: https://github.com/Consensys/gnark/blob/42dcb0c3673b2394bf1fd82f5128f7a121d7d48e/std/algebra/emulated/sw_bls12381/pairing.go#L322
pub const GNARK_BLS12_381_PBE: [i32; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1,
];

pub struct BLS12_381 {
    pub xi: Fq2,
    pub negative_x: bool,
    pub pseudo_binary_encoding: [i32; 64],
}

impl BLS12_381 {
    pub fn new() -> Self {
        Self {
            xi: BLS12_381_XI,
            pseudo_binary_encoding: GNARK_BLS12_381_PBE,
            negative_x: true,
        }
    }
}
