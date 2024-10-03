use halo2curves_axiom::bn256::{Fq, Fq2};

pub const BN254_XI: Fq2 = Fq2 {
    c0: Fq::from_raw([9, 0, 0, 0]),
    c1: Fq::one(),
};

// from gnark implementation: https://github.com/Consensys/gnark/blob/42dcb0c3673b2394bf1fd82f5128f7a121d7d48e/std/algebra/emulated/sw_bn254/pairing.go#L356
// loopCounter = 6x₀+2 = 29793968203157093288
// in 2-NAF (nonadjacent form)
pub const GNARK_BN254_PBE_NAF: [i32; 66] = [
    0, 0, 0, 1, 0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, -1, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0,
    -1, 0, 0, 1, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 1, 0, -1, 0, 0, 0, -1, 0, -1, 0,
    0, 0, 1, 0, -1, 0, 1,
];

pub struct BN254 {
    pub xi: Fq2,
    pub negative_x: bool,
    pub pseudo_binary_encoding: [i32; 66],
}

impl BN254 {
    pub fn new() -> Self {
        Self {
            xi: BN254_XI,
            negative_x: false,
            pseudo_binary_encoding: GNARK_BN254_PBE_NAF,
        }
    }
}
