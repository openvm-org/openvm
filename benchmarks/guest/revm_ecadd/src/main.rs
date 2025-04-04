extern crate openvm;

use core::hint::black_box;

use hex_literal::hex;
use openvm_algebra_guest::moduli_macros::moduli_init;
use openvm_ecc_guest::sw_macros::sw_init;
use revm_precompile::bn128;

#[allow(unused_imports)]
use openvm_pairing_guest::bn254::Bn254G1Affine;

// initialize moduli
moduli_init! {
    // bn254 (alt bn128)
    "0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47", // coordinate field
    "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001", // scalar field
}
// initialize elliptic curves
sw_init! {
    Bn254G1Affine,
}

const TEST_CASES: &[(&[u8], [u8; 64])] = &[
    (
        &hex!("18b18acfb4c2c30276db5411368e7185b311dd124691610c5d3b74034e093dc9063c909c4720840cb5134cb9f59fa749755796819658d32efc0d288198f3726607c2b7f58a84bd6145f00c9c2bc0bb1a187f20ff2c92963a88019e7c6a014eed06614e20c147e940f2d70da3f74c9a17df361706a4485c742bd6788478fa17d7"),
        hex!("2243525c5efd4b9c3d3c45ac0ca3fe4dd85e830a4ce6b65fa1eeaee202839703301d1d33be6da8e509df21cc35964723180eed7532537db9ae5e7d48f195c915"),
    ),
];

fn main() {
    {
        setup_all_moduli();
        setup_all_curves();
    }

    for (input, expected) in TEST_CASES {
        let input = black_box(input);

        let outcome = bn128::run_add(input, 0, 0).unwrap();
        assert_eq!(outcome.bytes.as_ref(), expected.as_slice());
    }
}
