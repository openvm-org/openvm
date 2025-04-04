use core::hint::black_box;
use openvm as _;

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
        &hex!("2bd3e6d0f3b142924f5ca7b49ce5b9d54c4703d7ae5648e61d02268b1a0a9fb721611ce0a6af85915e2f1d70300909ce2e49dfad4a4619c8390cae66cefdb20400000000000000000000000000000000000000000000000011138ce750fa15c2"),
        hex!("070a8d6a982153cae4be29d434e8faef8a47b274a053f5a4ee2a6c9c13c31e5c031b8ce914eba3a9ffb989f9cdd5b0f01943074bf4f0f315690ec3cec6981afc"),
    ),
];

fn main() {
    {
        setup_all_moduli();
        setup_all_curves();
    }

    for (input, expected) in TEST_CASES {
        let input = black_box(input);

        let outcome = bn128::run_mul(input, 0, 0).unwrap();
        assert_eq!(outcome.bytes.as_ref(), expected.as_slice());
    }
}
