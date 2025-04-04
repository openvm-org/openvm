extern crate openvm;

use core::hint::black_box;

use hex_literal::hex;
use openvm_algebra_guest::moduli_macros::moduli_init;
use openvm_ecc_guest::sw_macros::sw_init;
use revm_precompile::{primitives::address, secp256k1::ec_recover_run, Address, Bytes};

#[allow(unused_imports)]
use openvm_ecc_guest::k256::Secp256k1Point;

// initialize moduli
moduli_init! {
    // secp256k1
    "0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f", // coordinate field
    "0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141", // scalar field
}
// initialize elliptic curves
sw_init! {
    Secp256k1Point,
}

const TEST_CASES: &[(&[u8], Option<Address>)] = &[
    (
        &hex!("18c547e4f7b0f325ad1e56f57e26c745b09a3e503d86e00e5255ff7f715d3d1c000000000000000000000000000000000000000000000000000000000000001c73b1693892219d736caba55bdb67216e485557ea6b6af75f37096c9aa6a5a75feeb940b1d03b21e36b0e47e79769f095fe2ab855bd91e3a38756b7d75a9c4549"),
        Some(address!("a94f5374fce5edbc8e2a8697c15331677e6ebf0b")),
    ),
];

fn main() {
    {
        setup_all_moduli();
        setup_all_curves();
    }

    for (input, expected) in TEST_CASES {
        let input = black_box(Bytes::from_static(input));
        let result = ec_recover_run(&input, u64::MAX).unwrap();

        match expected {
            Some(address) => assert_eq!(Address::from_slice(&result.bytes[12..]), *address),
            None => assert!(result.bytes.is_empty()),
        }
    }
}
