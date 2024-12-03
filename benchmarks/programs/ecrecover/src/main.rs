#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use axvm::io::read_vec;
#[allow(unused_imports)]
use axvm_ecc_guest::k256::Secp256k1Coord;
use revm_precompile::secp256k1::ec_recover_run;
use revm_primitives::alloy_primitives::Bytes;

axvm::entry!(main);

axvm_algebra_guest::moduli_setup::moduli_init! {
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F",
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141"
}
axvm_ecc_guest::sw_setup::sw_init! {
    Secp256k1Coord,
}

pub fn main() {
    setup_all_moduli();
    setup_all_curves();

    let expected_address = read_vec();
    for _ in 0..5 {
        let input = read_vec();
        let recovered = ec_recover_run(&Bytes::from(input), 3000).unwrap();
        assert_eq!(recovered.bytes.as_ref(), expected_address);
    }
}
