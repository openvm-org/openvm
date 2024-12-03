#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::hint::black_box;

use axvm_ecc_guest::VerifyingKey;
use axvm_keccak256_guest::keccak256;
use hex_literal::hex;
use k256::ecdsa::{self, RecoveryId, Signature};
axvm::entry!(main);

use axvm_ecc_guest::sw::Secp256k1Coord;
axvm_algebra_moduli_setup::moduli_init! {
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F",
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141"
}
axvm_ecc_sw_setup::sw_init! {
    Secp256k1Coord,
}

// Ref: https://docs.rs/k256/latest/k256/ecdsa/index.html
pub fn main() {
    setup_all_moduli();
    setup_all_curves();

    let msg = b"example message";

    let signature = Signature::try_from(
        hex!(
            "46c05b6368a44b8810d79859441d819b8e7cdc8bfd371e35c53196f4bcacdb5135c7facce2a97b95eacba8a586d87b7958aaf8368ab29cee481f76e871dbd9cb"
        )
        .as_slice(),
    )
    .unwrap();

    let recid = RecoveryId::try_from(1u8).unwrap();

    let prehash = keccak256(black_box(msg));

    let recovered_key = VerifyingKey::recover_from_prehash(&prehash, &signature, recid).unwrap();

    let expected_key = ecdsa::VerifyingKey::from_sec1_bytes(&hex!(
        "0200866db99873b09fc2fb1e3ba549b156e96d1a567e3284f5f0e859a83320cb8b"
    ))
    .unwrap();

    assert_eq!(recovered_key.0, expected_key);
}
