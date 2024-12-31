#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]
#![allow(unused_imports)]

use core::hint::black_box;

use hex_literal::hex;
use k256::{
    ecdsa::{self, RecoveryId},
    Secp256k1,
};
use openvm_ecc_guest::{
    algebra::IntMod, ecdsa::VerifyingKey, k256::Secp256k1Coord, weierstrass::WeierstrassPoint,
};
use openvm_keccak256_guest::keccak256;
openvm::entry!(main);

openvm_algebra_moduli_setup::moduli_init! {
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F",
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141"
}
openvm_ecc_sw_setup::sw_init! {
    Secp256k1Coord,
}

// Ref: https://docs.rs/k256/latest/k256/ecdsa/index.html
pub fn main() {
    setup_all_moduli();
    setup_all_curves();

    let msg = b"example message";

    let signature = hex!(
            "46c05b6368a44b8810d79859441d819b8e7cdc8bfd371e35c53196f4bcacdb5135c7facce2a97b95eacba8a586d87b7958aaf8368ab29cee481f76e871dbd9cb"
        );

    let recid = RecoveryId::try_from(1u8).unwrap();

    let prehash = keccak256(black_box(msg));

    let recovered_key =
        VerifyingKey::<Secp256k1>::recover_from_prehash_noverify(&prehash, &signature, recid);

    let expected_key = ecdsa::VerifyingKey::from_sec1_bytes(&hex!(
        "0200866db99873b09fc2fb1e3ba549b156e96d1a567e3284f5f0e859a83320cb8b"
    ))
    .unwrap();
    // sec1 encoding, the first byte is for compression flag
    let expected_key = expected_key.to_encoded_point(false);
    let public_key = recovered_key.as_affine();
    let mut buffer = [0u8; 64];
    buffer[..32].copy_from_slice(&public_key.x().to_be_bytes());
    buffer[32..].copy_from_slice(&public_key.y().to_be_bytes());
    assert_eq!(&buffer, &expected_key.as_bytes()[1..]);

    recovered_key
        .verify_prehashed(&prehash, &signature)
        .unwrap();
}
