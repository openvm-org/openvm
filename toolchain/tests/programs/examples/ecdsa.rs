#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::hint::black_box;

use axvm::intrinsics::keccak256;
use hex_literal::hex;
use k256::ecdsa::{RecoveryId, Signature, VerifyingKey};

axvm::entry!(main);

// Ref: https://docs.rs/k256/latest/k256/ecdsa/index.html
pub fn main() {
    let msg = b"example message";

    let signature = Signature::try_from(
        hex!(
            "46c05b6368a44b8810d79859441d819b8e7cdc8bfd371e35c53196f4bcacdb51
     35c7facce2a97b95eacba8a586d87b7958aaf8368ab29cee481f76e871dbd9cb"
        )
        .as_slice(),
    )
    .unwrap();

    let recid = RecoveryId::try_from(1u8).unwrap();

    let prehash = keccak256(black_box(msg));

    // TODO: Swap out this function with axvm intrinsics.
    let recovered_key = VerifyingKey::recover_from_prehash(&prehash, &signature, recid).unwrap();

    let expected_key = VerifyingKey::from_sec1_bytes(&hex!(
        "0200866db99873b09fc2fb1e3ba549b156e96d1a567e3284f5f0e859a83320cb8b"
    ))
    .unwrap();

    if recovered_key != expected_key {
        panic!();
    }
}
