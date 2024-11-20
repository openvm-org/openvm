#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::hint::black_box;

use axvm::{
    intrinsics::keccak256,
    io::{read, reveal},
};
use hex_literal::hex;
use k256::ecdsa::{self, RecoveryId, Signature};
use revm_precompile::secp256k1::ec_recover_run;
use revm_primitives::alloy_primitives::Bytes;

axvm::entry!(main);

pub fn main() {
    // TODO: read_vec
    let msg = b"example message";
    let signature = hex!(
        "46c05b6368a44b8810d79859441d819b8e7cdc8bfd371e35c53196f4bcacdb5135c7facce2a97b95eacba8a586d87b7958aaf8368ab29cee481f76e871dbd9cb"
    );
    let recid = 1u8;
    let prehash = keccak256(black_box(msg));

    // Input format: prehash || [0; 31] || v || signature
    let mut input = prehash.to_vec();
    let v = recid + 27u8;
    input.extend_from_slice(&[0; 31]);
    input.push(v);
    input.extend_from_slice(&signature);
    let recovered = ec_recover_run(&Bytes::from(input), 3000).unwrap();

    let expected_key = ecdsa::VerifyingKey::from_sec1_bytes(&hex!(
        "0200866db99873b09fc2fb1e3ba549b156e96d1a567e3284f5f0e859a83320cb8b"
    ))
    .unwrap();
    let mut expected_address = keccak256(
        &expected_key
            .to_encoded_point(/* compress = */ false)
            .as_bytes()[1..],
    );
    expected_address[..12].fill(0); // 20 bytes as the address.

    assert_eq!(recovered.bytes.len(), 32);
    assert_eq!(recovered.bytes.as_ref(), expected_address);
}
