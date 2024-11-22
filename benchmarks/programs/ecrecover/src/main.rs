#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use core::hint::black_box;

use axvm::{intrinsics::keccak256, io::read_vec};
use axvm_algebra::IntMod;
use k256::ecdsa::{SigningKey, VerifyingKey};
use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;
use revm_precompile::secp256k1::ec_recover_run;
use revm_primitives::alloy_primitives::Bytes;

axvm::entry!(main);

axvm::moduli_setup! {
    Secp256k1Coord = "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F";
    Secp256k1Scalar = "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141";
}

pub fn main() {
    let msg = read_vec();
    for byte in msg {
        let prehash = keccak256(black_box(&[byte]));

        let mut rng = ChaCha8Rng::seed_from_u64(12345);
        let signing_key: SigningKey = SigningKey::random(&mut rng);
        let verifying_key = VerifyingKey::from(&signing_key);
        let (signature, recid) = signing_key.sign_prehash_recoverable(&prehash).unwrap();

        // Input format: prehash || [0; 31] || v || signature
        let mut input = prehash.to_vec();
        let v = recid.to_byte() + 27u8;
        input.extend_from_slice(&[0; 31]);
        input.push(v);
        input.extend_from_slice(signature.to_bytes().as_ref());
        let recovered = ec_recover_run(&Bytes::from(input), 3000).unwrap();

        let mut expected_address = keccak256(
            &verifying_key
                .to_encoded_point(/* compress = */ false)
                .as_bytes()[1..],
        );
        expected_address[..12].fill(0); // 20 bytes as the address.

        assert_eq!(recovered.bytes.len(), 32);
        assert_eq!(recovered.bytes.as_ref(), expected_address);
    }
}
