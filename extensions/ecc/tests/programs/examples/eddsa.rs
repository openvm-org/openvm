#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use hex_literal::hex;
use openvm_ecc_guest::{ed25519::Ed25519Point, eddsa::VerifyingKey};

openvm::entry!(main);

openvm::init!("openvm_init_eddsa_ed25519.rs");

// Ref: https://docs.rs/k256/latest/k256/ecdsa/index.html
pub fn main() {
    // Test data taken from the RFC: https://datatracker.ietf.org/doc/html/rfc8032#section-7.3
    // Unfortuantely, the RFC only provides one test for Ed25519ph
    // TODO: find/generate more tests
    let msg = b"abc";

    let signature = hex!(
            "98a70222f0b8121aa9d30f813d683f809e462b469c7ff87639499bb94e6dae4131f85042463c2a355a2003d062adf5aaa10b8c61e636062aaad11c2a26083406"
        );

    let vk = VerifyingKey::<Ed25519Point>::from_bytes(&hex!(
        "ec172b93ad5e563bf4932c70e1245034c35467ef2efd4d64ebf819683467e2bf"
    ))
    .unwrap();

    assert!(vk.verify(msg, &signature));
}
