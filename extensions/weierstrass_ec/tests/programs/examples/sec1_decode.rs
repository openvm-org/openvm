#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;

use openvm::io::read;
#[allow(unused_imports)]
use openvm_k256::{ecdsa::VerifyingKey, Secp256k1Coord, Secp256k1Point};
use openvm_weierstrass_test_programs::Sec1DecodingTestVector;

openvm::entry!(main);

openvm::init!("openvm_init_ec_k256.rs");

pub fn main() {
    let test_vectors: Vec<Sec1DecodingTestVector> = read();
    for vector in test_vectors {
        assert_eq!(
            vector.ok,
            VerifyingKey::from_sec1_bytes(&vector.bytes).is_ok()
        );
    }
}
