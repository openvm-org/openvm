#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;

use ecdsa_core::{signature::hazmat::PrehashVerifier, RecoveryId};
use openvm::io::read;
use openvm_ecc_test_programs::RecoveryTestVector;
#[allow(unused_imports)]
use openvm_k256::{
    ecdsa::{Signature, VerifyingKey},
    Secp256k1Coord, Secp256k1Point,
};

openvm::entry!(main);

openvm::init!("openvm_init_ec_k256.rs");

pub fn main() {
    let test_vectors: Vec<RecoveryTestVector> = read();
    for vector in test_vectors {
        let sig = match Signature::try_from(vector.sig.as_slice()) {
            Ok(_v) => _v,
            Err(_) => {
                assert_eq!(vector.ok, false);
                continue;
            }
        };
        let recid = match RecoveryId::from_byte(vector.recid) {
            Some(_v) => _v,
            None => {
                assert_eq!(vector.ok, false);
                continue;
            }
        };
        let vk = match VerifyingKey::recover_from_prehash(&vector.msg, &sig, recid) {
            Ok(_v) => _v,
            Err(_) => {
                openvm::io::println("Recovery failed");
                assert_eq!(vector.ok, false);
                continue;
            }
        };
        openvm::io::println(alloc::format!("{:?}", vk.to_sec1_bytes(false)));
        vk.verify_prehash(&vector.msg, &sig).unwrap();
        // If reached here, recovery succeeded
        assert_eq!(vector.ok, true);
    }
}
