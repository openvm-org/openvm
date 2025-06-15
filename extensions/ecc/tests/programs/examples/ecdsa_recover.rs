#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;

use ecdsa_core::RecoveryId;
use openvm::io::read;
#[allow(unused_imports)]
use openvm_p256::{
    ecdsa::{Signature, VerifyingKey},
    P256Coord, P256Point,
};
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, Bytes};

openvm::entry!(main);

openvm::init!("openvm_init_ecdsa_recover_p256.rs");

/// Signature recovery test vectors
#[repr(C)]
#[serde_as]
#[derive(Serialize, Deserialize)]
struct RecoveryTestVector {
    #[serde_as(as = "Bytes")]
    pk: [u8; 33],
    #[serde_as(as = "Bytes")]
    msg: [u8; 32],
    #[serde_as(as = "Bytes")]
    sig: [u8; 64],
    recid: u8,
    ok: bool,
}

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
        let _ = match VerifyingKey::recover_from_prehash(&vector.msg, &sig, recid) {
            Ok(_v) => _v,
            Err(_) => {
                openvm::io::println("Recovery failed");
                assert_eq!(vector.ok, false);
                continue;
            }
        };
        // If reached here, recovery succeeded
        assert_eq!(vector.ok, true);
    }
}
