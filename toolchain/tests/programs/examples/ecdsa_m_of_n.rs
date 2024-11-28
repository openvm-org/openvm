#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
extern crate serde_arrays;

use alloc::vec::Vec;
use core::hint::black_box;

use axvm::{intrinsics::keccak256, io::read};
use axvm_ecc::VerifyingKey;
use k256::ecdsa::{self, RecoveryId, Signature};
use serde_arrays::*;

axvm::entry!(main);

/// Inputs to the m-of-n ECDSA verification
#[derive(serde::Deserialize)]
struct Inputs {
    /// calculated keystore address
    keystore_address: [u8; 32],
    /// message hash
    msg_hash: [u8; 32],
    /// m number of signatures
    m: u32,
    /// n number of EOAs
    n: u32,
    /// vector of signatures
    #[serde(with = "BigArray")]
    signatures: Vec<[u8; 64]>,
    /// vector of EOAs
    #[serde(with = "BigArray")]
    eoa_addrs: Vec<[u8; 20]>,
}

/// Inputs: keystoreAddress, msgHash, m, n, m signatures, n EOA addresses
/// Outputs: none; panics if invalid
///
/// Circuit statement:
/// * keccak256(concat([eoa_addrs])) == dataHash
/// * there are [ECDSA signatures] for keccak256(keystoreAddress || dataHash || msgHash) which verifies against [pub_keys]
/// * [eoa_addrs] corresponds to [pub_keys]
pub fn main() {
    // read IO
    let io: Inputs = read();

    let keystore_address = io.keystore_address;
    let msg_hash = io.msg_hash;
    let m = io.m;
    let n = io.n;

    // Concatenate all EOA addresses into a single Vec<u8>
    let mut concat_eoa_addrs = Vec::<u8>::with_capacity(io.eoa_addrs.len() * 20);
    for addr in &io.eoa_addrs {
        concat_eoa_addrs.extend_from_slice(addr);
    }

    // NOTE: data_hash is currently defined as `keccak256([m, n, concat(eoa_addrs)])`
    let data_hash = keccak256(
        [
            m.to_be_bytes().to_vec(),
            n.to_be_bytes().to_vec(),
            concat_eoa_addrs,
        ]
        .concat()
        .as_slice(),
    );

    // check that the EOA
}
