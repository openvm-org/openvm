#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::{collections::btree_set::BTreeSet, vec::Vec};

use axvm::{intrinsics::keccak256, io::read};
use axvm_ecc::VerifyingKey;
use k256::ecdsa::{RecoveryId, Signature};

axvm::entry!(main);

/// Inputs to the m-of-n ECDSA verification
#[derive(serde::Deserialize, Eq, PartialEq)]
struct Inputs {
    /// calculated keystore address
    keystore_address: [u8; 32],
    /// message hash
    msg_hash: [u8; 32],
    /// data hash
    data_hash: [u8; 32],
    /// m number of signatures
    m: u32,
    /// n number of EOAs
    n: u32,
    /// vector of signatures
    #[serde(deserialize_with = "deserialize_u8_65_vec")]
    signatures: Vec<[u8; 65]>,
    /// vector of lowercased EOAs
    #[serde(deserialize_with = "deserialize_u8_20_vec")]
    eoa_addrs: Vec<[u8; 20]>,
}

/// Inputs: keystoreAddress, msgHash, m, n, m signatures, n EOA addresses
/// Outputs: none; panics if invalid
///
/// Circuit statement:
/// * keccak256([m, n, concat(eoa_addrs)]) == dataHash
/// * there are [ECDSA signatures] for keccak256(keystoreAddress || dataHash || msgHash) which verifies against [pub_keys]
/// * [eoa_addrs] corresponds to [pub_keys]
pub fn main() {
    // read IO
    let io: Inputs = read();

    // Validate there are m signatures and that each is distinct
    let signatures = io
        .signatures
        .iter()
        .map(|s| s.to_vec())
        .collect::<Vec<Vec<u8>>>();
    assert!(signatures.len() as u32 == io.m);
    let mut signature_set = BTreeSet::new();
    for sig in signatures.iter() {
        signature_set.insert(sig.clone());
    }
    assert!(signature_set.len() as u32 == io.m);

    // Concatenate all EOA addresses into a single Vec<u8>
    let mut concat_eoa_addrs = Vec::<u8>::with_capacity(io.eoa_addrs.len() * 20);
    for addr in &io.eoa_addrs {
        concat_eoa_addrs.extend_from_slice(addr);
    }

    // NOTE: data_hash is currently defined as `keccak256([m, n, concat(eoa_addrs)])`
    let data_hash = keccak256(
        [
            io.m.to_be_bytes().to_vec(),
            io.n.to_be_bytes().to_vec(),
            concat_eoa_addrs,
        ]
        .concat()
        .as_slice(),
    );
    assert_eq!(data_hash, io.data_hash);

    let full_hash = keccak256(
        [
            io.keystore_address.to_vec(),
            io.data_hash.to_vec(),
            io.msg_hash.to_vec(),
        ]
        .concat()
        .as_slice(),
    );

    // Verify the signatures are valid
    for sig in signatures.iter() {
        // Reconstruct Signature struct
        // Signature is for `keccak256(keystoreAddress || dataHash || msgHash)`
        // We take the first 64 bytes because the 65th byte (recovery ID) is not part of k256::ecdsa::Signature
        let signature = Signature::from_slice(&sig[..64]).expect("Invalid signature");
        let recovery_id = RecoveryId::new((sig[64] & 1) == 1, (sig[64] >> 1) == 1);

        // Get verifying key from signature and verify the signature
        let vk = VerifyingKey::recover_from_prehash(&full_hash, &signature, recovery_id)
            .expect("Unable to recover from prehash");
        vk.verify_prehashed(&full_hash, &signature).unwrap();

        // Get ethereum address from verifying key (public key)
        let enc_pt = vk.0.to_encoded_point(false);
        let pt_bytes = enc_pt.as_bytes();
        let eth_addr: [u8; 20] = keccak256(&pt_bytes[1..])[12..]
            .try_into()
            .expect("Invalid ethereum address");

        // // Check that the ethereum address calculated from the public key is in the list of EOA addresses
        assert!(io.eoa_addrs.contains(&eth_addr));
    }
}

fn deserialize_u8_65_vec<'de, D>(deserializer: D) -> Result<Vec<[u8; 65]>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct VecVisitor;

    impl<'de> serde::de::Visitor<'de> for VecVisitor {
        type Value = Vec<[u8; 65]>;

        fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
            formatter.write_str("a sequence of bytes")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::SeqAccess<'de>,
        {
            let mut result = Vec::new();
            let mut current_array = [0u8; 65];
            let mut index = 0;

            while let Some(value) = seq.next_element()? {
                current_array[index] = value;
                index += 1;
                if index == 65 {
                    result.push(current_array);
                    index = 0;
                }
            }

            Ok(result)
        }
    }

    deserializer.deserialize_seq(VecVisitor)
}

fn deserialize_u8_20_vec<'de, D>(deserializer: D) -> Result<Vec<[u8; 20]>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct VecVisitor;

    impl<'de> serde::de::Visitor<'de> for VecVisitor {
        type Value = Vec<[u8; 20]>;

        fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
            formatter.write_str("a sequence of bytes")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::SeqAccess<'de>,
        {
            let mut result = Vec::new();
            let mut current_array = [0u8; 20];
            let mut index = 0;

            while let Some(value) = seq.next_element()? {
                current_array[index] = value;
                index += 1;
                if index == 20 {
                    result.push(current_array);
                    index = 0;
                }
            }

            Ok(result)
        }
    }

    deserializer.deserialize_seq(VecVisitor)
}
