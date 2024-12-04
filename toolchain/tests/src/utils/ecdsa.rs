use ax_ecc_execution::axvm_ecc::VerifyingKey;
use axvm::intrinsics::keccak256;
use k256::ecdsa::{signature::hazmat::PrehashSigner, RecoveryId, Signature, SigningKey};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// keystore_address = keccak256(salt, keccak256([eoa_addrs]), keccak256(vk))
pub fn calculate_keystore_address(
    salt: [u8; 32],
    data_hash: [u8; 32],
    vk_hash: [u8; 32],
) -> [u8; 32] {
    keccak256(
        [salt.to_vec(), data_hash.to_vec(), vk_hash.to_vec()]
            .concat()
            .as_slice(),
    )
}

pub fn calculate_salt(seed: u64) -> [u8; 32] {
    let mut rng = StdRng::seed_from_u64(seed);
    let salt: [u8; 32] = rng.gen();
    salt
}

/// Calculates data_hash, where data_hash = keccak256(m || n || [eoa_addrs].concat())
pub fn calculate_data_hash(m: u32, n: u32, eoa_addrs: Vec<[u8; 20]>) -> [u8; 32] {
    keccak256(
        [
            m.to_be_bytes().to_vec(),
            n.to_be_bytes().to_vec(),
            eoa_addrs.concat().to_vec(),
        ]
        .concat()
        .as_slice(),
    )
}

/// Calculates the recovery id from a signature
pub fn calculate_recovery_id(signature: &Signature, full_hash: [u8; 32], eoa_addr: [u8; 20]) -> u8 {
    // Iterate through recovery IDs (0, 1, 2, 3)
    for i in 0..=3 {
        let recovery_id = RecoveryId::from_byte(i).unwrap();
        let vk = VerifyingKey::recover_from_prehash(&full_hash, &signature, recovery_id).unwrap();
        let calc_eoa_addr = calculate_eoa_addr(&vk.0);
        if eoa_addr == calc_eoa_addr {
            return i;
        }
    }
    panic!("Unable to calculate recovery_id for signature");
}

pub fn calculate_eoa_addr(vk: &k256::ecdsa::VerifyingKey) -> [u8; 20] {
    let enc_pt = vk.to_encoded_point(false);
    let pt_bytes = enc_pt.as_bytes();
    keccak256(&pt_bytes[1..])[12..].try_into().unwrap()
}

/// Signs a message with a private key and returns a signature with recovery id
/// signature of full_hash == keccak256(keystore_address || data_hash || msg_hash)
pub fn ecdsa_sign(pk: [u8; 32], full_hash: [u8; 32]) -> [u8; 65] {
    let sk = SigningKey::from_bytes(&pk.into()).unwrap();
    let vk = sk.verifying_key();
    let eoa_addr = calculate_eoa_addr(vk);
    let signature = sk.sign_prehash(&full_hash).unwrap();
    let recovery_id = calculate_recovery_id(&signature, full_hash, eoa_addr);
    let mut sig_bytes = signature.to_bytes().to_vec();
    sig_bytes.push(recovery_id);
    sig_bytes
        .as_slice()
        .try_into()
        .expect("Signature with recovery id must be 65 bytes")
}
