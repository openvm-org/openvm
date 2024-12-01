use axvm::intrinsics::keccak256;
use k256::{
    ecdsa::{signature::SignerMut, Signature, SigningKey},
    elliptic_curve::{bigint::Encoding, PrimeField},
    Scalar, U256,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// keystore_address = keccak256(salt, keccak256([eoa_addrs]), keccak256(vk))
pub fn generate_keystore_address(
    salt: [u8; 32],
    eoa_addrs: Vec<[u8; 20]>,
    vk_hash: [u8; 32],
) -> [u8; 32] {
    // concatenate EOAs
    let concat_eoas = eoa_addrs
        .iter()
        .fold(vec![], |acc, addr| [acc, addr.to_vec()].concat());
    let data_hash = keccak256(concat_eoas.as_slice());

    let concat_all = [salt.to_vec(), data_hash.to_vec(), vk_hash.to_vec()].concat();
    keccak256(concat_all.as_slice())
}

pub fn generate_salt(seed: u64) -> [u8; 32] {
    let mut rng = StdRng::seed_from_u64(seed);
    let salt: [u8; 32] = rng.gen();
    salt
}

pub fn ecdsa_sign(pk: [u8; 32], msg_hash: &[u8]) -> [u8; 65] {
    let mut sk = SigningKey::from_bytes(&pk.into()).unwrap();
    let signature = sk.sign(msg_hash);
    let recovery_id = calculate_recovery_id(&signature);
    let mut sig_bytes = signature.to_bytes().to_vec();
    sig_bytes.push(recovery_id);
    sig_bytes
        .as_slice()
        .try_into()
        .expect("Signature with recovery id must be 65 bytes")
}

fn calculate_recovery_id(signature: &Signature) -> u8 {
    // Extract r and s
    let r = Scalar::from_repr(signature.r().into()).unwrap();
    let s = Scalar::from_repr(signature.s().into()).unwrap();

    // Curve order for secp256k1
    let curve_order = U256::from_be_hex(Scalar::MODULUS).to_be_bytes();
    let curve_order = Scalar::from_repr(curve_order.into()).unwrap();

    // Check if s is greater than half the curve order
    let is_s_high = s > (curve_order >> 1);

    // Determine the parity of r (0 if even, 1 if odd)
    let r_parity = r.is_odd().unwrap_u8();

    // Combine r_parity and is_s_high bits
    (r_parity as u8) | ((is_s_high as u8) << 1)
}
