//! `zkvm_secp256k1_verify` and `zkvm_secp256k1_ecrecover`, backed by the OpenVM ecc extension
//! via the `openvm-k256`.

use openvm_k256::ecdsa::{signature::hazmat::PrehashVerifier, RecoveryId, Signature, VerifyingKey};

use crate::types::{ZkvmSecp256k1Hash, ZkvmSecp256k1Pubkey, ZkvmSecp256k1Signature, ZkvmStatus};

/// Verifies an ECDSA/secp256k1 signature over a prehashed message.
///
/// Returns [`ZkvmStatus::Fail`] if any pointer is NULL, the public key is not a non-identity affine
/// point on EC, or the signature is not a valid encoding. Otherwise returns [`ZkvmStatus::Ok`] and
/// writes the verification result to `verified`.
///
/// # Safety
///
/// - `msg`, `sig`, and `pubkey` must be valid for reads of 32, 64, and 64 bytes respectively.
/// - `verified` must be valid for writes of 1 byte.
#[no_mangle]
pub unsafe extern "C" fn zkvm_secp256k1_verify(
    msg: *const ZkvmSecp256k1Hash,
    sig: *const ZkvmSecp256k1Signature,
    pubkey: *const ZkvmSecp256k1Pubkey,
    verified: *mut bool,
) -> ZkvmStatus {
    if msg.is_null() || sig.is_null() || pubkey.is_null() || verified.is_null() {
        return ZkvmStatus::Fail;
    }
    // SAFETY: all pointers are non-NULL per the checks above and valid per the caller contract.
    unsafe {
        let mut sec1 = [0u8; 65];
        sec1[0] = 0x04;
        sec1[1..].copy_from_slice(&(*pubkey).data);
        let Ok(vk) = VerifyingKey::from_sec1_bytes(&sec1) else {
            return ZkvmStatus::Fail;
        };
        let sig_bytes: &[u8] = &(*sig).data;
        let Ok(signature) = Signature::try_from(sig_bytes) else {
            return ZkvmStatus::Fail;
        };
        *verified = vk.verify_prehash(&(*msg).data, &signature).is_ok();
    }
    ZkvmStatus::Ok
}

/// Recovers the public key that signed a prehashed message (ecrecover, precompile 0x01).
///
/// Returns [`ZkvmStatus::Fail`] if any pointer is NULL, `recid > 3`, the signature is not a
/// valid encoding, or no public key can be recovered. Otherwise returns [`ZkvmStatus::Ok`]
/// and writes the uncompressed public key to `output`.
///
/// # Safety
///
/// - `msg` and `sig` must be valid for reads of 32 and 64 bytes respectively.
/// - `output` must be valid for writes of 64 bytes.
#[no_mangle]
pub unsafe extern "C" fn zkvm_secp256k1_ecrecover(
    msg: *const ZkvmSecp256k1Hash,
    sig: *const ZkvmSecp256k1Signature,
    recid: u8,
    output: *mut ZkvmSecp256k1Pubkey,
) -> ZkvmStatus {
    if msg.is_null() || sig.is_null() || output.is_null() {
        return ZkvmStatus::Fail;
    }
    let Some(recovery_id) = RecoveryId::from_byte(recid) else {
        return ZkvmStatus::Fail;
    };
    // SAFETY: all pointers are non-NULL per the checks above and valid per the caller contract.
    unsafe {
        let Ok(vk) =
            VerifyingKey::recover_from_prehash_noverify(&(*msg).data, &(*sig).data, recovery_id)
        else {
            return ZkvmStatus::Fail;
        };
        let sec1 = vk.to_sec1_bytes(false);
        if sec1.len() != 65 {
            return ZkvmStatus::Fail;
        }
        (*output).data.copy_from_slice(&sec1[1..]);
    }
    ZkvmStatus::Ok
}
