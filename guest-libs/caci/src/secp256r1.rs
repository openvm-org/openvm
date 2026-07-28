//! `zkvm_secp256r1_verify`, backed by the OpenVM ecc extension via the `openvm-p256`.

use openvm_p256::ecdsa::{signature::hazmat::PrehashVerifier, Signature, VerifyingKey};

use crate::types::{ZkvmSecp256r1Hash, ZkvmSecp256r1Pubkey, ZkvmSecp256r1Signature, ZkvmStatus};

/// Verifies an ECDSA/secp256r1 (P-256) signature over a prehashed message.
///
/// Returns [`ZkvmStatus::Fail`] if any pointer is NULL, the public key is not non-identity point on
/// EC, or the signature is not a valid encoding. Otherwise returns [`ZkvmStatus::Ok`]
/// and writes the verification result to `verified`.
///
/// # Safety
///
/// - `msg`, `sig`, and `pubkey` must be valid for reads of 32, 64, and 64 bytes respectively.
/// - `verified` must be valid for writes of 1 byte.
#[no_mangle]
pub unsafe extern "C" fn zkvm_secp256r1_verify(
    msg: *const ZkvmSecp256r1Hash,
    sig: *const ZkvmSecp256r1Signature,
    pubkey: *const ZkvmSecp256r1Pubkey,
    verified: *mut bool,
) -> ZkvmStatus {
    if msg.is_null() || sig.is_null() || pubkey.is_null() || verified.is_null() {
        return ZkvmStatus::Fail;
    }
    // SAFETY: all pointers are non-NULL per the checks above and valid.
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
