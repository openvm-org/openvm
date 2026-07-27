//! `zkvm_sha256`, backed by the OpenVM sha2 extension.

use openvm_sha2::Sha256;

use crate::types::{ZkvmSha256Hash, ZkvmStatus};

/// Computes the SHA-256 hash of `data[..len]` and writes it to `output`.
///
/// Returns [`ZkvmStatus::Fail`] on a NULL `output`, or a NULL `data` when `len > 0`.
/// Returns [`ZkvmStatus::Ok`] otherwise.
///
/// # Safety
///
/// - `data` must be valid for reads of `len` bytes (ignored when `len == 0`).
/// - `output` must be valid for writes of 32 bytes.
#[no_mangle]
pub unsafe extern "C" fn zkvm_sha256(
    data: *const u8,
    len: usize,
    output: *mut ZkvmSha256Hash,
) -> ZkvmStatus {
    if output.is_null() {
        return ZkvmStatus::Fail;
    }
    if len > 0 && data.is_null() {
        return ZkvmStatus::Fail;
    }
    // SAFETY: `data` is non-NULL when `len > 0` per the check above and valid per the caller
    // contract; the empty slice is used when `len == 0`, so a NULL `data` is never dereferenced.
    let input: &[u8] = if len == 0 {
        &[]
    } else {
        unsafe { core::slice::from_raw_parts(data, len) }
    };
    let digest = Sha256::digest(input);
    // SAFETY: `output` is non-NULL per the check above and valid per the caller contract.
    unsafe {
        (*output).data = digest;
    }
    ZkvmStatus::Ok
}
