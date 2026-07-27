//! `zkvm_keccak256`, backed by the OpenVM keccak256 extension.

use crate::types::{ZkvmKeccak256Hash, ZkvmStatus};

/// Computes the Keccak-256 hash of `data[..len]` and writes it to `output`.
///
/// Returns [`ZkvmStatus::Fail`] on a NULL `output`, or a NULL `data` when `len > 0`.
/// Returns [`ZkvmStatus::Ok`] otherwise.
///
/// # Safety
///
/// - `data` must be valid for reads of `len` bytes (ignored when `len == 0`).
/// - `output` must be valid for writes of 32 bytes.
#[no_mangle]
pub unsafe extern "C" fn zkvm_keccak256(
    data: *const u8,
    len: usize,
    output: *mut ZkvmKeccak256Hash,
) -> ZkvmStatus {
    if output.is_null() {
        return ZkvmStatus::Fail;
    }
    if len > 0 && data.is_null() {
        return ZkvmStatus::Fail;
    }
    // SAFETY: the caller guarantees the pointer contracts above; `ZkvmKeccak256Hash` is a
    // `#[repr(C)]` wrapper over `[u8; 32]`, so `output.data` is a valid 32-byte destination.
    // `native_keccak256` never dereferences `data` when `len == 0`.
    unsafe {
        openvm_keccak256::native_keccak256(data, len, (*output).data.as_mut_ptr());
    }
    ZkvmStatus::Ok
}
