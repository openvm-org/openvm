#![no_std]

#[cfg(any(feature = "bn254", feature = "bls12_381"))]
mod operations;

#[allow(unused_imports)]
#[cfg(any(feature = "bn254", feature = "bls12_381"))]
pub(crate) use operations::*;

/// Types for BLS12-381 curve with intrinsic functions.
#[cfg(feature = "bls12_381")]
pub mod bls12_381;
/// Types for BN254 curve with intrinsic functions.
#[cfg(feature = "bn254")]
pub mod bn254;

pub use openvm_pairing_guest::pairing::PairingCheck;

#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
#[cfg_attr(feature = "rvr-checkpoint-tests", inline(never))]
#[cfg_attr(not(feature = "rvr-checkpoint-tests"), inline(always))]
pub(crate) unsafe fn materialize_pairing_hint(ptr: *mut u8, dwords: usize) {
    openvm_riscv_guest::hint_buffer_chunked(ptr, dwords);
}
