#![cfg_attr(not(feature = "std"), no_std)]

// TODO[arayi]: Revisit this
/// This is custom-0 defined in RISC-V spec document
pub const OPCODE: u8 = 0;
pub const SHA256_FUNCT3: u8 = 0;

#[cfg(target_os = "zkvm")]
use core::mem::MaybeUninit;

/// The sha256 cryptographic hash function.
#[inline(always)]
pub fn sha256(input: &[u8]) -> [u8; 32] {
    #[cfg(not(target_os = "zkvm"))]
    {
        use sha2::{Digest, Sha256};
        let mut output = [0u8; 32];
        let mut hasher = Sha256::new();
        hasher.update(input);
        output.copy_from_slice(hasher.finalize().as_ref());
        output
    }
    #[cfg(target_os = "zkvm")]
    {
        let mut output = MaybeUninit::<[u8; 32]>::uninit();
        zkvm_sha256_impl(input.as_ptr(), input.len(), output.as_mut_ptr() as *mut u8);
        unsafe { output.assume_init() }
    }
}

/// zkvm native implementation of sha256
/// # Safety
///
/// The VM accepts the preimage by pointer and length, and writes the
/// 32-byte hash.
/// - `bytes` must point to an input buffer at least `len` long.
/// - `output` must point to a buffer that is at least 32-bytes long.
///
/// [`sha2-256`]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf
#[cfg(target_os = "zkvm")]
#[inline(always)]
#[no_mangle]
extern "C" fn zkvm_sha256_impl(bytes: *const u8, len: usize, output: *mut u8) {
    openvm_platform::custom_insn_r!(OPCODE, SHA256_FUNCT3, 0x0, output, bytes, len);
}
