#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(target_os = "zkvm")]
use core::mem::MaybeUninit;

/// This is custom-0 defined in RISC-V spec document
pub const OPCODE: u8 = 0x0b;
pub const FUNCT3: u8 = 0b100;

/// The keccak256 cryptographic hash function.
#[inline(always)]
pub fn keccak256(input: &[u8]) -> [u8; 32] {
    #[cfg(not(target_os = "zkvm"))]
    {
        let mut output = [0u8; 32];
        set_keccak256(input, &mut output);
        output
    }
    #[cfg(target_os = "zkvm")]
    {
        let mut output = MaybeUninit::<[u8; 32]>::uninit();
        native_keccak256(input.as_ptr(), input.len(), output.as_mut_ptr() as *mut u8);
        unsafe { output.assume_init() }
    }
}

/// Native hook for keccak256 for use with `alloy-primitives` "native-keccak" feature.
///
/// # Safety
///
/// The VM accepts the preimage by pointer and length, and writes the
/// 32-byte hash.
/// - `bytes` must point to an input buffer at least `len` long.
/// - `output` must point to a buffer that is at least 32-bytes long.
///
/// [`keccak256`]: https://en.wikipedia.org/wiki/SHA-3
/// [`sha3`]: https://docs.rs/sha3/latest/sha3/
/// [`tiny_keccak`]: https://docs.rs/tiny-keccak/latest/tiny_keccak/
#[cfg(target_os = "zkvm")]
#[inline(always)]
#[no_mangle]
extern "C" fn native_keccak256(bytes: *const u8, len: usize, output: *mut u8) {
    axvm_platform::custom_insn_r!(OPCODE, FUNCT3, 0x0, output, bytes, len);
}

/// Sets `output` to the keccak256 hash of `input`.
pub fn set_keccak256(input: &[u8], output: &mut [u8; 32]) {
    #[cfg(not(target_os = "zkvm"))]
    {
        use tiny_keccak::Hasher;
        let mut hasher = tiny_keccak::Keccak::v256();
        hasher.update(input);
        hasher.finalize(output);
    }
    #[cfg(target_os = "zkvm")]
    native_keccak256(input.as_ptr(), input.len(), output.as_mut_ptr() as *mut u8);
}
