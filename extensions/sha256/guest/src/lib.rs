#![no_std]

/// This is custom-0 defined in RISC-V spec document
pub const OPCODE: u8 = 0x0b;
pub const SHA2_FUNCT3: u8 = 0b100;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum Sha2BaseFunct7 {
    Sha256 = 0x1,
    Sha512 = 0x2,
}

/// The sha256 cryptographic hash function.
#[inline(always)]
pub fn sha256(input: &[u8]) -> [u8; 32] {
    let mut output = [0u8; 32];
    set_sha256(input, &mut output);
    output
}

/// The sha512 cryptographic hash function.
#[inline(always)]
pub fn sha512(input: &[u8]) -> [u8; 64] {
    let mut output = [0u8; 64];
    set_sha512(input, &mut output);
    output
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
    openvm_platform::custom_insn_r!(opcode = OPCODE, funct3 = SHA2_FUNCT3, funct7 = Sha2BaseFunct7::Sha256 as u8, rd = In output, rs1 = In bytes, rs2 = In len);
}

/// zkvm native implementation of sha512
/// # Safety
///
/// The VM accepts the preimage by pointer and length, and writes the
/// 64-byte hash.
/// - `bytes` must point to an input buffer at least `len` long.
/// - `output` must point to a buffer that is at least 64-bytes long.
///
/// [`sha2-512`]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf
#[cfg(target_os = "zkvm")]
#[inline(always)]
#[no_mangle]
extern "C" fn zkvm_sha512_impl(bytes: *const u8, len: usize, output: *mut u8) {
    openvm_platform::custom_insn_r!(opcode = OPCODE, funct3 = SHA2_FUNCT3, funct7 = Sha2BaseFunct7::Sha512 as u8, rd = In output, rs1 = In bytes, rs2 = In len);
}

/// Sets `output` to the sha256 hash of `input`.
pub fn set_sha256(input: &[u8], output: &mut [u8; 32]) {
    #[cfg(not(target_os = "zkvm"))]
    {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(input);
        output.copy_from_slice(hasher.finalize().as_ref());
    }
    #[cfg(target_os = "zkvm")]
    {
        zkvm_sha256_impl(input.as_ptr(), input.len(), output.as_mut_ptr() as *mut u8);
    }
}

/// Sets `output` to the sha512 hash of `input`.
pub fn set_sha512(input: &[u8], output: &mut [u8; 64]) {
    #[cfg(not(target_os = "zkvm"))]
    {
        use sha2::{Digest, Sha512};
        let mut hasher = Sha512::new();
        hasher.update(input);
        output.copy_from_slice(hasher.finalize().as_ref());
    }
    #[cfg(target_os = "zkvm")]
    {
        zkvm_sha512_impl(input.as_ptr(), input.len(), output.as_mut_ptr() as *mut u8);
    }
}
