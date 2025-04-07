#![no_std]

/// This is custom-0 defined in RISC-V spec document
pub const OPCODE: u8 = 0x0b;
pub const SHA2_FUNCT3: u8 = 0b100;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum Sha2BaseFunct7 {
    Sha256 = 0x1,
    Sha512 = 0x2,
    Sha384 = 0x3,
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

/// zkvm native implementation of sha384
/// # Safety
///
/// The VM accepts the preimage by pointer and length, and writes the
/// 48-byte hash.
/// - `bytes` must point to an input buffer at least `len` long.
/// - `output` must point to a buffer that is at least 64-bytes long. The first 48 bytes written
///   will be the SHA-384 digest. The last 16 bytes are zeros.
///
/// [`sha2-384`]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf
#[cfg(target_os = "zkvm")]
#[inline(always)]
#[no_mangle]
extern "C" fn zkvm_sha384_impl(bytes: *const u8, len: usize, output: *mut u8) {
    openvm_platform::custom_insn_r!(opcode = OPCODE, funct3 = SHA2_FUNCT3, funct7 = Sha2BaseFunct7::Sha384 as u8, rd = In output, rs1 = In bytes, rs2 = In len);
}
