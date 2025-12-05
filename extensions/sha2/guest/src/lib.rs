#![no_std]

#[cfg(target_os = "zkvm")]
use openvm_platform::alloc::AlignedBuf;

/// This is custom-0 defined in RISC-V spec document
pub const OPCODE: u8 = 0x0b;
pub const SHA2_FUNCT3: u8 = 0b100;

// There is no Sha384 enum variant because the SHA-384 compression function is
// the same as the SHA-512 compression function.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum Sha2BaseFunct7 {
    Sha256 = 0x1,
    Sha512 = 0x2,
}

/// zkvm native implementation of sha256 compression function
/// # Safety
///
/// The VM accepts the previous hash state and the next block of input, and writes the
/// new hash state.
/// - `prev_state` must point to a buffer of at least 32 bytes, storing the previous hash state as 8
///   32-bit words in little-endian order
/// - `input` must point to a buffer of at least 64 bytes
/// - `output` must point to a buffer of at least 32 bytes. It will be filled with the new hash
///   state as 8 32-bit words in little-endian order
///
/// [`sha2-256`]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf
#[cfg(target_os = "zkvm")]
#[inline(always)]
#[no_mangle]
pub extern "C" fn zkvm_sha256_impl(prev_state: *const u8, input: *const u8, output: *mut u8) {
    // SAFETY: we handle all cases where `prev_state`, `input`, or `output` are not aligned to 4
    // bytes.

    // The minimum alignment required for the buffers
    const MIN_ALIGN: usize = 4;
    unsafe {
        let prev_state_is_aligned = prev_state as usize % MIN_ALIGN == 0;
        let input_is_aligned = input as usize % MIN_ALIGN == 0;
        let output_is_aligned = output as usize % MIN_ALIGN == 0;

        let prev_state_ptr = if prev_state_is_aligned {
            prev_state
        } else {
            AlignedBuf::new(prev_state, 32, MIN_ALIGN).ptr
        };

        let input_ptr = if input_is_aligned {
            input
        } else {
            AlignedBuf::new(input, 64, MIN_ALIGN).ptr
        };

        let output_ptr = if output_is_aligned {
            output
        } else {
            AlignedBuf::uninit(32, MIN_ALIGN).ptr
        };

        __native_sha256_compress(prev_state_ptr, input_ptr, output_ptr);

        if !output_is_aligned {
            core::ptr::copy_nonoverlapping(output_ptr, output, 32);
        }
    }
}

/// zkvm native implementation of sha512 compression function
/// # Safety
///
/// The VM accepts the previous hash state and the next block of input, and writes the
/// new hash state.
/// - `prev_state` must point to a buffer of at least 64 bytes, storing the previous hash state as 8
///   64-bit words in little-endian order
/// - `input` must point to a buffer of at least 128 bytes
/// - `output` must point to a buffer of at least 64 bytes. It will be filled with the new hash
///   state as 8 64-bit words in little-endian order
///
/// [`sha2-512`]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf
#[cfg(target_os = "zkvm")]
#[inline(always)]
#[no_mangle]
pub extern "C" fn zkvm_sha512_impl(prev_state: *const u8, input: *const u8, output: *mut u8) {
    // SAFETY: we handle all cases where `prev_state`, `input`, or `output` are not aligned to 4
    // bytes.

    // The minimum alignment required for the buffers
    const MIN_ALIGN: usize = 4;
    unsafe {
        let prev_state_is_aligned = prev_state as usize % MIN_ALIGN == 0;
        let input_is_aligned = input as usize % MIN_ALIGN == 0;
        let output_is_aligned = output as usize % MIN_ALIGN == 0;

        let prev_state_ptr = if prev_state_is_aligned {
            prev_state
        } else {
            AlignedBuf::new(prev_state, 64, MIN_ALIGN).ptr
        };

        let input_ptr = if input_is_aligned {
            input
        } else {
            AlignedBuf::new(input, 128, MIN_ALIGN).ptr
        };

        let output_ptr = if output_is_aligned {
            output
        } else {
            AlignedBuf::uninit(64, MIN_ALIGN).ptr
        };

        __native_sha512_compress(prev_state_ptr, input_ptr, output_ptr);

        if !output_is_aligned {
            core::ptr::copy_nonoverlapping(output_ptr, output, 64);
        }
    }
}

/// sha256 compression function intrinsic binding
///
/// # Safety
///
/// The VM accepts the previous hash state and the next block of input, and writes the
/// 32-byte hash.
/// - `prev_state` must point to a buffer of at least 32 bytes, storing the previous hash state as 8
///   32-bit words in little-endian order
/// - `input` must point to a buffer of at least 64 bytes
/// - `output` must point to a buffer of at least 32 bytes. It will be filled with the new hash
///   state as 8 32-bit words in little-endian order
#[cfg(target_os = "zkvm")]
#[inline(always)]
fn __native_sha256_compress(prev_state: *const u8, input: *const u8, output: *mut u8) {
    openvm_platform::custom_insn_r!(opcode = OPCODE, funct3 = SHA2_FUNCT3, funct7 = Sha2BaseFunct7::Sha256 as u8, rd = In output, rs1 = In prev_state, rs2 = In input);
}

/// sha512 intrinsic binding
///
/// # Safety
///
/// The VM accepts the previous hash state and the next block of input, and writes the
/// 64-byte hash.
/// - `prev_state` must point to a buffer of at least 32 bytes, storing the previous hash state as 8
///   64-bit words in little-endian order
/// - `input` must point to a buffer of at least 128 bytes
/// - `output` must point to a buffer of at least 64 bytes. It will be filled with the new hash
///   state as 8 64-bit words in little-endian order
#[cfg(target_os = "zkvm")]
#[inline(always)]
fn __native_sha512_compress(prev_state: *const u8, input: *const u8, output: *mut u8) {
    openvm_platform::custom_insn_r!(opcode = OPCODE, funct3 = SHA2_FUNCT3, funct7 = Sha2BaseFunct7::Sha512 as u8, rd = In output, rs1 = In prev_state, rs2 = In input);
}
