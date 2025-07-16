#![no_std]

#[cfg(target_os = "zkvm")]
use openvm_platform::alloc::AlignedBuf;

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
pub extern "C" fn zkvm_sha256_impl(bytes: *const u8, len: usize, output: *mut u8) {
    // SAFETY: assuming safety assumptions of the inputs, we handle all cases where `bytes` or
    // `output` are not aligned to 4 bytes.
    // The minimum alignment required for the input and output buffers
    const MIN_ALIGN: usize = 4;
    // The preferred alignment for the input buffer, since the input is read in chunks of 16 bytes
    const INPUT_ALIGN: usize = 16;
    // The preferred alignment for the output buffer, since the output is written in chunks of 32
    // bytes
    const OUTPUT_ALIGN: usize = 32;
    unsafe {
        if bytes as usize % MIN_ALIGN != 0 {
            let aligned_buff = AlignedBuf::new(bytes, len, INPUT_ALIGN);
            if output as usize % MIN_ALIGN != 0 {
                let aligned_out = AlignedBuf::uninit(32, OUTPUT_ALIGN);
                __native_sha256(aligned_buff.ptr, len, aligned_out.ptr);
                core::ptr::copy_nonoverlapping(aligned_out.ptr as *const u8, output, 32);
            } else {
                __native_sha256(aligned_buff.ptr, len, output);
            }
        } else {
            if output as usize % MIN_ALIGN != 0 {
                let aligned_out = AlignedBuf::uninit(32, OUTPUT_ALIGN);
                __native_sha256(bytes, len, aligned_out.ptr);
                core::ptr::copy_nonoverlapping(aligned_out.ptr as *const u8, output, 32);
            } else {
                __native_sha256(bytes, len, output);
            }
        };
    }
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
pub extern "C" fn zkvm_sha512_impl(bytes: *const u8, len: usize, output: *mut u8) {
    // SAFETY: assuming safety assumptions of the inputs, we handle all cases where `bytes` or
    // `output` are not aligned to 4 bytes.
    // The minimum alignment required for the input and output buffers
    const MIN_ALIGN: usize = 4;
    // The preferred alignment for the input buffer, since the input is read in chunks of 32 bytes
    const INPUT_ALIGN: usize = 32;
    // The preferred alignment for the output buffer, since the output is written in chunks of 32
    // bytes
    const OUTPUT_ALIGN: usize = 32;
    unsafe {
        if bytes as usize % MIN_ALIGN != 0 {
            let aligned_buff = AlignedBuf::new(bytes, len, INPUT_ALIGN);
            if output as usize % MIN_ALIGN != 0 {
                let aligned_out = AlignedBuf::uninit(64, OUTPUT_ALIGN);
                __native_sha512(aligned_buff.ptr, len, aligned_out.ptr);
                core::ptr::copy_nonoverlapping(aligned_out.ptr as *const u8, output, 64);
            } else {
                __native_sha512(aligned_buff.ptr, len, output);
            }
        } else {
            if output as usize % MIN_ALIGN != 0 {
                let aligned_out = AlignedBuf::uninit(64, OUTPUT_ALIGN);
                __native_sha512(bytes, len, aligned_out.ptr);
                core::ptr::copy_nonoverlapping(aligned_out.ptr as *const u8, output, 64);
            } else {
                __native_sha512(bytes, len, output);
            }
        };
    }
}

/// zkvm native implementation of sha384
/// # Safety
///
/// The VM accepts the preimage by pointer and length, and writes the
/// 48-byte hash followed by 16-bytes of zeros.
/// - `bytes` must point to an input buffer at least `len` long.
/// - `output` must point to a buffer that is at least 64-bytes long.
///
/// [`sha2-512`]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf
#[cfg(target_os = "zkvm")]
#[inline(always)]
#[no_mangle]
pub extern "C" fn zkvm_sha384_impl(bytes: *const u8, len: usize, output: *mut u8) {
    // SAFETY: assuming safety assumptions of the inputs, we handle all cases where `bytes` or
    // `output` are not aligned to 4 bytes.
    // The minimum alignment required for the input and output buffers
    const MIN_ALIGN: usize = 4;
    // The preferred alignment for the input buffer, since the input is read in chunks of 32 bytes
    const INPUT_ALIGN: usize = 32;
    // The preferred alignment for the output buffer, since the output is written in chunks of 32
    // bytes
    const OUTPUT_ALIGN: usize = 32;
    unsafe {
        if bytes as usize % MIN_ALIGN != 0 {
            let aligned_buff = AlignedBuf::new(bytes, len, INPUT_ALIGN);
            if output as usize % MIN_ALIGN != 0 {
                let aligned_out = AlignedBuf::uninit(64, OUTPUT_ALIGN);
                __native_sha384(aligned_buff.ptr, len, aligned_out.ptr);
                core::ptr::copy_nonoverlapping(aligned_out.ptr as *const u8, output, 64);
            } else {
                __native_sha384(aligned_buff.ptr, len, output);
            }
        } else {
            if output as usize % MIN_ALIGN != 0 {
                let aligned_out = AlignedBuf::uninit(64, OUTPUT_ALIGN);
                __native_sha384(bytes, len, aligned_out.ptr);
                core::ptr::copy_nonoverlapping(aligned_out.ptr as *const u8, output, 64);
            } else {
                __native_sha384(bytes, len, output);
            }
        };
    }
}

/// sha256 intrinsic binding
///
/// # Safety
///
/// The VM accepts the preimage by pointer and length, and writes the
/// 32-byte hash.
/// - `bytes` must point to an input buffer at least `len` long.
/// - `output` must point to a buffer that is at least 32-bytes long.
/// - `bytes` and `output` must be 4-byte aligned.
#[cfg(target_os = "zkvm")]
#[inline(always)]
fn __native_sha256(bytes: *const u8, len: usize, output: *mut u8) {
    openvm_platform::custom_insn_r!(opcode = OPCODE, funct3 = SHA2_FUNCT3, funct7 = Sha2BaseFunct7::Sha256 as u8, rd = In output, rs1 = In bytes, rs2 = In len);
}

/// sha512 intrinsic binding
///
/// # Safety
///
/// The VM accepts the preimage by pointer and length, and writes the
/// 64-byte hash.
/// - `bytes` must point to an input buffer at least `len` long.
/// - `output` must point to a buffer that is at least 64-bytes long.
/// - `bytes` and `output` must be 4-byte aligned.
#[cfg(target_os = "zkvm")]
#[inline(always)]
fn __native_sha512(bytes: *const u8, len: usize, output: *mut u8) {
    openvm_platform::custom_insn_r!(opcode = OPCODE, funct3 = SHA2_FUNCT3, funct7 = Sha2BaseFunct7::Sha512 as u8, rd = In output, rs1 = In bytes, rs2 = In len);
}

/// sha384 intrinsic binding
///
/// # Safety
///
/// The VM accepts the preimage by pointer and length, and writes the
/// 48-byte hash followed by 16-bytes of zeros.
/// - `bytes` must point to an input buffer at least `len` long.
/// - `output` must point to a buffer that is at least 64-bytes long.
/// - `bytes` and `output` must be 4-byte aligned.
#[cfg(target_os = "zkvm")]
#[inline(always)]
fn __native_sha384(bytes: *const u8, len: usize, output: *mut u8) {
    openvm_platform::custom_insn_r!(opcode = OPCODE, funct3 = SHA2_FUNCT3, funct7 = Sha2BaseFunct7::Sha384 as u8, rd = In output, rs1 = In bytes, rs2 = In len);
}
