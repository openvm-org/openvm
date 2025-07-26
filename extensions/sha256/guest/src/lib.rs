#![no_std]

#[cfg(target_os = "zkvm")]
use openvm_platform::alloc::AlignedBuf;

/// This is custom-0 defined in RISC-V spec document
pub const OPCODE: u8 = 0x0b;
pub const SHA256_FUNCT3: u8 = 0b100;
pub const SHA256_FUNCT7: u8 = 0x1;

/// Native hook for sha256
///
/// # Safety
///
/// The VM accepts the preimage by pointer and length, and writes the
/// 32-byte hash.
/// - `bytes` must point to an input buffer at least `len` long.
/// - `output` must point to a buffer that is at least 32-bytes long.
///
/// [`sha2`]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf
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
    openvm_platform::custom_insn_r!(opcode = OPCODE, funct3 = SHA256_FUNCT3, funct7 = SHA256_FUNCT7, rd = In output, rs1 = In bytes, rs2 = In len);
}
