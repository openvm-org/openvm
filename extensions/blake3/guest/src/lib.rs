#![no_std]

#[cfg(target_os = "zkvm")]
use openvm_platform::alloc::AlignedBuf;

/// This is custom-0 opcode defined in RISC-V spec document
pub const OPCODE: u8 = 0x0b;
/// Function selector (same as SHA256 - both are hash functions)
pub const BLAKE3_FUNCT3: u8 = 0b100;
/// Unique identifier for BLAKE3 (SHA256=0x1, Keccak256=0x0, BLAKE3=0x2)
pub const BLAKE3_FUNCT7: u8 = 0x2;

/// Native hook for blake3
///
/// # Safety
///
/// - `bytes` must point to an input buffer at least `len` long.
/// - `output` must point to a buffer that is at least 32-bytes long.
#[cfg(target_os = "zkvm")]
#[inline(always)]
#[no_mangle]
pub extern "C" fn zkvm_blake3_impl(bytes: *const u8, len: usize, output: *mut u8) {
    const MIN_ALIGN: usize = 4;
    const INPUT_ALIGN: usize = 16;
    const OUTPUT_ALIGN: usize = 32;

    unsafe {
        if bytes as usize % MIN_ALIGN != 0 {
            let aligned_buff = AlignedBuf::new(bytes, len, INPUT_ALIGN);
            if output as usize % MIN_ALIGN != 0 {
                let aligned_out = AlignedBuf::uninit(32, OUTPUT_ALIGN);
                __native_blake3(aligned_buff.ptr, len, aligned_out.ptr);
                core::ptr::copy_nonoverlapping(aligned_out.ptr as *const u8, output, 32);
            } else {
                __native_blake3(aligned_buff.ptr, len, output);
            }
        } else if output as usize % MIN_ALIGN != 0 {
            let aligned_out = AlignedBuf::uninit(32, OUTPUT_ALIGN);
            __native_blake3(bytes, len, aligned_out.ptr);
            core::ptr::copy_nonoverlapping(aligned_out.ptr as *const u8, output, 32);
        } else {
            __native_blake3(bytes, len, output);
        }
    }
}

/// BLAKE3 intrinsic binding
///
/// # Safety
///
/// - `bytes` must point to an input buffer at least `len` long.
/// - `output` must point to a buffer that is at least 32-bytes long.
/// - `bytes` and `output` must be 4-byte aligned.
#[cfg(target_os = "zkvm")]
#[inline(always)]
fn __native_blake3(bytes: *const u8, len: usize, output: *mut u8) {
    openvm_platform::custom_insn_r!(
        opcode = OPCODE,
        funct3 = BLAKE3_FUNCT3,
        funct7 = BLAKE3_FUNCT7,
        rd = In output,
        rs1 = In bytes,
        rs2 = In len
    );
}
