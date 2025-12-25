#![no_std]

#[cfg(target_os = "zkvm")]
extern crate alloc;
#[cfg(target_os = "zkvm")]
use openvm_platform::alloc::AlignedBuf;

pub const OPCODE: u8 = 0x0b;
pub const KECCAKF_FUNCT3: u8 = 0b100;
pub const KECCAKF_FUNCT7: u8 = 0;
pub const XORIN_FUNCT3: u8 = 0b100;
pub const XORIN_FUNCT7: u8 = 1;
pub const KECCAK_WIDTH_BYTES: usize = 200;
pub const KECCAK_RATE: usize = 136;
pub const KECCAK_OUTPUT_SIZE: usize = 32;
pub const MIN_ALIGN: usize = 4;

#[cfg(target_os = "zkvm")]
#[no_mangle]
pub extern "C" fn native_xorin(buffer: *mut u8, input: *const u8, len: usize) {
    if len == 0 {
        return;
    }
    unsafe {
        let aligned_buffer;
        let aligned_input;

        let actual_buffer = if buffer as usize % MIN_ALIGN == 0 {
            buffer
        } else {
            aligned_buffer = AlignedBuf::new(buffer, len, MIN_ALIGN);
            aligned_buffer.ptr
        };

        let actual_input = if input as usize % MIN_ALIGN == 0 {
            input
        } else {
            aligned_input = AlignedBuf::new(input, len, MIN_ALIGN);
            aligned_input.ptr
        };

        __native_xorin(actual_buffer, actual_input, len);

        if buffer as usize % MIN_ALIGN != 0 {
            core::ptr::copy_nonoverlapping(actual_buffer as *const u8, buffer, len);
        }
    }
}

#[cfg(target_os = "zkvm")]
#[no_mangle]
pub extern "C" fn native_keccakf(buffer: *mut u8) {
    unsafe {
        if buffer as usize % MIN_ALIGN == 0 {
            __native_keccakf(buffer);
        } else {
            let aligned_buffer = AlignedBuf::new(buffer, KECCAK_WIDTH_BYTES, MIN_ALIGN);
            __native_keccakf(aligned_buffer.ptr);
            core::ptr::copy_nonoverlapping(
                aligned_buffer.ptr as *const u8,
                buffer,
                KECCAK_WIDTH_BYTES,
            );
        }
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
#[no_mangle]
pub extern "C" fn native_keccak256(bytes: *const u8, len: usize, output: *mut u8) {
    // SAFETY: assuming safety assumptions of the inputs, we handle all cases where `bytes` or
    // `output` are not aligned to 4 bytes.
    unsafe {
        let aligned_bytes;
        let aligned_output;

        let actual_bytes = if len == 0 || bytes as usize % MIN_ALIGN == 0 {
            bytes
        } else {
            aligned_bytes = AlignedBuf::new(bytes, len, MIN_ALIGN);
            aligned_bytes.ptr
        };

        let actual_output = if output as usize % MIN_ALIGN == 0 {
            output
        } else {
            aligned_output = AlignedBuf::uninit(KECCAK_OUTPUT_SIZE, MIN_ALIGN);
            aligned_output.ptr
        };

        keccak256_impl(actual_bytes, len, actual_output);

        if output as usize % MIN_ALIGN != 0 {
            core::ptr::copy_nonoverlapping(actual_output as *const u8, output, KECCAK_OUTPUT_SIZE);
        }
    }
}

#[cfg(target_os = "zkvm")]
#[inline(always)]
fn keccak_update(buffer: &mut [u8; KECCAK_WIDTH_BYTES], input: *const u8, mut len: usize) -> usize {
    let buffer_ptr = buffer.as_mut_ptr();
    let mut offset = 0;
    let input_aligned = input as usize % MIN_ALIGN == 0;

    // Absorb full blocks
    while len >= KECCAK_RATE {
        if input_aligned {
            __native_xorin(buffer_ptr, unsafe { input.add(offset) }, KECCAK_RATE);
        } else {
            let mut block = [0u8; KECCAK_RATE];
            unsafe {
                core::ptr::copy_nonoverlapping(input.add(offset), block.as_mut_ptr(), KECCAK_RATE);
                __native_xorin(buffer_ptr, block.as_ptr(), KECCAK_RATE);
            }
        }
        unsafe {
            __native_keccakf(buffer_ptr);
        }
        offset += KECCAK_RATE;
        len -= KECCAK_RATE;
    }

    // Handle remaining bytes
    if len > 0 {
        unsafe {
            if input_aligned && len % MIN_ALIGN == 0 {
                __native_xorin(buffer_ptr, input.add(offset), len);
            } else {
                // Copy to aligned buffer, and pad length to MIN_ALIGN if needed.
                let adjusted_len = len.next_multiple_of(MIN_ALIGN);
                let mut padded_input = [0u8; KECCAK_RATE];
                core::ptr::copy_nonoverlapping(input.add(offset), padded_input.as_mut_ptr(), len);
                __native_xorin(buffer_ptr, padded_input.as_ptr(), adjusted_len);
            }
        }
    }

    len
}

#[cfg(target_os = "zkvm")]
#[inline(always)]
fn keccak_finalize(
    buffer: &mut [u8; KECCAK_WIDTH_BYTES],
    remaining: usize,
    output: *mut u8,
) {
    // Apply Keccak padding (pad10*1)
    buffer[remaining] ^= 0x01;
    buffer[KECCAK_RATE - 1] ^= 0x80;

    // Final permutation
    unsafe {
        __native_keccakf(buffer.as_mut_ptr());

        // Extract output
        core::ptr::copy_nonoverlapping(buffer.as_ptr(), output, KECCAK_OUTPUT_SIZE);
    }
}

#[cfg(target_os = "zkvm")]
#[inline(always)]
fn keccak256_impl(input: *const u8, len: usize, output: *mut u8) {
    let mut buffer = [0u8; KECCAK_WIDTH_BYTES];
    let remaining = keccak_update(&mut buffer, input, len);
    keccak_finalize(&mut buffer, remaining, output);
}

#[cfg(target_os = "zkvm")]
#[inline(always)]
fn __native_xorin(mut buffer: *mut u8, input: *const u8, len: usize) {
    openvm_platform::custom_insn_r!(
        opcode = OPCODE,
        funct3 = XORIN_FUNCT3,
        funct7 = XORIN_FUNCT7,
        rd = InOut buffer,
        rs1 = In input,
        rs2 = In len
    );
}

#[cfg(target_os = "zkvm")]
#[inline(always)]
fn __native_keccakf(mut buffer: *mut u8) {
    openvm_platform::custom_insn_r!(
        opcode = OPCODE,
        funct3 = KECCAKF_FUNCT3,
        funct7 = KECCAKF_FUNCT7,
        rd = InOut buffer,
        rs1 = Const "x0",
        rs2 = Const "x0",
    );
}
