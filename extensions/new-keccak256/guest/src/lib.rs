#![no_std]

#[cfg(target_os = "zkvm")]
extern crate alloc;
#[cfg(target_os = "zkvm")]
use openvm_platform::alloc::AlignedBuf;

pub const OPCODE: u8 = 0x0b;
pub const KECCAKF_FUNCT3: u8 = 0b100;
pub const KECCAKF_FUNCT7: u8 = 0;
pub const XORIN_FUNCT3: u8 = 0b100;
pub const XORIN_FUNCT7: u8 = 2;
pub const KECCAK_WIDTH_BYTES: usize = 200;
pub const KECCAK_RATE: usize = 136;
pub const KECCAK_OUTPUT_SIZE: usize = 32;
pub const MIN_ALIGN: usize = 4;

#[cfg(target_os = "zkvm")]
#[repr(align(4))]
struct AlignedStackBuf<const N: usize> {
    data: [u8; N],
}

/// SAFETY: Caller must ensure:
/// - buffer and input are aligned to MIN_ALIGN
/// - len is a multiple of MIN_ALIGN
#[cfg(target_os = "zkvm")]
#[inline(always)]
unsafe fn native_xorin_unchecked(buffer: *mut u8, input: *const u8, len: usize) {
    __native_xorin(buffer, input, len);
}

#[cfg(target_os = "zkvm")]
#[no_mangle]
pub extern "C" fn native_xorin(buffer: *mut u8, input: *const u8, len: usize) {
    if len == 0 {
        return;
    }
    unsafe {
        let buffer_aligned = buffer as usize % MIN_ALIGN == 0;
        let input_aligned = input as usize % MIN_ALIGN == 0;
        let len_aligned = len % MIN_ALIGN == 0;
        let all_aligned = buffer_aligned && input_aligned && len_aligned;

        if all_aligned {
            __native_xorin(buffer, input, len);
        } else {
            let adjusted_len = len.next_multiple_of(MIN_ALIGN);
            let aligned_buffer;
            let aligned_input;

            let actual_buffer = if buffer_aligned && len_aligned {
                buffer
            } else {
                aligned_buffer = AlignedBuf::new(buffer, adjusted_len, MIN_ALIGN);
                aligned_buffer.ptr
            };

            let actual_input = if input_aligned && len_aligned {
                input
            } else {
                aligned_input = AlignedBuf::new(input, adjusted_len, MIN_ALIGN);
                aligned_input.ptr
            };

            __native_xorin(actual_buffer, actual_input, adjusted_len);

            if !buffer_aligned || !len_aligned {
                core::ptr::copy_nonoverlapping(actual_buffer as *const u8, buffer, len);
            }
        }
    }
}

/// SAFETY: Caller must ensure buffer is aligned to MIN_ALIGN
#[cfg(target_os = "zkvm")]
#[inline(always)]
unsafe fn native_keccakf_unchecked(buffer: *mut u8) {
    __native_keccakf(buffer);
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
    unsafe {
        let bytes_aligned = bytes as usize % MIN_ALIGN == 0;
        let output_aligned = output as usize % MIN_ALIGN == 0;

        let aligned_bytes;
        let aligned_output;

        let actual_bytes = if len == 0 || bytes_aligned {
            bytes
        } else {
            aligned_bytes = AlignedBuf::new(bytes, len, MIN_ALIGN);
            aligned_bytes.ptr
        };

        let actual_output = if output_aligned {
            output
        } else {
            aligned_output = AlignedBuf::uninit(KECCAK_OUTPUT_SIZE, MIN_ALIGN);
            aligned_output.ptr
        };

        keccak256_impl(actual_bytes, len, actual_output);

        if !output_aligned {
            core::ptr::copy_nonoverlapping(actual_output as *const u8, output, KECCAK_OUTPUT_SIZE);
        }
    }
}

/// SAFETY: This function is only called from native_keccak256 which ensures:
/// - input is aligned to MIN_ALIGN
/// - output is aligned to MIN_ALIGN
/// - All internal buffers are aligned by AlignedStackBuf
#[cfg(target_os = "zkvm")]
#[inline(always)]
unsafe fn keccak256_impl(input: *const u8, len: usize, output: *mut u8) {
    let mut buffer = AlignedStackBuf::<KECCAK_WIDTH_BYTES> {
        data: [0u8; KECCAK_WIDTH_BYTES],
    };
    let buffer_ptr = buffer.data.as_mut_ptr();

    let mut offset = 0;
    let mut remaining = len;

    // Absorb full blocks
    while remaining >= KECCAK_RATE {
        native_xorin_unchecked(buffer_ptr, input.add(offset), KECCAK_RATE);
        native_keccakf_unchecked(buffer_ptr);
        offset += KECCAK_RATE;
        remaining -= KECCAK_RATE;
    }

    // Handle remaining bytes
    if remaining > 0 {
        if remaining % MIN_ALIGN == 0 {
            native_xorin_unchecked(buffer_ptr, input.add(offset), remaining);
        } else {
            let adjusted_len = remaining.next_multiple_of(MIN_ALIGN);
            let mut padded_input = AlignedStackBuf::<KECCAK_RATE> {
                data: [0u8; KECCAK_RATE],
            };
            core::ptr::copy_nonoverlapping(
                input.add(offset),
                padded_input.data.as_mut_ptr(),
                remaining,
            );
            native_xorin_unchecked(buffer_ptr, padded_input.data.as_ptr(), adjusted_len);
        }
    }

    // Apply Keccak padding (pad10*1)
    buffer.data[remaining] ^= 0x01;
    buffer.data[KECCAK_RATE - 1] ^= 0x80;

    // Final permutation
    native_keccakf_unchecked(buffer_ptr);

    // Extract output
    core::ptr::copy_nonoverlapping(buffer.data.as_ptr(), output, KECCAK_OUTPUT_SIZE);
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
