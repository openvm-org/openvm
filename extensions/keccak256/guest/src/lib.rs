#![no_std]

#[cfg(target_os = "zkvm")]
use openvm_platform::alloc::AlignedBuf;

pub const OPCODE: u8 = 0x0b;
pub const KECCAKF_FUNCT3: u8 = 0b100;
pub const KECCAKF_FUNCT7: u8 = 0;
pub const XORIN_FUNCT3: u8 = 0b100;
pub const XORIN_FUNCT7: u8 = 3;

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
