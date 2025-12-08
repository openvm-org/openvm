#![no_std]

#[cfg(target_os = "zkvm")]
extern crate alloc;
#[cfg(target_os = "zkvm")]
use openvm_platform::alloc::AlignedBuf;

// TODO: XORIN is not defined yet

pub const OPCODE: u8 = 0x0b;
pub const KECCAKF_FUNCT3: u8 = 0b100;
pub const KECCAKF_FUNCT7: u8 = 0;
pub const XORIN_FUNCT3: u8 = 0b100;
pub const XORIN_FUNCT7: u8 = 1;

// TODO: fill in the custom-0 const 

#[cfg(target_os = "zkvm")]
#[inline(always)]
#[no_mangle]
pub extern "C" fn native_xorin(buffer: *mut u8, input: *const u8, len: usize) {
    const MIN_ALIGN: usize = 4;
    unsafe {
        if buffer as usize % MIN_ALIGN != 0 {
            let aligned_buffer = AlignedBuf::new(buffer, len, MIN_ALIGN);
            if input as usize % MIN_ALIGN != 0 {
                let aligned_input = AlignedBuf::new(input, len, MIN_ALIGN);
                __native_xorin(aligned_buffer.ptr, aligned_input.ptr, len);
                core::ptr::copy_nonoverlapping(aligned_buffer.ptr as *const u8, buffer, len);
            } else {
                __native_xorin(aligned_buffer.ptr, input, len);
                core::ptr::copy_nonoverlapping(aligned_buffer.ptr as *const u8, buffer, len);
            }
        } else {
            if input as usize % MIN_ALIGN != 0 {
                let aligned_input = AlignedBuf::new(input, len, MIN_ALIGN);
                __native_xorin(buffer, aligned_input.ptr, len);
            } else {
                __native_xorin(buffer, input, len);
            }
        };
    }
}

#[cfg(target_os = "zkvm")]
#[inline(always)]
fn __native_xorin(buffer: *mut u8, input: *const u8, len: usize) {
    openvm_platform::custom_insn_r!(
        opcode = OPCODE,
        funct3 = XORIN_FUNCT3,
        funct7 = XORIN_FUNCT7,
        rd = InOut buffer,
        rs1 = In input,
        rs2 = In len
    );
}
