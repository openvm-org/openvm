#![no_std]

#[cfg(target_os = "zkvm")]
use openvm_platform::alloc::AlignedBuf;

pub const OPCODE: u8 = 0x0b;
pub const POSEIDON2_FUNCT3: u8 = 0b111;
pub const POSEIDON2_FUNCT7: u8 = 0;

pub const POSEIDON2_WIDTH: usize = 16;
pub const POSEIDON2_STATE_BYTES: usize = 64;
pub const MIN_ALIGN: usize = 4;

/// Apply the Poseidon2 permutation to the 16-word (64-byte) state buffer.
///
/// # Safety
///
/// - `buffer` must point to a buffer of at least `POSEIDON2_STATE_BYTES` (64) bytes.
#[cfg(target_os = "zkvm")]
#[no_mangle]
pub unsafe extern "C" fn native_poseidon2_permute(buffer: *mut u8) {
    unsafe {
        if buffer as usize % MIN_ALIGN == 0 {
            __native_poseidon2_permute(buffer);
        } else {
            let aligned_buffer = AlignedBuf::new(buffer, POSEIDON2_STATE_BYTES, MIN_ALIGN);
            __native_poseidon2_permute(aligned_buffer.ptr);
            core::ptr::copy_nonoverlapping(
                aligned_buffer.ptr as *const u8,
                buffer,
                POSEIDON2_STATE_BYTES,
            );
        }
    }
}

#[cfg(target_os = "zkvm")]
#[inline(always)]
fn __native_poseidon2_permute(mut buffer: *mut u8) {
    openvm_platform::custom_insn_r!(
        opcode = OPCODE,
        funct3 = POSEIDON2_FUNCT3,
        funct7 = POSEIDON2_FUNCT7,
        rd = InOut buffer,
        rs1 = Const "x0",
        rs2 = Const "x0",
    );
}
