#![allow(unused_imports)]
use crate::{PhantomImm, MAX_HINT_BUFFER_DWORDS, PHANTOM_FUNCT3, SYSTEM_OPCODE};

/// Size in bytes of one hint stream word (`hint_store_u64` / one `hint_buffer_chunked` unit).
pub const HINT_WORD_BYTES: usize = 8;

/// Store the next 8 bytes from the hint stream to [[rd]_1]_2.
#[macro_export]
macro_rules! hint_store_u64 {
    ($x:expr) => {
        openvm_custom_insn::custom_insn_i!(
            opcode = openvm_riscv_guest::SYSTEM_OPCODE,
            funct3 = openvm_riscv_guest::HINT_FUNCT3,
            rd = In $x,
            rs1 = Const "x0",
            imm = Const 0,
        )
    };
}

/// Store the next 8*len bytes from the hint stream to [[rd]_1]_2.
#[macro_export]
macro_rules! hint_buffer_u64 {
    ($x:expr, $len:expr) => {
        if $len != 0 {
            openvm_custom_insn::custom_insn_i!(
                opcode = $crate::SYSTEM_OPCODE,
                funct3 = $crate::HINT_FUNCT3,
                rd = In $x,
                rs1 = In $len,
                imm = Const 1,
            )
        }
    };
}

/// Read hint buffer with automatic chunking for large reads.
/// Splits reads larger than MAX_HINT_BUFFER_DWORDS into multiple instructions.
///
/// # Safety
///
/// `ptr` must be valid for writes of `num_dwords * HINT_WORD_BYTES` bytes.
#[inline(always)]
pub unsafe fn hint_buffer_chunked(mut ptr: *mut u8, mut num_dwords: usize) {
    while num_dwords > 0 {
        let chunk = core::cmp::min(num_dwords, MAX_HINT_BUFFER_DWORDS);
        hint_buffer_u64!(ptr, chunk);
        ptr = ptr.add(chunk * HINT_WORD_BYTES);
        num_dwords -= chunk;
    }
}

/// Read `nbytes` from the hint stream into `ptr`. The underlying instruction is
/// `HINT_WORD_BYTES`-granular; any trailing bytes (0..HINT_WORD_BYTES) are written
/// via a stack scratch dword so the caller's buffer isn't over-written.
///
/// # Safety
///
/// `ptr` must be valid for writes of `nbytes` bytes.
#[inline]
pub unsafe fn hint_buffer_bytes(ptr: *mut u8, nbytes: usize) {
    if nbytes == 0 {
        return;
    }
    let full_dwords = nbytes / HINT_WORD_BYTES;
    let trailing = nbytes & (HINT_WORD_BYTES - 1);
    if full_dwords > 0 {
        hint_buffer_chunked(ptr, full_dwords);
    }
    if trailing != 0 {
        let mut scratch: u64 = 0;
        hint_buffer_chunked(&mut scratch as *mut u64 as *mut u8, 1);
        let bytes = scratch.to_ne_bytes();
        core::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            ptr.add(full_dwords * HINT_WORD_BYTES),
            trailing,
        );
    }
}

/// Reset the hint stream with the next hint.
#[inline(always)]
pub fn hint_input() {
    openvm_custom_insn::custom_insn_i!(
        opcode = SYSTEM_OPCODE,
        funct3 = PHANTOM_FUNCT3,
        rd = Const "x0",
        rs1 = Const "x0",
        imm = Const PhantomImm::HintInput as u16
    );
}

/// Reset the hint stream with `len` random `u64`s.
///
/// `len` is passed as a full RV64 register value.
#[inline(always)]
pub fn hint_random(len: usize) {
    openvm_custom_insn::custom_insn_i!(
        opcode = SYSTEM_OPCODE,
        funct3 = PHANTOM_FUNCT3,
        rd = In len,
        rs1 = Const "x0",
        imm = Const PhantomImm::HintRandom as u16
    );
}

/// Store rs1 to [[rd] + imm]_3.
#[macro_export]
macro_rules! reveal {
    ($rd:ident, $rs1:ident, $imm:expr) => {
        openvm_custom_insn::custom_insn_i!(
            opcode = openvm_riscv_guest::SYSTEM_OPCODE,
            funct3 = openvm_riscv_guest::REVEAL_FUNCT3,
            rd = In $rd,
            rs1 = In $rs1,
            imm = Const $imm
        )
    };
}

/// Print UTF-8 string encoded as bytes to host stdout for debugging purposes.
#[inline(always)]
pub fn print_str_from_bytes(str_as_bytes: &[u8]) {
    raw_print_str_from_bytes(str_as_bytes.as_ptr(), str_as_bytes.len());
}

/// Both operands are passed as full RV64 register values.
#[inline(always)]
pub fn raw_print_str_from_bytes(msg_ptr: *const u8, len: usize) {
    openvm_custom_insn::custom_insn_i!(
        opcode = SYSTEM_OPCODE,
        funct3 = PHANTOM_FUNCT3,
        rd = In msg_ptr,
        rs1 = In len,
        imm = Const PhantomImm::PrintStr as u16
    );
}
