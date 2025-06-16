#![allow(unused_imports)]
use crate::{PhantomImm, PHANTOM_FUNCT3, SYSTEM_OPCODE};

/// Store the next 4 bytes from the hint stream to [[rd]_1]_2.
#[macro_export]
macro_rules! hint_store_u32 {
    ($x:expr) => {
        openvm_custom_insn::custom_insn_i!(
            opcode = openvm_rv32im_guest::SYSTEM_OPCODE,
            funct3 = openvm_rv32im_guest::HINT_FUNCT3,
            rd = In $x,
            rs1 = Const "x0",
            imm = Const 0,
        )
    };
}

/// Store the next 4*len bytes from the hint stream to [[rd]_1]_2.
#[macro_export]
macro_rules! hint_buffer_u32 {
    ($x:expr, $len:expr) => {
        if $len != 0 {
            openvm_custom_insn::custom_insn_i!(
                opcode = openvm_rv32im_guest::SYSTEM_OPCODE,
                funct3 = openvm_rv32im_guest::HINT_FUNCT3,
                rd = In $x,
                rs1 = In $len,
                imm = Const 1,
            )
        }
    };
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

/// Reset the hint stream with `len` random `u32`s
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

/// Hint the VM to load values with key = [ptr: len] into input streams.
#[inline(always)]
pub fn hint_load_by_key(ptr: *const u8, len: u32) {
    openvm_custom_insn::custom_insn_i!(
        opcode = SYSTEM_OPCODE,
        funct3 = PHANTOM_FUNCT3,
        rd = In ptr,
        rs1 = In len,
        imm = Const PhantomImm::HintLoadByKey as u16,
    );
}

/// Store rs1 to [[rd] + imm]_3.
#[macro_export]
macro_rules! reveal {
    ($rd:ident, $rs1:ident, $imm:expr) => {
        openvm_custom_insn::custom_insn_i!(
            opcode = openvm_rv32im_guest::SYSTEM_OPCODE,
            funct3 = openvm_rv32im_guest::REVEAL_FUNCT3,
            rd = In $rd,
            rs1 = In $rs1,
            imm = Const $imm
        )
    };
}

/// Store rs1 to [[rd]]_4.
#[macro_export]
macro_rules! store_to_native {
    ($rd:ident, $rs1:ident) => {
        openvm_custom_insn::custom_insn_r!(
            opcode = openvm_rv32im_guest::SYSTEM_OPCODE,
            funct3 = openvm_rv32im_guest::NATIVE_STOREW_FUNCT3,
            funct7 = openvm_rv32im_guest::NATIVE_STOREW_FUNCT7,
            rd = In $rd,
            rs1 = In $rs1,
            rs2 = In $rs1,
        )
    };
}

/// Print UTF-8 string encoded as bytes to host stdout for debugging purposes.
#[inline(always)]
pub fn print_str_from_bytes(str_as_bytes: &[u8]) {
    raw_print_str_from_bytes(str_as_bytes.as_ptr(), str_as_bytes.len());
}

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
