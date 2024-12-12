#![allow(unused_imports)]
use crate::{PhantomImm, PHANTOM_FUNCT3, SYSTEM_OPCODE};

/// Store the next 4 bytes from the hint stream to [[rd] + imm]_2.
#[macro_export]
macro_rules! hint_store_u32 {
    ($x:expr, $imm:expr) => {
        openvm_platform::custom_insn_i!(
            openvm_rv32im_guest::SYSTEM_OPCODE,
            openvm_rv32im_guest::HINT_STORE_W_FUNCT3,
            $x,
            "x0",
            $imm
        )
    };
}

/// Reset the hint stream with the next hint.
#[inline(always)]
pub fn hint_input() {
    openvm_platform::custom_insn_i!(
        SYSTEM_OPCODE,
        PHANTOM_FUNCT3,
        "x0",
        "x0",
        PhantomImm::HintInput as u16
    );
}

/// Store rs1 to [[rd] + imm]_2.
#[macro_export]
macro_rules! reveal {
    ($rd:ident, $rs1:ident, $imm:expr) => {
        openvm_platform::custom_insn_i!(
            openvm_rv32im_guest::SYSTEM_OPCODE,
            openvm_rv32im_guest::REVEAL_FUNCT3,
            $rd,
            $rs1,
            $imm
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
    openvm_platform::custom_insn_i!(
        SYSTEM_OPCODE,
        PHANTOM_FUNCT3,
        msg_ptr,
        len,
        PhantomImm::PrintStr as u16
    );
}
