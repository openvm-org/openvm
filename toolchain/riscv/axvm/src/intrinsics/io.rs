// use crate::custom_insn_i;

use axvm_platform::{custom_insn_i, intrinsics::CUSTOM_0};

/// Store the next 4 bytes from the hint stream to [[rd] + imm]_2.
#[macro_export]
macro_rules! hint_store_u32 {
    ($rd:literal, $imm:expr) => {
        unsafe { custom_insn_i!(CUSTOM_0, 0b001, $rd, "x0", $imm) }
    };
}

/// Read the next byte from the hint stream.
pub fn read_byte() -> u8 {
    let mut x: u8;
    custom_insn_i!(CUSTOM_0, 0b001, x, "x0", 0);
    x
}
