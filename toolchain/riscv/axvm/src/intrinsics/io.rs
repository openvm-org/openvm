// use crate::custom_insn_i;

use axvm_platform::{custom_insn_i, intrinsics::CUSTOM_0};

/// Store the next 4 bytes from the hint stream to [[rd] + imm]_2.
#[macro_export]
macro_rules! hint_store_u32 {
    ($x:ident, $imm:expr) => {
        axvm_platform::custom_insn_i!(axvm_platform::intrinsics::CUSTOM_0, 0b001, $x, "x0", $imm)
    };
}

/// Reset the hint stream with the next hint.
#[inline(always)]
pub fn hint_input() {
    custom_insn_i!(CUSTOM_0, 0b011, "x0", "x0", 0);
}
