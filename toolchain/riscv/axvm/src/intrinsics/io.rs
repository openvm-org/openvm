// use crate::custom_insn_i;

/// Store the next 4 bytes from the hint stream to [[rd] + imm]_2.
#[macro_export]
macro_rules! hint_store_u32 {
    ($rd:literal, $imm:expr) => {
        unsafe { custom_insn_i!(CUSTOM_0, 0b001, $rd, "x0", $imm) }
    };
}
