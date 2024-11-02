#[cfg(target_os = "zkvm")]
use axvm_platform::constants::{Custom0Funct3, CUSTOM_0};

/// Store the next 4 bytes from the hint stream to [[rd] + imm]_2.
#[cfg(target_os = "zkvm")]
#[macro_export]
macro_rules! hint_store_u32 {
    ($x:ident, $imm:expr) => {
        axvm_platform::custom_insn_i!(
            axvm_platform::constants::CUSTOM_0,
            axvm_platform::constants::Custom0Funct3::HintStoreW as u8,
            $x,
            "x0",
            $imm
        )
    };
}

/// Reset the hint stream with the next hint.
#[inline(always)]
pub fn hint_input() {
    #[cfg(target_os = "zkvm")]
    axvm_platform::custom_insn_i!(CUSTOM_0, Custom0Funct3::HintInput as u8, "x0", "x0", 0);
    #[cfg(not(target_os = "zkvm"))]
    todo!()
}

/// Store rs1 to [[rd] + imm]_2.
#[cfg(target_os = "zkvm")]
#[macro_export]
macro_rules! reveal {
    ($rd:ident, $rs1:ident, $imm:expr) => {
        axvm_platform::custom_insn_i!(
            axvm_platform::constants::CUSTOM_0,
            axvm_platform::constants::Custom0Funct3::Reveal as u8,
            $rd,
            $rs1,
            $imm
        )
    };
}
