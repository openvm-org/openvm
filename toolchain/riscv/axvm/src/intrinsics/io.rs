#[cfg(target_os = "zkvm")]
use axvm_platform::constants::CUSTOM_0;

/// Store the next 4 bytes from the hint stream to [[rd] + imm]_2.
#[cfg(target_os = "zkvm")]
#[macro_export]
macro_rules! hint_store_u32 {
    ($x:ident, $imm:expr) => {
        axvm_platform::custom_insn_i!(axvm_platform::intrinsics::CUSTOM_0, 0b001, $x, "x0", $imm)
    };
}

/// Reset the hint stream with the next hint.
#[inline(always)]
pub fn hint_input() {
    #[cfg(target_os = "zkvm")]
    axvm_platform::custom_insn_i!(CUSTOM_0, 0b011, "x0", "x0", 0);
    #[cfg(not(target_os = "zkvm"))]
    todo!()
}
