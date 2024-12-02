use axvm_platform::constants::{Custom0Funct3, PhantomImm, CUSTOM_0};

/// Store the next 4 bytes from the hint stream to [[rd] + imm]_2.
#[macro_export]
macro_rules! hint_store_u32 {
    ($x:expr, $imm:expr) => {
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
    axvm_platform::custom_insn_i!(
        CUSTOM_0,
        Custom0Funct3::Phantom as u8,
        "x0",
        "x0",
        PhantomImm::HintInput as u16
    );
}

/// Store rs1 to [[rd] + imm]_2.
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

/// Print UTF-8 string encoded as bytes to host stdout for debugging purposes.
#[inline(always)]
pub fn print_str_from_bytes(str_as_bytes: &[u8]) {
    raw_print_str_from_bytes(str_as_bytes.as_ptr(), str_as_bytes.len());
}

#[inline(always)]
pub fn raw_print_str_from_bytes(msg_ptr: *const u8, len: usize) {
    axvm_platform::custom_insn_i!(
        CUSTOM_0,
        Custom0Funct3::Phantom as u8,
        msg_ptr,
        len,
        PhantomImm::PrintStr as u16
    );
}
