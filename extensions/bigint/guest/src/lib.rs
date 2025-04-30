#![no_std]

use strum_macros::FromRepr;

/// This is custom-0 defined in RISC-V spec document
pub const OPCODE: u8 = 0x0b;
pub const INT256_FUNCT3: u8 = 0b101;
pub const BEQ256_FUNCT3: u8 = 0b110;

/// funct7 options for 256-bit integer instructions.
#[derive(Debug, Copy, Clone, PartialEq, Eq, FromRepr)]
#[repr(u8)]
pub enum Int256Funct7 {
    Add = 0,
    Sub,
    Xor,
    Or,
    And,
    Sll,
    Srl,
    Sra,
    Slt,
    Sltu,
    Mul,
}

#[cfg(all(feature = "export-intrinsics", target_os = "zkvm"))]
pub mod externs;
