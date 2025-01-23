#![no_std]

/// Library functions for user input/output.
#[cfg(target_os = "zkvm")]
mod io;
#[cfg(target_os = "zkvm")]
pub use io::*;
use strum_macros::FromRepr;

/// This is custom-0 defined in RISC-V spec document
pub const SYSTEM_OPCODE: u8 = 0x0b;
pub const CSR_OPCODE: u8 = 0b1110011;
pub const RV32_ALU_OPCODE: u8 = 0b0110011;
pub const RV32M_FUNCT7: u8 = 0x01;

pub const TERMINATE_FUNCT3: u8 = 0b000;
pub const HINT_FUNCT3: u8 = 0b001;
pub const HINT_STOREW_IMM: u32 = 0;
pub const HINT_BUFFER_IMM: u32 = 1;
pub const REVEAL_FUNCT3: u8 = 0b010;
pub const PHANTOM_FUNCT3: u8 = 0b011;
pub const CSRRW_FUNCT3: u8 = 0b001;

/// imm options for system phantom instructions
#[derive(Debug, Copy, Clone, PartialEq, Eq, FromRepr)]
#[repr(u16)]
pub enum PhantomImm {
    HintInput = 0,
    PrintStr,
}
