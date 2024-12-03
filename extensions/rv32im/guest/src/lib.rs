#![no_std]

/// Library functions for user input/output.
#[cfg(target_os = "zkvm")]
mod io;
#[cfg(target_os = "zkvm")]
pub use io::*;

pub const SYSTEM_OPCODE: u8 = 0x0b;
pub const CSR_OPCODE: u8 = 0b1110011;
pub const RV32_ALU_OPCODE: u8 = 0b0110011;
pub const RV32M_FUNCT7: u8 = 0x01;

pub const TERMINATE_FUNCT3: u8 = 0b000;
pub const HINT_STORE_W_FUNCT3: u8 = 0b001;
pub const REVEAL_FUNCT3: u8 = 0b010;
pub const PHANTOM_FUNCT3: u8 = 0b011;
pub const CSRRW_FUNCT3: u8 = 0b001;

/// imm options for system phantom instructions
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u16)]
pub enum PhantomImm {
    HintInput = 0,
    PrintStr,
}

impl PhantomImm {
    pub fn from_repr(repr: u16) -> Option<Self> {
        if repr == PhantomImm::HintInput as u16 {
            Some(PhantomImm::HintInput)
        } else if repr == PhantomImm::PrintStr as u16 {
            Some(PhantomImm::PrintStr)
        } else {
            None
        }
    }
}
