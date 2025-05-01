#![no_std]
extern crate alloc;

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
pub const NATIVE_STOREW_FUNCT3: u8 = 0b111;
pub const NATIVE_STOREW_FUNCT7: u32 = 2;

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
    HintRandom,
    HintLoadByKey,
}

/// Encode a 2d-array of field elements into bytes for `hint_load_by_key`
#[cfg(not(target_os = "zkvm"))]
pub fn hint_load_by_key_encode<F: p3_field::PrimeField32>(
    value: &[alloc::vec::Vec<F>],
) -> alloc::vec::Vec<u8> {
    let len = value.len();
    let mut ret = (len as u32).to_le_bytes().to_vec();
    for v in value {
        ret.extend((v.len() as u32).to_le_bytes());
        ret.extend(v.iter().flat_map(|x| x.as_canonical_u32().to_le_bytes()));
    }
    ret
}
