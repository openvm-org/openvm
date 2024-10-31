use crate::{EccOpcode, Rv32ModularArithmeticOpcode};

/// 32-bit register stored as 4 bytes (4 limbs of 8-bits) in axVM memory.
pub const RV32_REGISTER_NUM_LIMBS: usize = 4;
pub const RV32_CELL_BITS: usize = 8;

/// The 7-bit opcode prefix for a 32-bit RISC-V instruction.
#[repr(u8)]
pub enum RvOpcodePrefix {
    Custom0 = 0b0001011,
    Custom1 = 0b0101011,
}

/// Trait to implement on opcode class enum to specify custom 32-bit RISC-V instruction definition.
pub trait RvIntrinsic {
    /// The 3-bit funct3 field to use in custom intrinsic 32-bit RISC-V instructions.
    const FUNCT3: u8;
}

impl RvIntrinsic for Rv32ModularArithmeticOpcode {
    const FUNCT3: u8 = 0b000;
}

impl RvIntrinsic for EccOpcode {
    const FUNCT3: u8 = 0b001;
}
