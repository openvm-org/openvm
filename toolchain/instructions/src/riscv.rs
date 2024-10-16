use crate::{EccOpcode, ModularArithmeticOpcode};

/// The 7-bit opcode prefix for a 32-bit RISC-V instruction.
#[repr(u8)]
pub enum RvOpcodePrefix {
    Custom0 = 0b0001011,
    Custom1 = 0b0101011,
}

/// Trait to implement on opcode class enum to specify custom 32-bit RISC-V instruction definition.
pub trait RvIntrinsic {
    /// The 3-bit funct3 field to use in custom intrinsic 32-bit RISC-V instructions.
    fn funct3() -> u8;

    /// The base 7-bit funct7 field, before adding any offsets, in custom intrinsic 32-bit RISC-V instructions.
    fn base_funct7(&self) -> u8;
}

impl RvIntrinsic for ModularArithmeticOpcode {
    fn funct3() -> u8 {
        0b000
    }

    fn base_funct7(&self) -> u8 {
        match self {
            ModularArithmeticOpcode::ADD => 0x00,
            ModularArithmeticOpcode::SUB => 0x01,
            ModularArithmeticOpcode::MUL => 0x02,
            ModularArithmeticOpcode::DIV => 0x03,
        }
    }
}

impl RvIntrinsic for EccOpcode {
    fn funct3() -> u8 {
        0b001
    }

    fn base_funct7(&self) -> u8 {
        match self {
            EccOpcode::EC_ADD_NE => 0x00,
            EccOpcode::EC_DOUBLE => 0x01,
        }
    }
}
