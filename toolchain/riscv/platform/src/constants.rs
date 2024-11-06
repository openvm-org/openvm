use strum_macros::FromRepr;

pub const CUSTOM_0: u8 = 0x0b;
pub const CUSTOM_1: u8 = 0x2b;

/// Different funct3 for custom RISC-V instructions using the [CUSTOM_0] 7-bit opcode prefix.
#[derive(Debug, Copy, Clone, PartialEq, Eq, FromRepr)]
#[repr(u8)]
pub enum Custom0Funct3 {
    Terminate = 0,
    HintStoreW,
    Reveal,
    Phantom,
    Keccak256 = 0b100,
    Int256 = 0b101,
    Beq256,
}

/// Different funct3 for custom RISC-V instructions using the [CUSTOM_1] 7-bit opcode prefix.
#[derive(Debug, Copy, Clone, PartialEq, Eq, FromRepr)]
#[repr(u8)]
pub enum Custom1Funct3 {
    ModularArithmetic = 0,
    ShortWeierstrass,
}

/// imm options for system phantom instructions
#[derive(Debug, Copy, Clone, PartialEq, Eq, FromRepr)]
#[repr(u16)]
pub enum PhantomImm {
    HintInput = 0,
    PrintStr,
}

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

pub const MODULAR_ARITHMETIC_MAX_KINDS: u8 = 8;

/// Modular arithmetic is configurable. The funct7 field equals `mod_idx * MODULAR_ARITHMETIC_MAX_KINDS + base_funct7`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, FromRepr)]
#[repr(u8)]
pub enum ModArithBaseFunct7 {
    AddMod = 0,
    SubMod,
    MulMod,
    DivMod,
    IsEqMod,
}

pub const SHORT_WEIERSTRASS_MAX_KINDS: u8 = 8;

/// Short Weierstrass curves are configurable. The funct7 field equals `curve_idx * SHORT_WEIERSTRASS_MAX_KINDS + base_funct7`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, FromRepr)]
#[repr(u8)]
pub enum SwBaseFunct7 {
    SwAddNe = 0,
    SwDouble,
}
