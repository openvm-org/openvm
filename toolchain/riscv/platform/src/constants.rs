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
    /// Modular arithmetic
    ModularArithmetic = 0,
    /// Short Weierstrass elliptic curve arithmetic
    ShortWeierstrass,
    /// Arithmetic for quadratic extension field of a prime field, with irreducible polynomial `X^2 + 1`.
    ComplexExtField,
    /// Instructions for optimal Ate pairing
    Pairing,
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

/// Modular arithmetic is configurable.
/// The funct7 field equals `mod_idx * MODULAR_ARITHMETIC_MAX_KINDS + base_funct7`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, FromRepr)]
#[repr(u8)]
pub enum ModArithBaseFunct7 {
    AddMod = 0,
    SubMod,
    MulMod,
    DivMod,
    IsEqMod,
    SetupMod,
}

pub const SHORT_WEIERSTRASS_MAX_KINDS: u8 = 8;

/// Short Weierstrass curves are configurable.
/// The funct7 field equals `curve_idx * SHORT_WEIERSTRASS_MAX_KINDS + base_funct7`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, FromRepr)]
#[repr(u8)]
pub enum SwBaseFunct7 {
    SwAddNe = 0,
    SwDouble,
    SwSetup,
}

pub const COMPLEX_EXT_FIELD_MAX_KINDS: u8 = 8;

/// Complex extension field is configurable.
/// The funct7 field equals `fp2_idx * COMPLEX_EXT_FIELD_MAX_KINDS + base_funct7`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, FromRepr)]
#[repr(u8)]
pub enum ComplexExtFieldBaseFunct7 {
    Add = 0,
    Sub,
    Mul,
    Div,
    Setup,
}

pub const PAIRING_MAX_KINDS: u8 = 16;

/// The funct7 field equals `pairing_idx * PAIRING_MAX_KINDS + base_funct7`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, FromRepr)]
#[repr(u8)]
pub enum PairingBaseFunct7 {
    MillerDoubleStep = 0,
    MillerDoubleAndAddStep,
    Fp12Mul,
    EvaluateLine,
    Mul013By013,
    MulBy01234,
    Mul023By023,
    MulBy02345,
    HintFinalExp,
}
