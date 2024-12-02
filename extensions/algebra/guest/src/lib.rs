use strum_macros::FromRepr;

pub const OPCODE: u8 = 0x2b;
pub const MODULAR_ARITHMETIC_FUNCT3: u8 = 0b000;
pub const COMPLEX_EXT_FIELD_FUNCT3: u8 = 0b010;

// TODO: when moving the actual guest program to this crate, make everything use these constants.
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

impl ModArithBaseFunct7 {
    pub const MODULAR_ARITHMETIC_MAX_KINDS: u8 = 8;
}

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

impl ComplexExtFieldBaseFunct7 {
    pub const COMPLEX_EXT_FIELD_MAX_KINDS: u8 = 8;
}
