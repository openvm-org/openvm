use strum_macros::FromRepr;

// TODO: when moving the actual guest program to this crate, make everything use these constants.
pub const OPCODE: u8 = 0x2b;
pub const ECC_FUNCT3: u8 = 0b001;

/// Short Weierstrass curves are configurable.
/// The funct7 field equals `curve_idx * SHORT_WEIERSTRASS_MAX_KINDS + base_funct7`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, FromRepr)]
#[repr(u8)]
pub enum SwBaseFunct7 {
    SwAddNe = 0,
    SwDouble,
    SwSetup,
}

impl SwBaseFunct7 {
    pub const SHORT_WEIERSTRASS_MAX_KINDS: u8 = 8;
}
