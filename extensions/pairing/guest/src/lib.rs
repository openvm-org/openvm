use strum_macros::FromRepr;


pub const OPCODE: u8 = 0x2b;
pub const PAIRING_FUNCT3: u8 = 0b011;

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

impl PairingBaseFunct7 {
    pub const PAIRING_MAX_KINDS: u8 = 16;
}
