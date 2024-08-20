use std::fmt;

use enum_utils::FromStr;

use OpCode::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq, FromStr, PartialOrd, Ord)]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum OpCode {
    LOADW = 0,
    STOREW = 1,
    LOADW2 = 2,
    STOREW2 = 3,
    JAL = 4,
    BEQ = 5,
    BNE = 6,
    TERMINATE = 7,
    PUBLISH = 8,

    FADD = 10,
    FSUB = 11,
    FMUL = 12,
    FDIV = 13,

    F_LESS_THAN = 14,

    FAIL = 20,
    PRINTF = 21,

    FE4ADD = 30,
    FE4SUB = 31,
    BBE4MUL = 32,
    BBE4INV = 33,

    PERM_POS2 = 40,
    COMP_POS2 = 41,

    /// Instruction to write the next hint word into memory.
    SHINTW = 50,

    /// Phantom instruction to prepare the next input vector for hinting.
    HINT_INPUT = 51,
    /// Phantom instruction to prepare the little-endian bit decomposition of a variable for hinting.
    HINT_BITS = 52,

    /// Phantom instruction to start tracing
    CT_START = 60,
    /// Phantom instruction to end tracing
    CT_END = 61,

    MOD_SECP256K1_ADD = 70,
    MOD_SECP256K1_SUB = 71,
    MOD_SECP256K1_MUL = 72,
    MOD_SECP256K1_DIV = 73,

    ADD256 = 80,
    SUB256 = 81,
    // save 82 for MUL
    LT256 = 83,
    EQ256 = 84,

    NOP = 100,
}

impl fmt::Display for OpCode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub const CORE_INSTRUCTIONS: [OpCode; 15] = [
    LOADW, STOREW, JAL, BEQ, BNE, TERMINATE, SHINTW, HINT_INPUT, HINT_BITS, PUBLISH, CT_START,
    CT_END, NOP, LOADW2, STOREW2,
];
pub const FIELD_ARITHMETIC_INSTRUCTIONS: [OpCode; 4] = [FADD, FSUB, FMUL, FDIV];
pub const FIELD_EXTENSION_INSTRUCTIONS: [OpCode; 4] = [FE4ADD, FE4SUB, BBE4MUL, BBE4INV];
pub const MODULAR_ARITHMETIC_INSTRUCTIONS: [OpCode; 4] = [
    MOD_SECP256K1_ADD,
    MOD_SECP256K1_SUB,
    MOD_SECP256K1_MUL,
    MOD_SECP256K1_DIV,
];

impl OpCode {
    pub fn all_opcodes() -> Vec<OpCode> {
        let mut all_opcodes = vec![];
        all_opcodes.extend(CORE_INSTRUCTIONS);
        all_opcodes.extend(FIELD_ARITHMETIC_INSTRUCTIONS);
        all_opcodes.extend(FIELD_EXTENSION_INSTRUCTIONS);
        all_opcodes.extend([FAIL, PRINTF]);
        all_opcodes.extend([PERM_POS2, COMP_POS2]);
        all_opcodes
    }

    pub fn from_u8(value: u8) -> Option<Self> {
        Self::all_opcodes()
            .into_iter()
            .find(|&opcode| value == opcode as u8)
    }
}
