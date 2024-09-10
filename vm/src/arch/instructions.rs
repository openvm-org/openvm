use std::fmt;

use enum_utils::FromStr;
use Opcode::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq, FromStr, PartialOrd, Ord)]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Opcode {
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
    BBE4DIV = 33,

    PERM_POS2 = 40,
    COMP_POS2 = 41,
    KECCAK256 = 42,

    /// Instruction to write the next hint word into memory.
    SHINTW = 50,

    /// Phantom instruction to prepare the next input vector for hinting.
    HINT_INPUT = 51,
    /// Phantom instruction to prepare the little-endian bit decomposition of a variable for hinting.
    HINT_BITS = 52,
    /// Phantom instruction to prepare the little-endian byte decomposition of a variable for hinting.
    HINT_BYTES = 53,

    /// Phantom instruction to start tracing
    CT_START = 60,
    /// Phantom instruction to end tracing
    CT_END = 61,

    SECP256K1_COORD_ADD = 70,
    SECP256K1_COORD_SUB = 71,
    SECP256K1_COORD_MUL = 72,
    SECP256K1_COORD_DIV = 73,

    SECP256K1_SCALAR_ADD = 74,
    SECP256K1_SCALAR_SUB = 75,
    SECP256K1_SCALAR_MUL = 76,
    SECP256K1_SCALAR_DIV = 77,

    ADD256 = 80,
    SUB256 = 81,
    // save 82 for MUL
    LT256 = 83,
    EQ256 = 84,

    NOP = 100,
}

impl fmt::Display for Opcode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub const CORE_INSTRUCTIONS: [Opcode; 16] = [
    LOADW, STOREW, JAL, BEQ, BNE, TERMINATE, SHINTW, HINT_INPUT, HINT_BITS, HINT_BYTES, PUBLISH,
    CT_START, CT_END, NOP, LOADW2, STOREW2,
];
pub const FIELD_ARITHMETIC_INSTRUCTIONS: [Opcode; 4] = [FADD, FSUB, FMUL, FDIV];
pub const FIELD_EXTENSION_INSTRUCTIONS: [Opcode; 4] = [FE4ADD, FE4SUB, BBE4MUL, BBE4DIV];
pub const UINT256_ARITHMETIC_INSTRUCTIONS: [Opcode; 4] = [ADD256, SUB256, LT256, EQ256];
pub const SECP256K1_COORD_MODULAR_ARITHMETIC_INSTRUCTIONS: [Opcode; 4] = [
    SECP256K1_COORD_ADD,
    SECP256K1_COORD_SUB,
    SECP256K1_COORD_MUL,
    SECP256K1_COORD_DIV,
];

pub const SECP256K1_SCALAR_MODULAR_ARITHMETIC_INSTRUCTIONS: [Opcode; 4] = [
    SECP256K1_SCALAR_ADD,
    SECP256K1_SCALAR_SUB,
    SECP256K1_SCALAR_MUL,
    SECP256K1_SCALAR_DIV,
];

impl Opcode {
    pub fn all_opcodes() -> Vec<Opcode> {
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
