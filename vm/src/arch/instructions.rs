use std::fmt;

use afs_derive::UsizeOpcode;
use enum_utils::FromStr;
use strum_macros::{EnumCount, EnumIter, FromRepr};

pub trait UsizeOpcode {
    // fn to_usize(&self) -> usize;
    fn from_usize(value: usize) -> Self;
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    FromStr,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    UsizeOpcode,
)]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum CoreOpcode {
    NOP,
    LOADW,
    STOREW,
    LOADW2,
    STOREW2,
    JAL,
    BEQ,
    BNE,
    TERMINATE,
    PUBLISH,
    FAIL,
    PRINTF,

    // TODO: move these to a separate class, PhantomOpcode or something
    /// Instruction to write the next hint word into memory.
    SHINTW,

    /// Phantom instruction to prepare the next input vector for hinting.
    HINT_INPUT,
    /// Phantom instruction to prepare the little-endian bit decomposition of a variable for hinting.
    HINT_BITS,
    /// Phantom instruction to prepare the little-endian byte decomposition of a variable for hinting.
    HINT_BYTES,

    /// Phantom instruction to start tracing
    CT_START,
    /// Phantom instruction to end tracing
    CT_END,
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    FromStr,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    UsizeOpcode,
)]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum FieldArithmeticOpcode {
    FADD,
    FSUB,
    FMUL,
    FDIV,
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    FromStr,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    UsizeOpcode,
)]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum FieldExtensionOpcode {
    FE4ADD,
    FE4SUB,
    BBE4MUL,
    BBE4DIV,
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    FromStr,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    UsizeOpcode,
)]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum CastfOpcode {
    CASTF,
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    FromStr,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    UsizeOpcode,
)]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Poseidon2Opcode {
    PERM_POS2,
    COMP_POS2,
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    FromStr,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    UsizeOpcode,
)]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Keccak256Opcode {
    KECCAK256,
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    FromStr,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    UsizeOpcode,
)]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum ModularArithmeticOpcode {
    COORD_ADD,
    COORD_SUB,
    SCALAR_ADD,
    SCALAR_SUB,

    COORD_MUL,
    COORD_DIV,
    SCALAR_MUL,
    SCALAR_DIV,
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    FromStr,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    UsizeOpcode,
)]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum EccOpcode {
    EC_ADD_NE,
    EC_DOUBLE,
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    FromStr,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    UsizeOpcode,
)]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum U256Opcode {
    // maybe later we will make it uint and specify the parameters in the config
    ADD,
    SUB,
    MUL,
    LT,
    EQ,
    XOR,
    AND,
    OR,
    SLT,
    SLL,
    SRL,
    SRA,
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    FromStr,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    UsizeOpcode,
)]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum U32Opcode {
    LUI,
    AUIPC,
}
