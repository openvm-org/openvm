//! This crate is intended for use on host machine. This includes usage within procedural macros.

#![allow(non_camel_case_types)]

use axvm_instructions_derive::UsizeOpcode;
use strum::IntoEnumIterator;
use strum_macros::{EnumCount, EnumIter, FromRepr};

pub mod config;
mod curves;
pub mod exe;
pub mod instruction;
mod phantom;
pub mod program;
/// Module with traits and constants for RISC-V instruction definitions for custom axVM instructions.
pub mod riscv;
pub mod utils;

pub use curves::*;
pub use phantom::*;

pub trait UsizeOpcode {
    fn default_offset() -> usize;
    /// Convert from the discriminant of the enum to the typed enum variant.
    /// Default implementation uses `from_repr`.
    fn from_usize(value: usize) -> Self;
    fn as_usize(&self) -> usize;

    fn with_default_offset(&self) -> usize {
        self.as_usize() + Self::default_offset()
    }
}

// =================================================================================================
// System opcodes
// =================================================================================================

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0]
#[repr(usize)]
pub enum SystemOpcode {
    TERMINATE,
    PHANTOM,
}

// =================================================================================================
// Native kernel opcodes
// =================================================================================================

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x100]
#[repr(usize)]
pub enum NativeLoadStoreOpcode {
    LOADW,
    STOREW,
    LOADW2,
    STOREW2,
    /// Instruction to write the next hint word into memory.
    SHINTW,
}

#[derive(Copy, Clone, Debug, UsizeOpcode)]
#[opcode_offset = 0x110]
pub struct NativeBranchEqualOpcode(pub BranchEqualOpcode);

impl NativeBranchEqualOpcode {
    pub fn iter() -> impl Iterator<Item = Self> {
        BranchEqualOpcode::iter().map(Self)
    }
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x115]
#[repr(usize)]
pub enum NativeJalOpcode {
    JAL,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x120]
#[repr(usize)]
pub enum PublishOpcode {
    PUBLISH,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x125]
#[repr(usize)]
pub enum CastfOpcode {
    CASTF,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x130]
#[repr(usize)]
pub enum FieldArithmeticOpcode {
    ADD,
    SUB,
    MUL,
    DIV,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x140]
#[repr(usize)]
pub enum FieldExtensionOpcode {
    FE4ADD,
    FE4SUB,
    BBE4MUL,
    BBE4DIV,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x150]
#[repr(usize)]
pub enum Poseidon2Opcode {
    PERM_POS2,
    COMP_POS2,
}

/// Opcodes for FRI opening proofs.
#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x160]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum FriOpcode {
    /// In FRI pcs opening verification, the reduced opening polynomial is computed one evaluation
    /// per column polynomial, per opening point
    FRI_REDUCED_OPENING,
}

// =================================================================================================
// RV32IM support opcodes.
// Enum types that do not start with Rv32 can be used for generic big integers, but the default
// offset is reserved for RV32IM.
//
// Create a new wrapper struct U256BaseAluOpcode(pub BaseAluOpcode) with the UsizeOpcode macro to
// specify a different offset.
// =================================================================================================

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x200]
#[repr(usize)]
pub enum BaseAluOpcode {
    ADD,
    SUB,
    XOR,
    OR,
    AND,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x205]
#[repr(usize)]
pub enum ShiftOpcode {
    SLL,
    SRL,
    SRA,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x208]
#[repr(usize)]
pub enum LessThanOpcode {
    SLT,
    SLTU,
}

// TODO[jpw]: life would be easier if we split LOADB,LOADH into a separate enum
#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x210]
#[repr(usize)]
pub enum Rv32LoadStoreOpcode {
    LOADW,
    /// LOADBU, LOADHU are unsigned extend opcodes, implemented in the same chip with LOADW
    LOADBU,
    LOADHU,
    STOREW,
    STOREH,
    STOREB,
    /// The following are signed extend opcodes
    LOADB,
    LOADH,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x220]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum BranchEqualOpcode {
    BEQ,
    BNE,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x225]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum BranchLessThanOpcode {
    BLT,
    BLTU,
    BGE,
    BGEU,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x230]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv32JalLuiOpcode {
    JAL,
    LUI,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x235]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv32JalrOpcode {
    JALR,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x240]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv32AuipcOpcode {
    AUIPC,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x250]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum MulOpcode {
    MUL,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x251]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum MulHOpcode {
    MULH,
    MULHSU,
    MULHU,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x254]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum DivRemOpcode {
    DIV,
    DIVU,
    REM,
    REMU,
}

// =================================================================================================
// Intrinsics opcodes
// =================================================================================================

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x300]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv32HintStoreOpcode {
    HINT_STOREW,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x310]
#[repr(usize)]
pub enum Rv32KeccakOpcode {
    KECCAK256,
}

// =================================================================================================
// Intrinsics: 256-bit Integers
// =================================================================================================

#[derive(Copy, Clone, Debug, UsizeOpcode)]
#[opcode_offset = 0x400]
pub struct Rv32BaseAlu256Opcode(pub BaseAluOpcode);

impl Rv32BaseAlu256Opcode {
    pub fn iter() -> impl Iterator<Item = Self> {
        BaseAluOpcode::iter().map(Self)
    }
}

#[derive(Copy, Clone, Debug, UsizeOpcode)]
#[opcode_offset = 0x405]
pub struct Rv32Shift256Opcode(pub ShiftOpcode);

impl Rv32Shift256Opcode {
    pub fn iter() -> impl Iterator<Item = Self> {
        ShiftOpcode::iter().map(Self)
    }
}

#[derive(Copy, Clone, Debug, UsizeOpcode)]
#[opcode_offset = 0x408]
pub struct Rv32LessThan256Opcode(pub LessThanOpcode);

impl Rv32LessThan256Opcode {
    pub fn iter() -> impl Iterator<Item = Self> {
        LessThanOpcode::iter().map(Self)
    }
}

#[derive(Copy, Clone, Debug, UsizeOpcode)]
#[opcode_offset = 0x420]
pub struct Rv32BranchEqual256Opcode(pub BranchEqualOpcode);

impl Rv32BranchEqual256Opcode {
    pub fn iter() -> impl Iterator<Item = Self> {
        BranchEqualOpcode::iter().map(Self)
    }
}

#[derive(Copy, Clone, Debug, UsizeOpcode)]
#[opcode_offset = 0x425]
pub struct Rv32BranchLessThan256Opcode(pub BranchLessThanOpcode);

impl Rv32BranchLessThan256Opcode {
    pub fn iter() -> impl Iterator<Item = Self> {
        BranchLessThanOpcode::iter().map(Self)
    }
}

#[derive(Copy, Clone, Debug, UsizeOpcode)]
#[opcode_offset = 0x450]
pub struct Rv32Mul256Opcode(pub MulOpcode);

impl Rv32Mul256Opcode {
    pub fn iter() -> impl Iterator<Item = Self> {
        MulOpcode::iter().map(Self)
    }
}

// =================================================================================================
// Intrinsics: Prime Field Arithmetic
// =================================================================================================
#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x500]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv32ModularArithmeticOpcode {
    ADD,
    SUB,
    SETUP_ADDSUB,
    MUL,
    DIV,
    SETUP_MULDIV,
    IS_EQ,
    SETUP_ISEQ,
}

// to be deleted and replaced by Rv32SwOpcode
#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x600]
#[repr(usize)]
pub enum Rv32WeierstrassOpcode {
    EC_ADD_NE,
    EC_DOUBLE,
    SETUP_EC_ADD_NE,
    SETUP_EC_DOUBLE,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x700]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Fp12Opcode {
    ADD,
    SUB,
    MUL,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x710]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Fp2Opcode {
    ADD,
    SUB,
    SETUP_ADDSUB,
    MUL,
    DIV,
    SETUP_MULDIV,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x750]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum PairingOpcode {
    MILLER_DOUBLE_STEP,
    MILLER_DOUBLE_AND_ADD_STEP,
    EVALUATE_LINE,
    MUL_013_BY_013,
    MUL_BY_013,
    MUL_BY_01234,
    MUL_023_BY_023,
    MUL_BY_023,
    MUL_BY_02345,
}

// =================================================================================================
// For internal dev use only
// =================================================================================================

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0xdeadaf]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum UnimplementedOpcode {
    REPLACE_ME,
}
