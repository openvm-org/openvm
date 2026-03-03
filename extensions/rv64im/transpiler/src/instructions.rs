// =================================================================================================
// RV32IM support opcodes.
// Enum types that do not start with Rv32 can be used for generic big integers, but the default
// offset is reserved for RV32IM.
//
// Create a new wrapper struct U256BaseAluOpcode(pub BaseAluOpcode) with the LocalOpcode macro to
// specify a different offset.
// =================================================================================================

use openvm_instructions::LocalOpcode;
use openvm_instructions_derive::LocalOpcode;
use serde::{Deserialize, Serialize};
use strum::{EnumCount, EnumIter, FromRepr};

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    LocalOpcode,
    Serialize,
    Deserialize,
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
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    LocalOpcode,
    Serialize,
    Deserialize,
)]
#[opcode_offset = 0x205]
#[repr(usize)]
pub enum ShiftOpcode {
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
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    LocalOpcode,
    Serialize,
    Deserialize,
)]
#[opcode_offset = 0x208]
#[repr(usize)]
pub enum LessThanOpcode {
    SLT,
    SLTU,
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    LocalOpcode,
    Serialize,
    Deserialize,
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
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    LocalOpcode,
    Serialize,
    Deserialize,
)]
#[opcode_offset = 0x220]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum BranchEqualOpcode {
    BEQ,
    BNE,
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    LocalOpcode,
    Serialize,
    Deserialize,
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
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x230]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv32JalLuiOpcode {
    JAL,
    LUI,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x235]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv32JalrOpcode {
    JALR,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x240]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv32AuipcOpcode {
    AUIPC,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x250]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum MulOpcode {
    MUL,
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    LocalOpcode,
    Serialize,
    Deserialize,
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
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    LocalOpcode,
    Serialize,
    Deserialize,
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
// Rv32HintStore Instruction
// =================================================================================================

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x260]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv32HintStoreOpcode {
    HINT_STOREW,
    HINT_BUFFER,
}

// =================================================================================================
// Phantom opcodes
// =================================================================================================

#[derive(Copy, Clone, Debug, PartialEq, Eq, FromRepr)]
#[repr(u16)]
pub enum Rv32Phantom {
    /// Prepare the next input vector for hinting, but prepend it with a 4-byte decomposition of
    /// its length instead of one field element.
    HintInput = 0x20,
    /// Peek string from memory and print it to stdout.
    PrintStr,
    /// Prepare given amount of random numbers for hinting.
    HintRandom,
    /// Hint the VM to load values from the stream KV store into input streams.
    HintLoadByKey,
}
