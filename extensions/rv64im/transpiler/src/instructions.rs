// =================================================================================================
// RV64IM support opcodes.
// All offsets start at 0x800+ to avoid collisions with existing extensions.
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
#[opcode_offset = 0x800]
#[repr(usize)]
pub enum Rv64BaseAluOpcode {
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
#[opcode_offset = 0x805]
#[repr(usize)]
pub enum Rv64ShiftOpcode {
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
#[opcode_offset = 0x808]
#[repr(usize)]
pub enum Rv64LessThanOpcode {
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
#[opcode_offset = 0x80A]
#[repr(usize)]
pub enum Rv64BaseAluWOpcode {
    ADDW,
    SUBW,
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
#[opcode_offset = 0x80C]
#[repr(usize)]
pub enum Rv64ShiftWOpcode {
    SLLW,
    SRLW,
    SRAW,
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
#[opcode_offset = 0x810]
#[repr(usize)]
pub enum Rv64LoadStoreOpcode {
    // LoadStoreCore (no sign extension needed):
    LOADD,
    LOADWU,
    LOADBU,
    LOADHU,
    STORED,
    STOREW,
    STOREH,
    STOREB,
    // LoadSignExtendCore (sign extension required):
    LOADB,
    LOADH,
    LOADW,
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
#[opcode_offset = 0x820]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv64BranchEqualOpcode {
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
#[opcode_offset = 0x825]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv64BranchLessThanOpcode {
    BLT,
    BLTU,
    BGE,
    BGEU,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x830]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv64JalLuiOpcode {
    JAL,
    LUI,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x835]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv64JalrOpcode {
    JALR,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x840]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv64AuipcOpcode {
    AUIPC,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x850]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv64MulOpcode {
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
#[opcode_offset = 0x851]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv64MulHOpcode {
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
#[opcode_offset = 0x854]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv64DivRemOpcode {
    DIV,
    DIVU,
    REM,
    REMU,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x858]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv64MulWOpcode {
    MULW,
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
#[opcode_offset = 0x859]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv64DivRemWOpcode {
    DIVW,
    DIVUW,
    REMW,
    REMUW,
}

// =================================================================================================
// Rv64HintStore Instruction
// =================================================================================================

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x860]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv64HintStoreOpcode {
    HINT_STORED,
    HINT_BUFFER,
}

// =================================================================================================
// Phantom opcodes
// =================================================================================================

#[derive(Copy, Clone, Debug, PartialEq, Eq, FromRepr)]
#[repr(u16)]
pub enum Rv64Phantom {
    HintInput = 0x40,
    PrintStr = 0x41,
    HintRandom = 0x42,
    HintLoadByKey = 0x43,
}
