// =================================================================================================
// RV64IM support opcodes.
// Enum types that do not start with Rv64 can be used for generic big integers, but the default
// offset is reserved for RV64IM.
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
pub enum Rv64LoadStoreOpcode {
    // Ordering matters: local opcode values are recorded in traces and mirrored by CUDA kernels.
    LOADD,
    LOADBU,
    LOADHU,
    LOADWU,
    STORED,
    STOREW,
    STOREH,
    STOREB,
    // Sign-extend loads. LOADW sign-extends 32→64 to fill the full register.
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
pub enum Rv64JalLuiOpcode {
    JAL,
    LUI,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x235]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv64JalrOpcode {
    JALR,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x240]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv64AuipcOpcode {
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
// Rv64HintStore Instruction
// =================================================================================================

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x260]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv64HintStoreOpcode {
    HINT_STORED,
    HINT_BUFFER,
}

// =================================================================================================
// RV64-specific W-suffix opcodes (32-bit operations on 64-bit registers)
// =================================================================================================

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
#[opcode_offset = 0x270]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum BaseAluWOpcode {
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
#[opcode_offset = 0x288]
#[repr(usize)]
pub enum BaseAluWImmOpcode {
    ADDIW,
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
#[opcode_offset = 0x275]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum ShiftWOpcode {
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
#[opcode_offset = 0x289]
#[repr(usize)]
pub enum ShiftWImmOpcode {
    SLLIW,
    SRLIW,
    SRAIW,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x280]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum MulWOpcode {
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
#[opcode_offset = 0x284]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum DivRemWOpcode {
    DIVW,
    DIVUW,
    REMW,
    REMUW,
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
#[opcode_offset = 0x290]
#[repr(usize)]
pub enum BaseAluImmOpcode {
    ADDI,
    XORI,
    ORI,
    ANDI,
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
#[opcode_offset = 0x294]
#[repr(usize)]
pub enum ShiftImmOpcode {
    SLLI,
    SRLI,
    SRAI,
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
#[opcode_offset = 0x297]
#[repr(usize)]
pub enum LessThanImmOpcode {
    SLTI,
    SLTIU,
}

// =================================================================================================
// Bit-manipulation opcodes (Zba / Zbb / Zbs), used by the RV64B extension.
// Classes occupy 0x2A0..0x2C8; 0x299..0x30F is otherwise unallocated.
// =================================================================================================

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
#[opcode_offset = 0x2a0]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum ShAddOpcode {
    SH1ADD,
    SH2ADD,
    SH3ADD,
    ADD_UW,
    SH1ADD_UW,
    SH2ADD_UW,
    SH3ADD_UW,
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
#[opcode_offset = 0x2a7]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum SlliUwOpcode {
    SLLI_UW,
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
#[opcode_offset = 0x2a8]
#[repr(usize)]
pub enum BitwiseInvOpcode {
    ANDN,
    ORN,
    XNOR,
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
#[opcode_offset = 0x2ab]
#[repr(usize)]
pub enum RotateOpcode {
    ROL,
    ROR,
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
#[opcode_offset = 0x2ad]
#[repr(usize)]
pub enum RotateImmOpcode {
    RORI,
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
#[opcode_offset = 0x2ae]
#[repr(usize)]
pub enum RotateWOpcode {
    ROLW,
    RORW,
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
#[opcode_offset = 0x2b0]
#[repr(usize)]
pub enum RotateWImmOpcode {
    RORIW,
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
#[opcode_offset = 0x2b1]
#[repr(usize)]
pub enum CountZerosOpcode {
    CLZ,
    CTZ,
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
#[opcode_offset = 0x2b3]
#[repr(usize)]
pub enum CountZerosWOpcode {
    CLZW,
    CTZW,
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
#[opcode_offset = 0x2b5]
#[repr(usize)]
pub enum CpopOpcode {
    CPOP,
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
#[opcode_offset = 0x2b6]
#[repr(usize)]
pub enum CpopWOpcode {
    CPOPW,
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
#[opcode_offset = 0x2b7]
#[repr(usize)]
pub enum MinMaxOpcode {
    MIN,
    MINU,
    MAX,
    MAXU,
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
#[opcode_offset = 0x2bb]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum ByteUnaryOpcode {
    SEXT_B,
    SEXT_H,
    ZEXT_H,
    ORC_B,
    REV8,
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
#[opcode_offset = 0x2c0]
#[repr(usize)]
pub enum SingleBitOpcode {
    BCLR,
    BSET,
    BINV,
    BEXT,
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
#[opcode_offset = 0x2c4]
#[repr(usize)]
pub enum SingleBitImmOpcode {
    BCLRI,
    BSETI,
    BINVI,
    BEXTI,
}

// =================================================================================================
// Phantom opcodes
// =================================================================================================

#[derive(Copy, Clone, Debug, PartialEq, Eq, FromRepr)]
#[repr(u16)]
pub enum Rv64Phantom {
    /// Prepare the next input vector for hinting, but prepend it with an 8-byte decomposition of
    /// its length instead of one field element.
    HintInput = 0x20,
    /// Peek string from memory and print it to stdout.
    PrintStr,
    /// Prepare given amount of random numbers for hinting.
    HintRandom,
}

#[cfg(test)]
mod tests {
    use strum::EnumCount;

    use super::*;

    /// Every opcode class in this file, as (name, class offset, opcode count).
    fn all_classes() -> Vec<(&'static str, usize, usize)> {
        vec![
            ("BaseAlu", BaseAluOpcode::CLASS_OFFSET, BaseAluOpcode::COUNT),
            ("Shift", ShiftOpcode::CLASS_OFFSET, ShiftOpcode::COUNT),
            (
                "LessThan",
                LessThanOpcode::CLASS_OFFSET,
                LessThanOpcode::COUNT,
            ),
            (
                "Rv64LoadStore",
                Rv64LoadStoreOpcode::CLASS_OFFSET,
                Rv64LoadStoreOpcode::COUNT,
            ),
            (
                "BranchEqual",
                BranchEqualOpcode::CLASS_OFFSET,
                BranchEqualOpcode::COUNT,
            ),
            (
                "BranchLessThan",
                BranchLessThanOpcode::CLASS_OFFSET,
                BranchLessThanOpcode::COUNT,
            ),
            (
                "Rv64JalLui",
                Rv64JalLuiOpcode::CLASS_OFFSET,
                Rv64JalLuiOpcode::COUNT,
            ),
            (
                "Rv64Jalr",
                Rv64JalrOpcode::CLASS_OFFSET,
                Rv64JalrOpcode::COUNT,
            ),
            (
                "Rv64Auipc",
                Rv64AuipcOpcode::CLASS_OFFSET,
                Rv64AuipcOpcode::COUNT,
            ),
            ("Mul", MulOpcode::CLASS_OFFSET, MulOpcode::COUNT),
            ("MulH", MulHOpcode::CLASS_OFFSET, MulHOpcode::COUNT),
            ("DivRem", DivRemOpcode::CLASS_OFFSET, DivRemOpcode::COUNT),
            (
                "Rv64HintStore",
                Rv64HintStoreOpcode::CLASS_OFFSET,
                Rv64HintStoreOpcode::COUNT,
            ),
            (
                "BaseAluW",
                BaseAluWOpcode::CLASS_OFFSET,
                BaseAluWOpcode::COUNT,
            ),
            (
                "BaseAluWImm",
                BaseAluWImmOpcode::CLASS_OFFSET,
                BaseAluWImmOpcode::COUNT,
            ),
            ("ShiftW", ShiftWOpcode::CLASS_OFFSET, ShiftWOpcode::COUNT),
            (
                "ShiftWImm",
                ShiftWImmOpcode::CLASS_OFFSET,
                ShiftWImmOpcode::COUNT,
            ),
            ("MulW", MulWOpcode::CLASS_OFFSET, MulWOpcode::COUNT),
            ("DivRemW", DivRemWOpcode::CLASS_OFFSET, DivRemWOpcode::COUNT),
            (
                "BaseAluImm",
                BaseAluImmOpcode::CLASS_OFFSET,
                BaseAluImmOpcode::COUNT,
            ),
            (
                "ShiftImm",
                ShiftImmOpcode::CLASS_OFFSET,
                ShiftImmOpcode::COUNT,
            ),
            (
                "LessThanImm",
                LessThanImmOpcode::CLASS_OFFSET,
                LessThanImmOpcode::COUNT,
            ),
            ("ShAdd", ShAddOpcode::CLASS_OFFSET, ShAddOpcode::COUNT),
            ("SlliUw", SlliUwOpcode::CLASS_OFFSET, SlliUwOpcode::COUNT),
            (
                "BitwiseInv",
                BitwiseInvOpcode::CLASS_OFFSET,
                BitwiseInvOpcode::COUNT,
            ),
            ("Rotate", RotateOpcode::CLASS_OFFSET, RotateOpcode::COUNT),
            (
                "RotateImm",
                RotateImmOpcode::CLASS_OFFSET,
                RotateImmOpcode::COUNT,
            ),
            ("RotateW", RotateWOpcode::CLASS_OFFSET, RotateWOpcode::COUNT),
            (
                "RotateWImm",
                RotateWImmOpcode::CLASS_OFFSET,
                RotateWImmOpcode::COUNT,
            ),
            (
                "CountZeros",
                CountZerosOpcode::CLASS_OFFSET,
                CountZerosOpcode::COUNT,
            ),
            (
                "CountZerosW",
                CountZerosWOpcode::CLASS_OFFSET,
                CountZerosWOpcode::COUNT,
            ),
            ("Cpop", CpopOpcode::CLASS_OFFSET, CpopOpcode::COUNT),
            ("CpopW", CpopWOpcode::CLASS_OFFSET, CpopWOpcode::COUNT),
            ("MinMax", MinMaxOpcode::CLASS_OFFSET, MinMaxOpcode::COUNT),
            (
                "ByteUnary",
                ByteUnaryOpcode::CLASS_OFFSET,
                ByteUnaryOpcode::COUNT,
            ),
            (
                "SingleBit",
                SingleBitOpcode::CLASS_OFFSET,
                SingleBitOpcode::COUNT,
            ),
            (
                "SingleBitImm",
                SingleBitImmOpcode::CLASS_OFFSET,
                SingleBitImmOpcode::COUNT,
            ),
        ]
    }

    #[test]
    fn opcode_classes_are_disjoint() {
        let mut classes = all_classes();
        classes.sort_by_key(|(_, offset, _)| *offset);
        for pair in classes.windows(2) {
            let (prev_name, prev_offset, prev_count) = pair[0];
            let (next_name, next_offset, _) = pair[1];
            assert!(
                prev_offset + prev_count <= next_offset,
                "opcode classes {prev_name} ({prev_offset:#x}+{prev_count}) and {next_name} \
                 ({next_offset:#x}) overlap"
            );
        }
    }

    #[test]
    fn bitmanip_classes_stay_in_reserved_range() {
        // The B-extension classes live in 0x2A0..0x2C8; the next extension
        // (keccak256) starts at 0x310. Growing past 0x2FF requires checking
        // the workspace-wide offset allocation again.
        let bitmanip_start = ShAddOpcode::CLASS_OFFSET;
        let bitmanip_end = SingleBitImmOpcode::CLASS_OFFSET + SingleBitImmOpcode::COUNT;
        assert_eq!(bitmanip_start, 0x2a0);
        assert_eq!(bitmanip_end, 0x2c8);
    }
}
