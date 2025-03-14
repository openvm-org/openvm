#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]

use openvm_instructions::LocalOpcode;
use openvm_instructions_derive::LocalOpcode;
use openvm_rv32im_transpiler::BranchEqualOpcode;
use serde::{Deserialize, Serialize};
use strum::{EnumCount, EnumIter, FromRepr, IntoEnumIterator};

extern crate alloc;
extern crate core;

pub mod asm;
pub mod constraints;
pub mod conversion;
pub mod ir;

pub mod prelude {
    pub use openvm_native_compiler_derive::DslVariable;

    pub use crate::{asm::AsmCompiler, ir::*};
}

// =================================================================================================
// Native kernel opcodes
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
#[opcode_offset = 0x100]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum NativeLoadStoreOpcode {
    LOADW,
    STOREW,
    /// Instruction to write the next hint word into memory.
    HINT_STOREW,
}

#[derive(Copy, Clone, Debug, LocalOpcode)]
#[opcode_offset = 0x108]
pub struct NativeLoadStore4Opcode(pub NativeLoadStoreOpcode);

impl NativeLoadStore4Opcode {
    pub fn iter() -> impl Iterator<Item = Self> {
        NativeLoadStoreOpcode::iter().map(Self)
    }
}

pub const BLOCK_LOAD_STORE_SIZE: usize = 4;

#[derive(Copy, Clone, Debug, LocalOpcode)]
#[opcode_offset = 0x110]
pub struct NativeBranchEqualOpcode(pub BranchEqualOpcode);

impl NativeBranchEqualOpcode {
    pub fn iter() -> impl Iterator<Item = Self> {
        BranchEqualOpcode::iter().map(Self)
    }
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x115]
#[repr(usize)]
pub enum NativeJalOpcode {
    JAL,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x120]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum NativeRangeCheckOpcode {
    RANGE_CHECK,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x125]
#[repr(usize)]
pub enum CastfOpcode {
    CASTF,
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
#[opcode_offset = 0x130]
#[repr(usize)]
pub enum FieldArithmeticOpcode {
    ADD,
    SUB,
    MUL,
    DIV,
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
#[opcode_offset = 0x140]
#[repr(usize)]
pub enum FieldExtensionOpcode {
    FE4ADD,
    FE4SUB,
    BBE4MUL,
    BBE4DIV,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, FromRepr)]
#[repr(u16)]
pub enum NativePhantom {
    /// Native field element print
    Print = 0x10,
    /// Prepare the next input vector for hinting.
    HintInput,
    /// Prepare the little-endian bit decomposition of a variable for hinting.
    HintBits,
    /// Move data from input stream into hint space
    HintLoad,
    /// Prepare the next felt for hinting.
    HintFelt,
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
#[opcode_offset = 0x150]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Poseidon2Opcode {
    PERM_POS2,
    COMP_POS2,
}

/// Opcodes for FRI opening proofs.
#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x160]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum FriOpcode {
    /// In FRI pcs opening verification, the reduced opening polynomial is computed one evaluation
    /// per column polynomial, per opening point
    FRI_REDUCED_OPENING,
}

/// Opcodes for verify batch.
#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x170]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum VerifyBatchOpcode {
    /// In FRI pcs opening verification, the reduced opening polynomial is computed one evaluation
    /// per column polynomial, per opening point
    VERIFY_BATCH,
}
