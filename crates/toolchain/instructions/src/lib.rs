//! This crate is intended for use on host machine. This includes usage within procedural macros.

#![allow(non_camel_case_types)]

use openvm_instructions_derive::UsizeOpcode;
use openvm_stark_backend::p3_field::Field;
use serde::{Deserialize, Serialize};
use strum_macros::{EnumCount, EnumIter, FromRepr};

pub mod exe;
pub mod instruction;
mod phantom;
pub mod program;
/// Module with traits and constants for RISC-V instruction definitions for custom OpenVM instructions.
pub mod riscv;
pub mod utils;

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

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, derive_new::new, Serialize, Deserialize)]
pub struct VmOpcode(usize);

impl VmOpcode {
    /// Returns the corresponding `local_opcode_idx`
    pub fn local_opcode_idx(&self, offset: usize) -> usize {
        self.as_usize() - offset
    }

    /// Returns the opcode as a usize
    pub fn as_usize(&self) -> usize {
        self.0
    }

    /// Create a new [VmOpcode] from a usize
    pub fn from_usize(value: usize) -> Self {
        Self(value)
    }

    /// Returns the corresponding [VmOpcode] from `local_opcode` with default offset
    pub fn with_default_offset<Opcode: UsizeOpcode>(local_opcode: Opcode) -> VmOpcode {
        Self(local_opcode.with_default_offset())
    }

    /// Convert the VmOpcode into a field element
    pub fn to_field<F: Field>(&self) -> F {
        F::from_canonical_usize(self.as_usize())
    }
}

impl std::fmt::Display for VmOpcode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
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

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0x120]
#[repr(usize)]
pub enum PublishOpcode {
    PUBLISH,
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
    UsizeOpcode,
    Serialize,
    Deserialize,
)]
#[opcode_offset = 0x150]
#[repr(usize)]
pub enum Poseidon2Opcode {
    PERM_POS2,
    COMP_POS2,
}

// =================================================================================================
// For internal dev use only
// =================================================================================================

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
)]
#[opcode_offset = 0xdeadaf]
#[repr(usize)]
pub enum UnimplementedOpcode {
    REPLACE_ME,
}
