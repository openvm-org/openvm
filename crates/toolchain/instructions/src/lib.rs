//! This crate is intended for use on host machine. This includes usage within procedural macros.

#![allow(non_camel_case_types)]

use axvm_instructions_derive::UsizeOpcode;
use strum_macros::{EnumCount, EnumIter, FromRepr};

pub mod exe;
pub mod instruction;
mod phantom;
pub mod program;
/// Module with traits and constants for RISC-V instruction definitions for custom axVM instructions.
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
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, UsizeOpcode,
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
