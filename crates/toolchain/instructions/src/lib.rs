//! This crate is intended for use on host machine. This includes usage within procedural macros.

#![allow(non_camel_case_types)]

use openvm_instructions_derive::LocalOpcode;
use openvm_stark_backend::p3_field::Field;
use serde::{Deserialize, Serialize};
use strum_macros::{EnumCount, EnumIter, FromRepr};

pub mod exe;
pub mod instruction;
mod phantom;
pub mod program;
/// Module with traits and constants for RISC-V instruction definitions for custom OpenVM
/// instructions.
pub mod riscv;
pub mod utils;

pub use phantom::*;

pub const NATIVE_AS: u32 = 4;

pub trait LocalOpcode {
    const CLASS_OFFSET: usize;
    /// Convert from the discriminant of the enum to the typed enum variant.
    /// Default implementation uses `from_repr`.
    fn from_usize(value: usize) -> Self;
    fn local_usize(&self) -> usize;

    fn global_opcode_usize(&self) -> usize {
        self.local_usize() + Self::CLASS_OFFSET
    }
    fn global_opcode(&self) -> VmOpcode {
        VmOpcode::from_usize(self.global_opcode_usize())
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, derive_new::new, Serialize, Deserialize)]
pub struct VmOpcode(usize);

impl VmOpcode {
    /// Returns the corresponding `local_opcode_idx`
    #[inline(always)]
    pub const fn local_opcode_idx(&self, offset: usize) -> usize {
        self.as_usize() - offset
    }

    /// Returns the opcode as a usize
    #[inline(always)]
    pub const fn as_usize(&self) -> usize {
        self.0
    }

    /// Create a new [VmOpcode] from a usize
    pub const fn from_usize(value: usize) -> Self {
        Self(value)
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
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0]
#[repr(usize)]
pub enum SystemOpcode {
    TERMINATE,
    PHANTOM,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x020]
#[repr(usize)]
pub enum PublishOpcode {
    PUBLISH,
}

// =================================================================================================
// For internal dev use only
// =================================================================================================

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0xdeadaf]
#[repr(usize)]
pub enum UnimplementedOpcode {
    REPLACE_ME,
}
