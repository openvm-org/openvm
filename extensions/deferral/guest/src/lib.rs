#![no_std]

use strum_macros::FromRepr;

#[cfg(target_os = "zkvm")]
mod ops;
#[cfg(target_os = "zkvm")]
pub use ops::*;

/// This is custom-1 defined in RISC-V spec document
pub const OPCODE: u8 = 0x2b;
/// All deferral operations use funct3 0b100
pub const DEFERRAL_FUNCT3: u8 = 0b100;
/// Low bits in immediate used to pick deferral sub-opcode
pub const DEFERRAL_OPCODE_BITS: u32 = 2;

/// Location in native address space to start storing deferral accumulators
pub const NATIVE_START_POINTER: u32 = 0;
/// Maximum number of deferral circuits
pub const MAX_DEF_CIRCUITS: u16 = 1024;

/// Number of bytes in a commit, used as identifiers for raw deferral inputs/outputs
pub const COMMIT_NUM_BYTES: usize = 32;
/// Commit type use as a raw deferral input/output identifier
pub type Commit = [u8; COMMIT_NUM_BYTES];

/// Key for looking up raw output
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct OutputKey {
    pub output_commit: Commit,
    pub output_len_le: [u8; 4],
}

impl OutputKey {
    #[inline(always)]
    pub const fn new(output_commit: Commit, output_len: u32) -> Self {
        Self {
            output_commit,
            output_len_le: output_len.to_le_bytes(),
        }
    }

    #[inline(always)]
    pub const fn output_len(self) -> u32 {
        u32::from_le_bytes(self.output_len_le)
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *const u8 {
        (self as *const OutputKey).cast::<u8>()
    }

    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        (self as *mut OutputKey).cast::<u8>()
    }
}

/// Deferral sub-opcode encoded in immediate low bits
#[derive(Debug, Copy, Clone, PartialEq, Eq, FromRepr)]
#[repr(u16)]
pub enum DeferralImmOpcode {
    Setup = 0,
    Call = 1,
    Output = 2,
}

/// Encode deferral immediate as [def_idx(10 bits) | opcode(2 bits)]
#[inline(always)]
pub const fn encode_deferral_imm(deferral_idx: u16, opcode: DeferralImmOpcode) -> u16 {
    (deferral_idx << DEFERRAL_OPCODE_BITS) | (opcode as u16)
}
