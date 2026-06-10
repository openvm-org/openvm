#![no_std]

use strum_macros::FromRepr;

mod ops;
pub use ops::*;

/// This is custom-1 defined in RISC-V spec document
pub const OPCODE: u8 = 0x2b;
/// All deferral operations use funct3 0b111
pub const DEFERRAL_FUNCT3: u8 = 0b111;
/// Low bits in immediate used to pick deferral sub-opcode
pub const DEFERRAL_OPCODE_BITS: u32 = 2;

/// Maximum number of deferral circuits, as constrained in the deferrals part of the
/// continuations framework. Note that each deferral instruction stores its deferral
/// idx in the most significant 10 bits of the immediate field.
pub const MAX_DEF_CIRCUITS: u16 = 512;

/// Number of bytes in a commit, used as identifiers for raw deferral inputs/outputs
pub const COMMIT_NUM_BYTES: usize = 32;
/// Commit type use as a raw deferral input/output identifier
pub type Commit = [u8; COMMIT_NUM_BYTES];

/// Key for looking up raw output.
///
/// Trusted only when returned by `deferred_compute`. Manually constructed keys are
/// not trusted deferral results: `get_deferred_output` authenticates output bytes
/// against the key, but only `CALL` (via `deferred_compute`) establishes that the
/// key came from a deferred computation.
#[repr(C, align(4))]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct OutputKey {
    pub output_commit: Commit,
    pub output_len: u64,
}

impl OutputKey {
    /// Constructs an `OutputKey` value.
    ///
    /// This does not make the key a trusted deferral result.
    #[inline(always)]
    pub const fn new(output_commit: Commit, output_len: u64) -> Self {
        Self {
            output_commit,
            output_len,
        }
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
    Call = 0,
    Output = 1,
}

/// Encode deferral immediate as [def_idx(10 bits) | opcode(2 bits)]
#[inline(always)]
pub const fn encode_deferral_imm(deferral_idx: u16, opcode: DeferralImmOpcode) -> u16 {
    (deferral_idx << DEFERRAL_OPCODE_BITS) | (opcode as u16)
}
