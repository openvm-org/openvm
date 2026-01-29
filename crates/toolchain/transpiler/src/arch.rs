//! Architecture-specific type aliases and constants for RV32/RV64 support.
//!
//! By default, this module provides RV32 types. Enable the `rv64` feature to
//! switch to RV64 types.
//!
//! # Example
//!
//! ```toml
//! [dependencies]
//! openvm-transpiler = { version = "...", features = ["rv64"] }
//! ```

use elf::file::Class;

// =============================================================================
// Type aliases - these change based on the rv64 feature
// =============================================================================

/// Address type for memory addresses and program counter.
/// - RV32: u32 (32-bit addresses)
/// - RV64: u64 (64-bit addresses)
#[cfg(not(feature = "rv64"))]
pub type Addr = u32;
#[cfg(feature = "rv64")]
pub type Addr = u64;

/// Word type for data words in memory.
/// - RV32: u32 (32-bit words)
/// - RV64: u64 (64-bit words)
#[cfg(not(feature = "rv64"))]
pub type Word = u32;
#[cfg(feature = "rv64")]
pub type Word = u64;

// =============================================================================
// Constants - these change based on the rv64 feature
// =============================================================================

/// Register width in bits (XLEN in RISC-V terminology).
#[cfg(not(feature = "rv64"))]
pub const XLEN: usize = 32;
#[cfg(feature = "rv64")]
pub const XLEN: usize = 64;

/// Word size in bytes.
#[cfg(not(feature = "rv64"))]
pub const WORD_SIZE: usize = 4;
#[cfg(feature = "rv64")]
pub const WORD_SIZE: usize = 8;

/// Number of limbs per register in OpenVM memory.
/// Each limb is 8 bits, so:
/// - RV32: 4 limbs (32 bits / 8 bits)
/// - RV64: 8 limbs (64 bits / 8 bits)
#[cfg(not(feature = "rv64"))]
pub const REGISTER_NUM_LIMBS: usize = 4;
#[cfg(feature = "rv64")]
pub const REGISTER_NUM_LIMBS: usize = 8;

/// Expected ELF class for this architecture.
#[cfg(not(feature = "rv64"))]
pub const ELF_CLASS: Class = Class::ELF32;
#[cfg(feature = "rv64")]
pub const ELF_CLASS: Class = Class::ELF64;

/// Maximum address value.
#[cfg(not(feature = "rv64"))]
pub const ADDR_MAX: Addr = u32::MAX;
#[cfg(feature = "rv64")]
pub const ADDR_MAX: Addr = u64::MAX;

// =============================================================================
// Helper functions
// =============================================================================

/// Convert a u64 to the address type, with appropriate truncation/validation.
#[cfg(not(feature = "rv64"))]
pub fn addr_from_u64(val: u64) -> Result<Addr, &'static str> {
    val.try_into().map_err(|_| "address too large for RV32")
}
#[cfg(feature = "rv64")]
pub fn addr_from_u64(val: u64) -> Result<Addr, &'static str> {
    Ok(val)
}

/// Convert the address type to u64 (always succeeds).
pub fn addr_to_u64(val: Addr) -> u64 {
    val as u64
}

/// Convert a u64 to the word type, with appropriate truncation/validation.
#[cfg(not(feature = "rv64"))]
pub fn word_from_u64(val: u64) -> Result<Word, &'static str> {
    val.try_into().map_err(|_| "word too large for RV32")
}
#[cfg(feature = "rv64")]
pub fn word_from_u64(val: u64) -> Result<Word, &'static str> {
    Ok(val)
}

/// Convert the word type to u64 (always succeeds).
pub fn word_to_u64(val: Word) -> u64 {
    val as u64
}

/// Architecture description string for error messages.
#[cfg(not(feature = "rv64"))]
pub const ARCH_NAME: &str = "RV32";
#[cfg(feature = "rv64")]
pub const ARCH_NAME: &str = "RV64";
