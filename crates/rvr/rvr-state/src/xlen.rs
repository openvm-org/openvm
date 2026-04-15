//! 32-bit register width marker.
//!
//! OpenVM only uses RV32, so we keep just the trait + RV32 marker that
//! the `RvState` layout needs.

use std::fmt::{Debug, Display};
use std::hash::Hash;

/// Marker type for 32-bit register width.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Rv32;

/// Trait for register-width-dependent operations.
///
/// Uses marker types with associated types (rather than const generics) so
/// the register type can be `u32` for downstream code without a generic
/// width parameter.
pub trait Xlen: Copy + Clone + Send + Sync + Default + Debug + 'static {
    /// Register type (u32 for Rv32).
    type Reg: Copy
        + Clone
        + Default
        + Eq
        + Ord
        + Hash
        + Debug
        + Display
        + Send
        + Sync
        + From<u32>
        + Into<u64>;

    /// XLEN value (32).
    const VALUE: u8;

    /// Bytes per register.
    const REG_BYTES: usize;

    /// Convert a u64 to register width.
    fn from_u64(val: u64) -> Self::Reg;

    /// Convert register to u64.
    fn to_u64(val: Self::Reg) -> u64;
}

impl Xlen for Rv32 {
    type Reg = u32;
    const VALUE: u8 = 32;
    const REG_BYTES: usize = 4;

    #[inline]
    fn from_u64(val: u64) -> u32 {
        val as u32
    }

    #[inline]
    fn to_u64(val: u32) -> u64 {
        u64::from(val)
    }
}
