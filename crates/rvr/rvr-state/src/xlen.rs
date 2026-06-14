//! 64-bit register width marker.
//!
//! OpenVM uses RV64, so we keep just the trait + RV64 marker that
//! the `RvState` layout needs.

use std::{
    fmt::{Debug, Display},
    hash::Hash,
};

/// Marker type for 64-bit register width.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Rv64;

/// Trait for register-width-dependent operations.
///
/// Uses marker types with associated types (rather than const generics) so
/// the register type can be `u64` for downstream code without a generic
/// width parameter.
pub trait Xlen: Copy + Clone + Send + Sync + Default + Debug + 'static {
    /// Register type (u64 for Rv64).
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
        + From<u64>
        + Into<u64>;

    /// XLEN value (64).
    const VALUE: u8;

    /// Bytes per register.
    const REG_BYTES: usize;

    /// Convert a u64 to register width.
    fn from_u64(val: u64) -> Self::Reg;

    /// Convert register to u64.
    fn to_u64(val: Self::Reg) -> u64;
}

impl Xlen for Rv64 {
    type Reg = u64;
    const VALUE: u8 = 64;
    const REG_BYTES: usize = 8;

    #[inline]
    fn from_u64(val: u64) -> u64 {
        val
    }

    #[inline]
    fn to_u64(val: u64) -> u64 {
        val
    }
}
