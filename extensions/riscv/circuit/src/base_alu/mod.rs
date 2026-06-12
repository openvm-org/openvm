//! The BaseAlu core chip is no longer used by the RV64 extension itself (the 64-bit ALU is
//! split into the `add_sub` and `xor_or_and` chips), but the core AIR/executor/filler are
//! still used by the bigint (INT256) extension.

mod core;
pub use core::*;
