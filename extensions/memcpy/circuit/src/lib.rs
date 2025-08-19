mod bus;
mod core;
mod extension;
mod iteration;

pub use core::*;
pub use extension::*;
pub use iteration::*;

// ==== Do not change these constants! ====
pub const MEMCPY_LOOP_NUM_LIMBS: usize = 4;
pub const MEMCPY_LOOP_LIMB_BITS: usize = 8;

pub const A1_REGISTER_PTR: usize = 11 * 4;
pub const A2_REGISTER_PTR: usize = 12 * 4;
pub const A3_REGISTER_PTR: usize = 13 * 4;
pub const A4_REGISTER_PTR: usize = 14 * 4;
