mod bus;
mod extension;
mod iteration;
mod loops;

pub use bus::*;
pub use extension::*;
pub use iteration::*;
pub use loops::*;

// ==== Do not change these constants! ====
pub const MEMCPY_LOOP_NUM_LIMBS: usize = 4;
pub const MEMCPY_LOOP_LIMB_BITS: usize = 8;

pub const A1_REGISTER_PTR: usize = 11 * 4;
pub const A2_REGISTER_PTR: usize = 12 * 4;
pub const A3_REGISTER_PTR: usize = 13 * 4;
pub const A4_REGISTER_PTR: usize = 14 * 4;
