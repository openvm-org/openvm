mod rv32_alu;
mod rv32_heap;
mod rv32_loadstore;
mod rv32_mul;

pub use rv32_alu::*;
pub use rv32_heap::*;
pub use rv32_loadstore::*;
pub use rv32_mul::*;

/// 32-bit register stored as 4 bytes (4 lanes of 8-bits)
pub const RV32_REGISTER_NUM_LANES: usize = 4;

// For soundness, should be <= 16
pub const RV_IS_TYPE_IMM_BITS: usize = 12;
