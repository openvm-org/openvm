mod rv32_alu;
mod rv32_heap;

pub use rv32_alu::*;
pub use rv32_heap::*;

/// 32-bit register stored as 4 bytes (4 lanes of 8-bits)
pub const RV32_REGISTER_NUM_LANES: usize = 4;
