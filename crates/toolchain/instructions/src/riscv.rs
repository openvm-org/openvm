/// 32-bit register stored as 4 bytes (4 limbs of 8-bits) in OpenVM memory.
pub const RV32_REGISTER_NUM_LIMBS: usize = 4;
pub const RV32_CELL_BITS: usize = 8;

pub const RV32_IMM_AS: u32 = 0;
pub const RV32_REGISTER_AS: u32 = 1;
pub const RV32_MEMORY_AS: u32 = 2;

pub const RV32_NUM_REGISTERS: usize = 32;
