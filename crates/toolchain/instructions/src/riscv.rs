/// Size of an RV64 register in bytes.
pub const RV64_REGISTER_BYTES: u64 = 8;
/// Number of byte limbs used for arrays and indexing.
pub const RV64_REGISTER_NUM_LIMBS: usize = RV64_REGISTER_BYTES as usize;
pub const RV64_BYTE_BITS: usize = 8;
/// 32-bit word stored as 4 bytes (4 limbs of 8-bits), i.e. half a 64-bit register.
pub const RV64_WORD_NUM_LIMBS: usize = RV64_REGISTER_NUM_LIMBS / 2;

pub const RV64_IMM_AS: u32 = 0;
pub const RV64_REGISTER_AS: u32 = 1;
pub const RV64_MEMORY_AS: u32 = 2;

pub const RV64_NUM_REGISTERS: usize = 32;
