/// Size of an RV64 register in bytes.
pub const REGISTER_BYTES: u64 = 8;
/// Number of byte limbs used for arrays and indexing.
pub const REGISTER_NUM_LIMBS: usize = REGISTER_BYTES as usize;
pub const BYTE_BITS: usize = 8;
/// 32-bit word stored as 4 bytes (4 limbs of 8-bits), i.e. half a 64-bit register.
pub const WORD_NUM_LIMBS: usize = REGISTER_NUM_LIMBS / 2;

pub const IMM_AS: u32 = 0;
pub const REGISTER_AS: u32 = 1;
pub const MEMORY_AS: u32 = 2;

pub const NUM_REGISTERS: usize = 32;

/// Whether an OpenVM operand encodes an aligned RV64 register pointer.
#[inline]
pub fn is_valid_register_pointer(pointer: u32) -> bool {
    let pointer = u64::from(pointer);
    pointer < NUM_REGISTERS as u64 * REGISTER_BYTES && pointer.is_multiple_of(REGISTER_BYTES)
}
