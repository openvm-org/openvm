mod rv32_alu;
mod rv32_branch;
mod rv32_heap;
mod rv32_jalr;
mod rv32_loadstore;
mod rv32_mul;
mod rv32_rdwrite;

pub use rv32_alu::*;
pub use rv32_branch::*;
pub use rv32_heap::*;
pub use rv32_jalr::*;
pub use rv32_loadstore::*;
pub use rv32_mul::*;
pub use rv32_rdwrite::*;
/// 32-bit register stored as 4 bytes (4 lanes of 8-bits)
pub const RV32_REGISTER_NUM_LANES: usize = 4;

// For soundness, should be <= 16
pub const RV_IS_TYPE_IMM_BITS: usize = 12;

pub const RV_J_TYPE_IMM_BITS: usize = 21;

pub const PC_BITS: usize = 30;

use p3_field::PrimeField32;

use crate::memory::{MemoryChip, MemoryReadRecord};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct MinimalInstruction<T> {
    pub is_valid: T,
    /// Absolute opcode number
    pub opcode: T,
}

/// Convert the RISC-V register data (32 bits represented as 4 bytes, where each byte is represented as a field element)
/// back into its value as u32.
pub fn compose<F: PrimeField32>(ptr_data: [F; 4]) -> u32 {
    let mut val = 0;
    for (i, limb) in ptr_data.map(|x| x.as_canonical_u32()).iter().enumerate() {
        val += limb << (i * 8);
    }
    val
}

pub fn read_rv32_register<F: PrimeField32>(
    memory: &mut MemoryChip<F>,
    address_space: F,
    pointer: F,
) -> (MemoryReadRecord<F, RV32_REGISTER_NUM_LANES>, u32) {
    debug_assert_eq!(address_space, F::one());
    let record = memory.read::<RV32_REGISTER_NUM_LANES>(address_space, pointer);
    let val = compose(record.data);
    (record, val)
}
