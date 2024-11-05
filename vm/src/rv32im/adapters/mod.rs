mod alu;
mod branch;
mod eq_mod;
mod heap;
mod heap_branch;
mod hintstore;
mod jalr;
mod loadstore;
mod mul;
mod rdwrite;
mod vec_heap;
mod vec_heap_two_reads;

use std::ops::Mul;

pub use alu::*;
pub use axvm_instructions::riscv::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
pub use branch::*;
pub use eq_mod::*;
pub use heap::*;
pub use heap_branch::*;
pub use hintstore::*;
pub use jalr::*;
pub use loadstore::*;
pub use mul::*;
pub use rdwrite::*;
pub use vec_heap::*;
pub use vec_heap_two_reads::*;

/// 256-bit heap integer stored as 32 bytes (32 limbs of 8-bits)
pub const INT256_NUM_LIMBS: usize = 32;

// For soundness, should be <= 16
pub const RV_IS_TYPE_IMM_BITS: usize = 12;

// Branch immediate value is in [-2^12, 2^12)
pub const RV_B_TYPE_IMM_BITS: usize = 13;

pub const RV_J_TYPE_IMM_BITS: usize = 21;

use p3_field::{AbstractField, PrimeField32};

use crate::system::memory::{MemoryController, MemoryReadRecord};

/// Convert the RISC-V register data (32 bits represented as 4 bytes, where each byte is represented as a field element)
/// back into its value as u32.
pub fn compose<F: PrimeField32>(ptr_data: [F; RV32_REGISTER_NUM_LIMBS]) -> u32 {
    let mut val = 0;
    for (i, limb) in ptr_data.map(|x| x.as_canonical_u32()).iter().enumerate() {
        val += limb << (i * 8);
    }
    val
}

/// Read register value as [RV32_REGISTER_NUM_LIMBS] limbs from memory.
/// Returns the read record and the register value as u32.
/// Does not make any range check calls.
pub fn read_rv32_register<F: PrimeField32>(
    memory: &mut MemoryController<F>,
    address_space: F,
    pointer: F,
) -> (MemoryReadRecord<F, RV32_REGISTER_NUM_LIMBS>, u32) {
    debug_assert_eq!(address_space, F::one());
    let record = memory.read::<RV32_REGISTER_NUM_LIMBS>(address_space, pointer);
    let val = compose(record.data);
    (record, val)
}

pub fn abstract_compose<T: AbstractField, V: Mul<T, Output = T>>(
    data: [V; RV32_REGISTER_NUM_LIMBS],
) -> T {
    data.into_iter()
        .enumerate()
        .fold(T::zero(), |acc, (i, limb)| {
            acc + limb * T::from_canonical_u32(1 << (i * RV32_CELL_BITS))
        })
}
