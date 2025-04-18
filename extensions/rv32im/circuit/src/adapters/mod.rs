use std::ops::Mul;

use openvm_circuit::system::memory::{online::TracingMemory, MemoryController, RecordId};
use openvm_instructions::riscv::RV32_REGISTER_AS;
use openvm_stark_backend::p3_field::{FieldAlgebra, PrimeField32};

mod alu;
mod branch;
mod jalr;
mod loadstore;
mod mul;
mod rdwrite;

pub use alu::*;
pub use branch::*;
pub use jalr::*;
pub use loadstore::*;
pub use mul::*;
pub use openvm_instructions::riscv::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
pub use rdwrite::*;

/// 256-bit heap integer stored as 32 bytes (32 limbs of 8-bits)
pub const INT256_NUM_LIMBS: usize = 32;

// For soundness, should be <= 16
pub const RV_IS_TYPE_IMM_BITS: usize = 12;

// Branch immediate value is in [-2^12, 2^12)
pub const RV_B_TYPE_IMM_BITS: usize = 13;

pub const RV_J_TYPE_IMM_BITS: usize = 21;

/// Convert the RISC-V register data (32 bits represented as 4 bytes, where each byte is represented
/// as a field element) back into its value as u32.
pub fn compose<F: PrimeField32>(ptr_data: [F; RV32_REGISTER_NUM_LIMBS]) -> u32 {
    let mut val = 0;
    for (i, limb) in ptr_data.map(|x| x.as_canonical_u32()).iter().enumerate() {
        val += limb << (i * 8);
    }
    val
}

/// inverse of `compose`
pub fn decompose<F: PrimeField32>(value: u32) -> [F; RV32_REGISTER_NUM_LIMBS] {
    std::array::from_fn(|i| {
        F::from_canonical_u32((value >> (RV32_CELL_BITS * i)) & ((1 << RV32_CELL_BITS) - 1))
    })
}

pub fn tracing_read_reg(
    memory: &mut TracingMemory,
    reg_ptr: u32,
) -> (u32, [u8; RV32_REGISTER_NUM_LIMBS]) {
    // SAFETY:
    // - address space `RV32_REGISTER_AS` will always have cell type `u8` and minimum alignment of
    //   `RV32_REGISTER_NUM_LIMBS`
    unsafe {
        memory
            .read::<u8, RV32_REGISTER_NUM_LIMBS, RV32_REGISTER_NUM_LIMBS>(RV32_REGISTER_AS, reg_ptr)
    }
}

pub fn tracing_write_reg(
    memory: &mut TracingMemory,
    reg_ptr: u32,
    reg_val: &[u8; RV32_REGISTER_NUM_LIMBS],
) -> (u32, [u8; RV32_REGISTER_NUM_LIMBS]) {
    // SAFETY:
    // - address space `RV32_REGISTER_AS` will always have cell type `u8` and minimum alignment of
    //   `RV32_REGISTER_NUM_LIMBS`
    unsafe {
        memory.write::<u8, RV32_REGISTER_NUM_LIMBS, RV32_REGISTER_NUM_LIMBS>(
            RV32_REGISTER_AS,
            reg_ptr,
            reg_val,
        )
    }
}

/// Read register value as [RV32_REGISTER_NUM_LIMBS] limbs from memory.
/// Returns the read record and the register value as u32.
/// Does not make any range check calls.
pub fn read_rv32_register<F: PrimeField32>(
    memory: &mut MemoryController<F>,
    address_space: F,
    pointer: F,
) -> (RecordId, u32) {
    debug_assert_eq!(address_space, F::ONE);
    let record = memory.read::<u8, RV32_REGISTER_NUM_LIMBS>(address_space, pointer);
    let val = u32::from_le_bytes(record.1);
    (record.0, val)
}

/// Peeks at the value of a register without updating the memory state or incrementing the
/// timestamp.
pub fn unsafe_read_rv32_register<F: PrimeField32>(memory: &MemoryController<F>, pointer: F) -> u32 {
    let data = memory.unsafe_read::<u8, RV32_REGISTER_NUM_LIMBS>(F::ONE, pointer);
    u32::from_le_bytes(data)
}

pub fn abstract_compose<T: FieldAlgebra, V: Mul<T, Output = T>>(
    data: [V; RV32_REGISTER_NUM_LIMBS],
) -> T {
    data.into_iter()
        .enumerate()
        .fold(T::ZERO, |acc, (i, limb)| {
            acc + limb * T::from_canonical_u32(1 << (i * RV32_CELL_BITS))
        })
}

// TEMP[jpw]
pub fn tmp_convert_to_u8s<F: PrimeField32, const N: usize>(data: [F; N]) -> [u8; N] {
    data.map(|x| x.as_canonical_u32() as u8)
}
