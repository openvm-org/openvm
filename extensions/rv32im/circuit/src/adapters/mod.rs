use std::ops::Mul;

use openvm_circuit::system::memory::{
    offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols},
    online::TracingMemory,
    MemoryController, RecordId,
};
use openvm_instructions::riscv::{RV32_IMM_AS, RV32_REGISTER_AS};
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

/// Atomic read operation which increments the timestamp by 1.
/// Returns `(t_prev, [reg_ptr:4]_1)` where `t_prev` is the timestamp of the last memory access.
#[inline(always)]
pub fn timed_read_reg(
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

#[inline(always)]
pub fn timed_write_reg(
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

/// Reads register value at `reg_ptr` from memory and records the memory access in mutable buffer.
/// Trace generation relevant to this memory access can be done fully from the recorded buffer.
#[inline(always)]
pub fn tracing_read_reg<F: PrimeField32>(
    memory: &mut TracingMemory,
    reg_ptr: u32,
    (reg_ptr_mut, aux_cols): (&mut F, &mut MemoryReadAuxCols<F>), /* TODO[jpw]: switch to raw u8
                                                                   * buffer */
) -> [u8; RV32_REGISTER_NUM_LIMBS] {
    let (t_prev, data) = timed_read_reg(memory, reg_ptr);
    *reg_ptr_mut = F::from_canonical_u32(reg_ptr);
    aux_cols.set_prev(F::from_canonical_u32(t_prev));
    data
}

/// Writes `reg_ptr, reg_val` into memory and records the memory access in mutable buffer.
/// Trace generation relevant to this memory access can be done fully from the recorded buffer.
#[inline(always)]
pub fn tracing_write_reg<F: PrimeField32>(
    memory: &mut TracingMemory,
    reg_ptr: u32,
    reg_val: &[u8; RV32_REGISTER_NUM_LIMBS],
    (reg_ptr_mut, aux_cols): (&mut F, &mut MemoryWriteAuxCols<F, RV32_REGISTER_NUM_LIMBS>), /* TODO[jpw]: switch to raw u8
                                                                                             * buffer */
) {
    let (t_prev, data_prev) = timed_write_reg(memory, reg_ptr, reg_val);
    *reg_ptr_mut = F::from_canonical_u32(reg_ptr);
    aux_cols.set_prev(
        F::from_canonical_u32(t_prev),
        data_prev.map(F::from_canonical_u8),
    );
}

/// Reads register value at `reg_ptr` from memory and records the memory access in mutable buffer.
/// Trace generation relevant to this memory access can be done fully from the recorded buffer.
///
/// Assumes that `addr_space` is [RV32_IMM_AS] or [RV32_REGISTER_AS].
#[inline(always)]
pub fn tracing_read_reg_or_imm<F: PrimeField32>(
    memory: &mut TracingMemory,
    addr_space: u32,
    reg_ptr_or_imm: u32,
    addr_space_mut: &mut F,
    (reg_ptr_or_imm_mut, aux_cols): (&mut F, &mut MemoryReadAuxCols<F>),
) -> [u8; RV32_REGISTER_NUM_LIMBS] {
    debug_assert!(addr_space == RV32_IMM_AS || addr_space == RV32_REGISTER_AS);
    if addr_space == RV32_IMM_AS {
        *addr_space_mut = F::ZERO;
        let imm = reg_ptr_or_imm;
        *reg_ptr_or_imm_mut = F::from_canonical_u32(imm);
        debug_assert_eq!(imm >> 24, 0); // highest byte should be zero to prevent overflow
        memory.increment_timestamp();
        let mut imm_le = imm.to_le_bytes();
        // Important: we set the highest byte equal to the second highest byte, using the assumption
        // that imm is at most 24 bits
        imm_le[3] = imm_le[2];
        imm_le
    } else {
        *addr_space_mut = F::ONE; // F::from_canonical_u32(RV32_REGISTER_AS)
        let reg_ptr = reg_ptr_or_imm;
        tracing_read_reg(memory, reg_ptr, (reg_ptr_or_imm_mut, aux_cols))
    }
}

// TODO: delete
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
