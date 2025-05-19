use std::ops::Mul;

use openvm_circuit::{
    arch::{execution_mode::E1E2ExecutionCtx, VmStateMut},
    system::memory::{
        offline_checker::{MemoryBaseAuxCols, MemoryReadAuxCols, MemoryWriteAuxCols},
        online::{GuestMemory, TracingMemory},
        tree::public_values::PUBLIC_VALUES_AS,
        MemoryController, RecordId,
    },
};
use openvm_instructions::riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS};
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

#[inline(always)]
pub fn memory_read<const N: usize>(memory: &GuestMemory, address_space: u32, ptr: u32) -> [u8; N] {
    debug_assert!(
        address_space == RV32_REGISTER_AS
            || address_space == RV32_MEMORY_AS
            || address_space == PUBLIC_VALUES_AS,
    );

    // TODO(ayush): PUBLIC_VALUES_AS safety?
    // SAFETY:
    // - address space `RV32_REGISTER_AS` and `RV32_MEMORY_AS` will always have cell type `u8` and
    //   minimum alignment of `RV32_REGISTER_NUM_LIMBS`
    unsafe { memory.read::<u8, N>(address_space, ptr) }
}

#[inline(always)]
pub fn memory_write<const N: usize>(
    memory: &mut GuestMemory,
    address_space: u32,
    ptr: u32,
    data: &[u8; N],
) {
    debug_assert!(
        address_space == RV32_REGISTER_AS
            || address_space == RV32_MEMORY_AS
            || address_space == PUBLIC_VALUES_AS
    );

    // TODO(ayush): PUBLIC_VALUES_AS safety?
    // SAFETY:
    // - address space `RV32_REGISTER_AS` and `RV32_MEMORY_AS` will always have cell type `u8` and
    //   minimum alignment of `RV32_REGISTER_NUM_LIMBS`
    unsafe { memory.write::<u8, N>(address_space, ptr, data) }
}

#[inline(always)]
pub fn memory_read_from_state<Ctx, const N: usize>(
    state: &mut VmStateMut<GuestMemory, Ctx>,
    address_space: u32,
    ptr: u32,
) -> [u8; N]
where
    Ctx: E1E2ExecutionCtx,
{
    state.ctx.on_memory_operation(address_space, ptr, N);

    memory_read(state.memory, address_space, ptr)
}

#[inline(always)]
pub fn memory_write_from_state<Ctx, const N: usize>(
    state: &mut VmStateMut<GuestMemory, Ctx>,
    address_space: u32,
    ptr: u32,
    data: &[u8; N],
) where
    Ctx: E1E2ExecutionCtx,
{
    state.ctx.on_memory_operation(address_space, ptr, N);

    memory_write(state.memory, address_space, ptr, data)
}

/// Atomic read operation which increments the timestamp by 1.
/// Returns `(t_prev, [ptr:4]_{address_space})` where `t_prev` is the timestamp of the last memory
/// access.
#[inline(always)]
pub fn timed_read<F: PrimeField32, const N: usize>(
    memory: &mut TracingMemory<F>,
    address_space: u32,
    ptr: u32,
) -> (u32, [u8; N]) {
    debug_assert!(
        address_space == RV32_REGISTER_AS
            || address_space == RV32_MEMORY_AS
            || address_space == PUBLIC_VALUES_AS
    );

    // SAFETY:
    // - address space `RV32_REGISTER_AS` and `RV32_MEMORY_AS` will always have cell type `u8` and
    //   minimum alignment of `RV32_REGISTER_NUM_LIMBS`
    unsafe { memory.read::<u8, N, RV32_REGISTER_NUM_LIMBS>(address_space, ptr) }
}

#[inline(always)]
pub fn timed_write<F: PrimeField32, const N: usize>(
    memory: &mut TracingMemory<F>,
    address_space: u32,
    ptr: u32,
    data: &[u8; N],
) -> (u32, [u8; N]) {
    // TODO(ayush): should this allow public values address space
    debug_assert!(
        address_space == RV32_REGISTER_AS
            || address_space == RV32_MEMORY_AS
            || address_space == PUBLIC_VALUES_AS
    );

    // SAFETY:
    // - address space `RV32_REGISTER_AS` and `RV32_MEMORY_AS` will always have cell type `u8` and
    //   minimum alignment of `RV32_REGISTER_NUM_LIMBS`
    unsafe { memory.write::<u8, N, RV32_REGISTER_NUM_LIMBS>(address_space, ptr, data) }
}

/// Reads register value at `reg_ptr` from memory and records the memory access in mutable buffer.
/// Trace generation relevant to this memory access can be done fully from the recorded buffer.
#[inline(always)]
pub fn tracing_read<F, const N: usize>(
    memory: &mut TracingMemory<F>,
    address_space: u32,
    ptr: u32,
    aux_cols: &mut MemoryReadAuxCols<F>, /* TODO[jpw]: switch to raw u8
                                          * buffer */
) -> [u8; N]
where
    F: PrimeField32,
{
    let (t_prev, data) = timed_read(memory, address_space, ptr);
    aux_cols.set_prev(F::from_canonical_u32(t_prev));
    data
}

/// Writes `reg_ptr, reg_val` into memory and records the memory access in mutable buffer.
/// Trace generation relevant to this memory access can be done fully from the recorded buffer.
#[inline(always)]
pub fn tracing_write<F, const N: usize>(
    memory: &mut TracingMemory<F>,
    address_space: u32,
    ptr: u32,
    data: &[u8; N],
    aux_cols: &mut MemoryWriteAuxCols<F, N>, /* TODO[jpw]: switch to raw
                                              * u8
                                              * buffer */
) where
    F: PrimeField32,
{
    let (t_prev, data_prev) = timed_write(memory, address_space, ptr, data);
    aux_cols.set_prev(
        F::from_canonical_u32(t_prev),
        data_prev.map(F::from_canonical_u8),
    );
}

// TODO(ayush): this is bad but not sure how to avoid
#[inline(always)]
pub fn tracing_write_with_base_aux<F, const N: usize>(
    memory: &mut TracingMemory<F>,
    address_space: u32,
    ptr: u32,
    data: &[u8; N],
    base_aux_cols: &mut MemoryBaseAuxCols<F>,
) where
    F: PrimeField32,
{
    let (t_prev, _) = timed_write(memory, address_space, ptr, data);
    base_aux_cols.set_prev(F::from_canonical_u32(t_prev));
}

#[inline(always)]
pub fn tracing_read_imm<F>(
    memory: &mut TracingMemory<F>,
    imm: u32,
    imm_mut: &mut F,
) -> [u8; RV32_REGISTER_NUM_LIMBS]
where
    F: PrimeField32,
{
    *imm_mut = F::from_canonical_u32(imm);
    debug_assert_eq!(imm >> 24, 0); // highest byte should be zero to prevent overflow

    memory.increment_timestamp();

    let mut imm_le = imm.to_le_bytes();
    // Important: we set the highest byte equal to the second highest byte, using the assumption
    // that imm is at most 24 bits
    imm_le[3] = imm_le[2];
    imm_le
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

#[inline(always)]
pub fn new_read_rv32_register(memory: &GuestMemory, address_space: u32, ptr: u32) -> u32 {
    u32::from_le_bytes(memory_read(memory, address_space, ptr))
}

// TODO(AG): if "register", why `address_space` is not hardcoded to be 1?
#[inline(always)]
pub fn new_read_rv32_register_from_state<Ctx>(
    state: &mut VmStateMut<GuestMemory, Ctx>,
    address_space: u32,
    ptr: u32,
) -> u32
where
    Ctx: E1E2ExecutionCtx,
{
    u32::from_le_bytes(memory_read_from_state(state, address_space, ptr))
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
