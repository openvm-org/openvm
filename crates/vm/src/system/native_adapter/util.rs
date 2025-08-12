use openvm_circuit::system::memory::online::TracingMemory;
use openvm_instructions::{riscv::RV32_IMM_AS, NATIVE_AS};
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{
    arch::{execution_mode::ExecutionCtxTrait, VmStateMut},
    system::memory::{offline_checker::MemoryWriteAuxCols, online::GuestMemory},
};

#[inline(always)]
pub fn memory_read_native<F, const N: usize>(memory: &GuestMemory, ptr: u32) -> [F; N]
where
    F: PrimeField32,
{
    // SAFETY:
    // - address space `NATIVE_AS` will always have cell type `F` and minimum alignment of `1`
    unsafe { memory.read::<F, N>(NATIVE_AS, ptr) }
}

#[inline(always)]
pub fn memory_read_or_imm_native<F>(memory: &GuestMemory, addr_space: u32, ptr_or_imm: F) -> F
where
    F: PrimeField32,
{
    debug_assert!(addr_space == RV32_IMM_AS || addr_space == NATIVE_AS);

    if addr_space == NATIVE_AS {
        let [result]: [F; 1] = memory_read_native(memory, ptr_or_imm.as_canonical_u32());
        result
    } else {
        ptr_or_imm
    }
}

#[inline(always)]
pub fn memory_write_native<F, const N: usize>(memory: &mut GuestMemory, ptr: u32, data: [F; N])
where
    F: PrimeField32,
{
    // SAFETY:
    // - address space `NATIVE_AS` will always have cell type `F` and minimum alignment of `1`
    unsafe { memory.write::<F, N>(NATIVE_AS, ptr, data) }
}

#[inline(always)]
pub fn memory_read_native_from_state<Ctx, F, const N: usize>(
    state: &mut VmStateMut<F, GuestMemory, Ctx>,
    ptr: u32,
) -> [F; N]
where
    F: PrimeField32,
    Ctx: ExecutionCtxTrait,
{
    state.ctx.on_memory_operation(NATIVE_AS, ptr, N as u32);

    memory_read_native(state.memory, ptr)
}

#[inline(always)]
pub fn memory_read_or_imm_native_from_state<Ctx, F>(
    state: &mut VmStateMut<F, GuestMemory, Ctx>,
    addr_space: u32,
    ptr_or_imm: F,
) -> F
where
    F: PrimeField32,
    Ctx: ExecutionCtxTrait,
{
    debug_assert!(addr_space == RV32_IMM_AS || addr_space == NATIVE_AS);

    if addr_space == NATIVE_AS {
        let [result]: [F; 1] = memory_read_native_from_state(state, ptr_or_imm.as_canonical_u32());
        result
    } else {
        ptr_or_imm
    }
}

#[inline(always)]
pub fn memory_write_native_from_state<Ctx, F, const N: usize>(
    state: &mut VmStateMut<F, GuestMemory, Ctx>,
    ptr: u32,
    data: [F; N],
) where
    F: PrimeField32,
    Ctx: ExecutionCtxTrait,
{
    state.ctx.on_memory_operation(NATIVE_AS, ptr, N as u32);

    memory_write_native(state.memory, ptr, data)
}

/// Atomic read operation which increments the timestamp by 1.
/// Returns `(t_prev, [ptr:BLOCK_SIZE]_4)` where `t_prev` is the timestamp of the last memory
/// access.
#[inline(always)]
pub fn timed_read_native<F, const BLOCK_SIZE: usize>(
    memory: &mut TracingMemory,
    ptr: u32,
) -> (u32, [F; BLOCK_SIZE])
where
    F: PrimeField32,
{
    // SAFETY:
    // - address space `Native` will always have cell type `F` and minimum alignment of `1`
    unsafe { memory.read::<F, BLOCK_SIZE, 1>(NATIVE_AS, ptr) }
}

#[inline(always)]
pub fn timed_write_native<F, const BLOCK_SIZE: usize>(
    memory: &mut TracingMemory,
    ptr: u32,
    vals: [F; BLOCK_SIZE],
) -> (u32, [F; BLOCK_SIZE])
where
    F: PrimeField32,
{
    // SAFETY:
    // - address space `Native` will always have cell type `F` and minimum alignment of `1`
    unsafe { memory.write::<F, BLOCK_SIZE, 1>(NATIVE_AS, ptr, vals) }
}

/// Reads register value at `ptr` from memory and records the previous timestamp.
/// Reads are only done from address space [NATIVE_AS].
#[inline(always)]
pub fn tracing_read_native<F, const BLOCK_SIZE: usize>(
    memory: &mut TracingMemory,
    ptr: u32,
    prev_timestamp: &mut u32,
) -> [F; BLOCK_SIZE]
where
    F: PrimeField32,
{
    let (t_prev, data) = timed_read_native(memory, ptr);
    *prev_timestamp = t_prev;
    data
}

/// Writes `ptr, vals` into memory and records the previous timestamp and data.
/// Writes are only done to address space [NATIVE_AS].
#[inline(always)]
pub fn tracing_write_native<F, const BLOCK_SIZE: usize>(
    memory: &mut TracingMemory,
    ptr: u32,
    vals: [F; BLOCK_SIZE],
    prev_timestamp: &mut u32,
    prev_data: &mut [F; BLOCK_SIZE],
) where
    F: PrimeField32,
{
    let (t_prev, data_prev) = timed_write_native(memory, ptr, vals);
    *prev_timestamp = t_prev;
    *prev_data = data_prev;
}

/// Writes `ptr, vals` into memory and records the previous timestamp and data.
#[inline(always)]
pub fn tracing_write_native_inplace<F, const BLOCK_SIZE: usize>(
    memory: &mut TracingMemory,
    ptr: u32,
    vals: [F; BLOCK_SIZE],
    cols: &mut MemoryWriteAuxCols<F, BLOCK_SIZE>,
) where
    F: PrimeField32,
{
    let (t_prev, data_prev) = timed_write_native(memory, ptr, vals);
    cols.base.set_prev(F::from_canonical_u32(t_prev));
    cols.prev_data = data_prev;
}

/// Reads value at `_ptr` from memory and records the previous timestamp.
/// If the read is an immediate, the previous timestamp will be set to `u32::MAX`.
#[inline(always)]
pub fn tracing_read_or_imm_native<F>(
    memory: &mut TracingMemory,
    addr_space: F,
    ptr_or_imm: F,
    prev_timestamp: &mut u32,
) -> F
where
    F: PrimeField32,
{
    debug_assert!(
        addr_space == F::ZERO || addr_space == F::from_canonical_u32(NATIVE_AS),
        "addr_space={} is not valid",
        addr_space
    );

    if addr_space == F::ZERO {
        *prev_timestamp = u32::MAX;
        memory.increment_timestamp();
        ptr_or_imm
    } else {
        let data: [F; 1] =
            tracing_read_native(memory, ptr_or_imm.as_canonical_u32(), prev_timestamp);
        data[0]
    }
}
