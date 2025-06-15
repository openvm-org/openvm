// TODO(ayush): this whole file is copied from extensions/native and shouldn't be here
use openvm_circuit::system::memory::online::TracingMemory;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{
    arch::{execution_mode::E1E2ExecutionCtx, VmStateMut},
    system::memory::online::GuestMemory,
};

// TODO(ayush): should be imported from somewhere
pub(super) const AS_IMMEDIATE: u32 = 0;
pub(super) const AS_NATIVE: u32 = 4;

#[inline(always)]
pub fn memory_read_native<F, const N: usize>(memory: &GuestMemory, ptr: u32) -> [F; N]
where
    F: PrimeField32,
{
    // SAFETY:
    // - address space `AS_NATIVE` will always have cell type `F` and minimum alignment of `1`
    unsafe { memory.read::<F, N>(AS_NATIVE, ptr) }
}

#[inline(always)]
pub fn memory_write_native<F, const N: usize>(memory: &mut GuestMemory, ptr: u32, data: &[F; N])
where
    F: PrimeField32,
{
    // SAFETY:
    // - address space `AS_NATIVE` will always have cell type `F` and minimum alignment of `1`
    unsafe { memory.write::<F, N>(AS_NATIVE, ptr, data) }
}

#[inline(always)]
pub fn memory_read_native_from_state<Ctx, F, const N: usize>(
    state: &mut VmStateMut<GuestMemory, Ctx>,
    ptr: u32,
) -> [F; N]
where
    F: PrimeField32,
    Ctx: E1E2ExecutionCtx,
{
    state.ctx.on_memory_operation(AS_NATIVE, ptr, N as u32);

    memory_read_native(state.memory, ptr)
}

#[inline(always)]
pub fn memory_read_or_imm_native_from_state<Ctx, F>(
    state: &mut VmStateMut<GuestMemory, Ctx>,
    addr_space: u32,
    ptr_or_imm: F,
) -> F
where
    F: PrimeField32,
    Ctx: E1E2ExecutionCtx,
{
    debug_assert!(addr_space == AS_IMMEDIATE || addr_space == AS_NATIVE);

    if addr_space == AS_NATIVE {
        let [result]: [F; 1] = memory_read_native_from_state(state, ptr_or_imm.as_canonical_u32());
        result
    } else {
        ptr_or_imm
    }
}

#[inline(always)]
pub fn memory_write_native_from_state<Ctx, F, const N: usize>(
    state: &mut VmStateMut<GuestMemory, Ctx>,
    ptr: u32,
    data: &[F; N],
) where
    F: PrimeField32,
    Ctx: E1E2ExecutionCtx,
{
    state.ctx.on_memory_operation(AS_NATIVE, ptr, N as u32);

    memory_write_native(state.memory, ptr, data)
}

/// Atomic read operation which increments the timestamp by 1.
/// Returns `(t_prev, [ptr:BLOCK_SIZE]_4)` where `t_prev` is the timestamp of the last memory
/// access.
#[inline(always)]
fn timed_read<F, const BLOCK_SIZE: usize>(
    memory: &mut TracingMemory<F>,
    ptr: u32,
) -> (u32, [F; BLOCK_SIZE])
where
    F: PrimeField32,
{
    // SAFETY:
    // - address space `Native` will always have cell type `F` and minimum alignment of `1`
    unsafe { memory.read::<F, BLOCK_SIZE, 1>(AS_NATIVE, ptr) }
}

#[inline(always)]
fn timed_write<F, const BLOCK_SIZE: usize>(
    memory: &mut TracingMemory<F>,
    ptr: u32,
    vals: &[F; BLOCK_SIZE],
) -> (u32, [F; BLOCK_SIZE])
where
    F: PrimeField32,
{
    // SAFETY:
    // - address space `Native` will always have cell type `F` and minimum alignment of `1`
    unsafe { memory.write::<F, BLOCK_SIZE, 1>(AS_NATIVE, ptr, vals) }
}

/// Reads register value at `ptr` from memory and records the previous timestamp.
#[inline(always)]
pub fn tracing_read_native<F, const BLOCK_SIZE: usize>(
    memory: &mut TracingMemory<F>,
    ptr: u32,
    prev_timestamp: &mut u32,
) -> [F; BLOCK_SIZE]
where
    F: PrimeField32,
{
    let (t_prev, data) = timed_read(memory, ptr);
    *prev_timestamp = t_prev.into();
    data
}

/// Writes `ptr, vals` into memory and records the previous timestamp and data.
#[inline(always)]
pub fn tracing_write_native<F, const BLOCK_SIZE: usize>(
    memory: &mut TracingMemory<F>,
    ptr: u32,
    vals: &[F; BLOCK_SIZE],
    prev_timestamp: &mut u32,
    prev_data: &mut [F; BLOCK_SIZE],
) where
    F: PrimeField32,
{
    let (t_prev, data_prev) = timed_write(memory, ptr, vals);
    *prev_timestamp = t_prev.into();
    *prev_data = data_prev;
}

/// Reads value at `_ptr` from memory and records the previous timestamp.
/// If the read is an immediate, the previous timestamp will be set to `u32::MAX`.
#[inline(always)]
pub fn tracing_read_or_imm_native<F>(
    memory: &mut TracingMemory<F>,
    addr_space: u32,
    ptr_or_imm: F,
    prev_timestamp: &mut u32,
) -> F
where
    F: PrimeField32,
{
    debug_assert!(
        addr_space == AS_IMMEDIATE || addr_space == AS_NATIVE,
        "addr_space={} is not valid",
        addr_space
    );

    if addr_space == AS_IMMEDIATE {
        *prev_timestamp = u32::MAX;
        memory.increment_timestamp();
        ptr_or_imm
    } else {
        let data: [F; 1] =
            tracing_read_native(memory, ptr_or_imm.as_canonical_u32(), prev_timestamp);
        data[0]
    }
}
