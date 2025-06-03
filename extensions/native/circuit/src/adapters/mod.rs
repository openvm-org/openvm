use openvm_circuit::{
    arch::{execution_mode::E1E2ExecutionCtx, VmStateMut},
    system::memory::{
        offline_checker::{MemoryBaseAuxCols, MemoryReadOrImmediateAuxCols, MemoryWriteAuxCols},
        online::{GuestMemory, TracingMemory},
    },
};
use openvm_native_compiler::conversion::AS;
use openvm_stark_backend::p3_field::PrimeField32;

pub mod alu_native_adapter;
// 2 reads, 0 writes, imm support, jump support
pub mod branch_native_adapter;
// 1 read, 1 write, arbitrary read size, arbitrary write size, no imm support
pub mod convert_adapter;
pub mod loadstore_native_adapter;
// 2 reads, 1 write, read size = write size = N, no imm support, read/write to address space d
pub mod native_vectorized_adapter;

#[inline(always)]
pub fn memory_read_native<F, const N: usize>(memory: &GuestMemory, ptr: u32) -> [F; N]
where
    F: PrimeField32,
{
    // SAFETY:
    // - address space `AS::Native` will always have cell type `F` and minimum alignment of `1`
    unsafe { memory.read::<F, N>(AS::Native as u32, ptr) }
}

#[inline(always)]
pub fn memory_read_or_imm_native<F>(memory: &GuestMemory, addr_space: u32, ptr_or_imm: F) -> F
where
    F: PrimeField32,
{
    debug_assert!(addr_space == AS::Immediate as u32 || addr_space == AS::Native as u32);

    if addr_space == AS::Native as u32 {
        let [result]: [F; 1] = memory_read_native(memory, ptr_or_imm.as_canonical_u32());
        result
    } else {
        ptr_or_imm
    }
}
#[inline(always)]
pub fn memory_write_native<F, const N: usize>(memory: &mut GuestMemory, ptr: u32, data: &[F; N])
where
    F: PrimeField32,
{
    // SAFETY:
    // - address space `AS::Native` will always have cell type `F` and minimum alignment of `1`
    unsafe { memory.write::<F, N>(AS::Native as u32, ptr, data) }
}

#[inline(always)]
pub fn memory_read_native_from_state<Ctx, F, const N: usize>(
    state: &mut VmStateMut<F, GuestMemory, Ctx>,
    ptr: u32,
) -> [F; N]
where
    F: PrimeField32,
    Ctx: E1E2ExecutionCtx,
{
    state
        .ctx
        .on_memory_operation(AS::Native as u32, ptr, N as u32);

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
    Ctx: E1E2ExecutionCtx,
{
    debug_assert!(addr_space == AS::Immediate as u32 || addr_space == AS::Native as u32);

    if addr_space == AS::Native as u32 {
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
    data: &[F; N],
) where
    F: PrimeField32,
    Ctx: E1E2ExecutionCtx,
{
    state
        .ctx
        .on_memory_operation(AS::Native as u32, ptr, N as u32);

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
    unsafe { memory.read::<F, BLOCK_SIZE, 1>(AS::Native as u32, ptr) }
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
    unsafe { memory.write::<F, BLOCK_SIZE, 1>(AS::Native as u32, ptr, vals) }
}

/// Reads register value at `ptr` from memory and records the memory access in mutable buffer.
/// Trace generation relevant to this memory access can be done fully from the recorded buffer.
#[inline(always)]
pub fn tracing_read_native<F, const BLOCK_SIZE: usize>(
    memory: &mut TracingMemory<F>,
    ptr: u32,
    aux_cols: &mut MemoryBaseAuxCols<F>,
) -> [F; BLOCK_SIZE]
where
    F: PrimeField32,
{
    let (t_prev, data) = timed_read(memory, ptr);
    aux_cols.set_prev(F::from_canonical_u32(t_prev));
    data
}

/// Writes `ptr, vals` into memory and records the memory access in mutable buffer.
/// Trace generation relevant to this memory access can be done fully from the recorded buffer.
#[inline(always)]
pub fn tracing_write_native<F, const BLOCK_SIZE: usize>(
    memory: &mut TracingMemory<F>,
    ptr: u32,
    vals: &[F; BLOCK_SIZE],
    aux_cols: &mut MemoryWriteAuxCols<F, BLOCK_SIZE>,
) where
    F: PrimeField32,
{
    let (t_prev, data_prev) = timed_write(memory, ptr, vals);
    aux_cols.set_prev(F::from_canonical_u32(t_prev), data_prev);
}

/// Reads value at `_ptr` from memory and records the memory access in mutable buffer.
/// Trace generation relevant to this memory access can be done fully from the recorded buffer.
#[inline(always)]
pub fn tracing_read_or_imm_native<F>(
    memory: &mut TracingMemory<F>,
    addr_space: u32,
    ptr_or_imm: F,
    addr_space_mut: &mut F,
    aux_cols: &mut MemoryReadOrImmediateAuxCols<F>,
) -> F
where
    F: PrimeField32,
{
    debug_assert!(
        addr_space == AS::Immediate as u32 || addr_space == AS::Native as u32,
        "addr_space={} is not valid",
        addr_space
    );

    if addr_space == AS::Immediate as u32 {
        *addr_space_mut = F::ZERO;
        memory.increment_timestamp();
        ptr_or_imm
    } else {
        *addr_space_mut = F::from_canonical_u32(AS::Native as u32);
        let data: [F; 1] =
            tracing_read_native(memory, ptr_or_imm.as_canonical_u32(), &mut aux_cols.base);
        data[0]
    }
}
