use openvm_circuit::system::memory::{
    offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols},
    online::{GuestMemory, TracingMemory},
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
pub fn memory_read<F, Mem, const N: usize>(memory: &Mem, ptr: u32) -> [u8; N]
where
    Mem: GuestMemory,
{
    // SAFETY:
    // - address space `AS::Native` will always have cell type `F` and minimum alignment of `1`
    unsafe { memory.read::<F, N>(AS::Native, ptr) }
}

#[inline(always)]
pub fn memory_read_or_imm<F, Mem>(memory: &Mem, addr_space: u32, ptr_or_imm: F) -> F
where
    F: PrimeField32,
    Mem: GuestMemory,
{
    debug_assert!(
        addr_space.as_canonical_u32() == AS::Immediate
            || addr_space.as_canonical_u32() == AS::Native
    );

    if addr_space == AS::Native {
        let [result]: [F; 1] = memory_read(memory, ptr_or_imm.as_canonical_u32());
        result
    } else {
        ptr_or_imm
    }
}
#[inline(always)]
pub fn memory_write<F, Mem, const N: usize>(memory: &mut Mem, ptr: u32, data: &[u8; N])
where
    Mem: GuestMemory,
{
    // SAFETY:
    // - address space `AS::Native` will always have cell type `F` and minimum alignment of `1`
    unsafe { memory.write::<F, N>(AS::Native, ptr, data) }
}

/// Atomic read operation which increments the timestamp by 1.
/// Returns `(t_prev, [ptr:BLOCK_SIZE]_4)` where `t_prev` is the timestamp of the last memory
/// access.
#[inline(always)]
pub fn timed_read<F, const BLOCK_SIZE: usize>(
    memory: &mut TracingMemory<F>,
    ptr: u32,
) -> (u32, [F; BLOCK_SIZE])
where
    F: PrimeField32,
{
    // SAFETY:
    // - address space `Native` will always have cell type `F` and minimum alignment of `1`
    unsafe { memory.read::<F, BLOCK_SIZE, 1>(AS::Native, ptr) }
}

#[inline(always)]
pub fn timed_write<F, const BLOCK_SIZE: usize>(
    memory: &mut TracingMemory<F>,
    ptr: u32,
    vals: &[F; BLOCK_SIZE],
) -> (u32, [F; BLOCK_SIZE])
where
    F: PrimeField32,
{
    // SAFETY:
    // - address space `Native` will always have cell type `F` and minimum alignment of `1`
    unsafe { memory.write::<F, BLOCK_SIZE, 1>(AS::Native, ptr, vals) }
}

/// Reads register value at `ptr` from memory and records the memory access in mutable buffer.
/// Trace generation relevant to this memory access can be done fully from the recorded buffer.
#[inline(always)]
pub fn tracing_read<F, const BLOCK_SIZE: usize>(
    memory: &mut TracingMemory<F>,
    ptr: u32,
    (ptr_mut, aux_cols): (&mut F, &mut MemoryReadAuxCols<F>),
) -> [F; BLOCK_SIZE]
where
    F: PrimeField32,
{
    let (t_prev, data) = timed_read(memory, ptr);
    *ptr_mut = F::from_canonical_u32(ptr);
    aux_cols.set_prev(F::from_canonical_u32(t_prev));
    data
}

/// Writes `ptr, vals` into memory and records the memory access in mutable buffer.
/// Trace generation relevant to this memory access can be done fully from the recorded buffer.
#[inline(always)]
pub fn tracing_write<F, const BLOCK_SIZE: usize>(
    memory: &mut TracingMemory<F>,
    ptr: u32,
    vals: &[F; BLOCK_SIZE],
    (ptr_mut, aux_cols): (&mut F, &mut MemoryWriteAuxCols<F, BLOCK_SIZE>),
) where
    F: PrimeField32,
{
    let (t_prev, data_prev) = timed_write(memory, ptr, vals);
    *ptr_mut = F::from_canonical_u32(ptr);
    aux_cols.set_prev(F::from_canonical_u32(t_prev), data_prev);
}

/// Reads value at `_ptr` from memory and records the memory access in mutable buffer.
/// Trace generation relevant to this memory access can be done fully from the recorded buffer.
#[inline(always)]
pub fn tracing_read_or_imm<F>(
    memory: &mut TracingMemory<F>,
    addr_space: u32,
    ptr_or_imm: F,
    addr_space_mut: &mut F,
    (ptr_or_imm_mut, aux_cols): (&mut F, &mut MemoryReadAuxCols<F>),
) -> F
where
    F: PrimeField32,
{
    debug_assert!(addr_space == AS::Immediate || addr_space == AS::Native);

    if addr_space == AS::Immediate {
        *addr_space_mut = F::ZERO;
        *ptr_or_imm_mut = ptr_or_imm;
        memory.increment_timestamp();
        ptr_or_imm
    } else {
        *addr_space_mut = F::from_canonical_u32(AS::Native);
        tracing_read(
            memory,
            ptr_or_imm.as_canonical_u32(),
            (ptr_or_imm_mut, aux_cols),
        )
    }
}
