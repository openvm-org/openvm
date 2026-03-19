use openvm_circuit::system::memory::online::{GuestMemory, TracingMemory};
use openvm_instructions::DEFERRAL_AS;
use openvm_stark_backend::p3_field::PrimeField32;

#[inline(always)]
pub fn memory_read_deferral<F, const N: usize>(memory: &GuestMemory, ptr: u32) -> [F; N]
where
    F: PrimeField32,
{
    // SAFETY:
    // - address space `DEFERRAL_AS` will always have cell type `F` and minimum alignment of `1`
    unsafe { memory.read::<F, N>(DEFERRAL_AS, ptr) }
}

#[inline(always)]
pub fn memory_write_deferral<F, const N: usize>(memory: &mut GuestMemory, ptr: u32, data: [F; N])
where
    F: PrimeField32,
{
    // SAFETY:
    // - address space `DEFERRAL_AS` will always have cell type `F` and minimum alignment of `1`
    unsafe { memory.write::<F, N>(DEFERRAL_AS, ptr, data) }
}

/// Atomic read operation which increments the timestamp by 1.
/// Returns `(t_prev, [ptr:BLOCK_SIZE]_4)` where `t_prev` is the timestamp of the last memory
/// access.
#[inline(always)]
pub fn timed_read_deferral<F, const BLOCK_SIZE: usize>(
    memory: &mut TracingMemory,
    ptr: u32,
) -> (u32, [F; BLOCK_SIZE])
where
    F: PrimeField32,
{
    // SAFETY:
    // - deferral address space will always have cell type `F` and minimum alignment of `1`
    unsafe { memory.read::<F, BLOCK_SIZE, 1>(DEFERRAL_AS, ptr) }
}

#[inline(always)]
pub fn timed_write_deferral<F, const BLOCK_SIZE: usize>(
    memory: &mut TracingMemory,
    ptr: u32,
    vals: [F; BLOCK_SIZE],
) -> (u32, [F; BLOCK_SIZE])
where
    F: PrimeField32,
{
    // SAFETY:
    // - deferral address space will always have cell type `F` and minimum alignment of `1`
    unsafe { memory.write::<F, BLOCK_SIZE, 1>(DEFERRAL_AS, ptr, vals) }
}

/// Reads register value at `ptr` from memory and records the previous timestamp.
/// Reads are only done from address space [DEFERRAL_AS].
#[inline(always)]
pub fn tracing_read_deferral<F, const BLOCK_SIZE: usize>(
    memory: &mut TracingMemory,
    ptr: u32,
    prev_timestamp: &mut u32,
) -> [F; BLOCK_SIZE]
where
    F: PrimeField32,
{
    let (t_prev, data) = timed_read_deferral(memory, ptr);
    *prev_timestamp = t_prev;
    data
}

/// Writes `ptr, vals` into memory and records the previous timestamp and data.
/// Writes are only done to address space [DEFERRAL_AS].
#[inline(always)]
pub fn tracing_write_deferral<F, const BLOCK_SIZE: usize>(
    memory: &mut TracingMemory,
    ptr: u32,
    vals: [F; BLOCK_SIZE],
    prev_timestamp: &mut u32,
    prev_data: &mut [F; BLOCK_SIZE],
) where
    F: PrimeField32,
{
    let (t_prev, data_prev) = timed_write_deferral(memory, ptr, vals);
    *prev_timestamp = t_prev;
    *prev_data = data_prev;
}
