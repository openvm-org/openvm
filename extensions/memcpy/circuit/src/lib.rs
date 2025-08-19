mod bus;
mod extension;
mod iteration;
mod loops;

pub use bus::*;
pub use extension::*;
pub use iteration::*;
pub use loops::*;
use openvm_circuit::system::memory::{
    merkle::public_values::PUBLIC_VALUES_AS,
    online::{GuestMemory, TracingMemory},
};
use openvm_instructions::riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS};

// ==== Do not change these constants! ====
pub const MEMCPY_LOOP_NUM_LIMBS: usize = 4;
pub const MEMCPY_LOOP_LIMB_BITS: usize = 8;

pub const A1_REGISTER_PTR: usize = 11 * 4;
pub const A2_REGISTER_PTR: usize = 12 * 4;
pub const A3_REGISTER_PTR: usize = 13 * 4;
pub const A4_REGISTER_PTR: usize = 14 * 4;

// TODO: These are duplicated from extensions/rv32im/circuit/src/adapters/mod.rs
// to prevent cyclic dependencies. Fix this.

#[inline(always)]
pub fn memory_read<const N: usize>(memory: &GuestMemory, address_space: u32, ptr: u32) -> [u8; N] {
    debug_assert!(
        address_space == RV32_REGISTER_AS
            || address_space == RV32_MEMORY_AS
            || address_space == PUBLIC_VALUES_AS,
    );

    // SAFETY:
    // - address space `RV32_REGISTER_AS` and `RV32_MEMORY_AS` will always have cell type `u8` and
    //   minimum alignment of `RV32_REGISTER_NUM_LIMBS`
    unsafe { memory.read::<u8, N>(address_space, ptr) }
}

/// Atomic read operation which increments the timestamp by 1.
/// Returns `(t_prev, [ptr:4]_{address_space})` where `t_prev` is the timestamp of the last memory
/// access.
#[inline(always)]
pub fn timed_read<const N: usize>(
    memory: &mut TracingMemory,
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
    //   minimum alignment of `MEMCPY_LOOP_NUM_LIMBS`
    unsafe { memory.read::<u8, N, MEMCPY_LOOP_NUM_LIMBS>(address_space, ptr) }
}

#[inline(always)]
pub fn timed_write<const N: usize>(
    memory: &mut TracingMemory,
    address_space: u32,
    ptr: u32,
    data: [u8; N],
) -> (u32, [u8; N]) {
    debug_assert!(
        address_space == RV32_REGISTER_AS
            || address_space == RV32_MEMORY_AS
            || address_space == PUBLIC_VALUES_AS
    );

    // SAFETY:
    // - address space `RV32_REGISTER_AS` and `RV32_MEMORY_AS` will always have cell type `u8` and
    //   minimum alignment of `MEMCPY_LOOP_NUM_LIMBS`
    unsafe { memory.write::<u8, N, MEMCPY_LOOP_NUM_LIMBS>(address_space, ptr, data) }
}

/// Reads register value at `reg_ptr` from memory and records the memory access in mutable buffer.
/// Trace generation relevant to this memory access can be done fully from the recorded buffer.
#[inline(always)]
pub fn tracing_read<const N: usize>(
    memory: &mut TracingMemory,
    address_space: u32,
    ptr: u32,
    prev_timestamp: &mut u32,
) -> [u8; N] {
    let (t_prev, data) = timed_read(memory, address_space, ptr);
    *prev_timestamp = t_prev;
    data
}

/// Writes `reg_ptr, reg_val` into memory and records the memory access in mutable buffer.
/// Trace generation relevant to this memory access can be done fully from the recorded buffer.
#[inline(always)]
pub fn tracing_write<const N: usize>(
    memory: &mut TracingMemory,
    address_space: u32,
    ptr: u32,
    data: [u8; N],
    prev_timestamp: &mut u32,
    prev_data: &mut [u8; N],
) {
    let (t_prev, data_prev) = timed_write(memory, address_space, ptr, data);
    *prev_timestamp = t_prev;
    *prev_data = data_prev;
}

#[inline(always)]
pub fn read_rv32_register(memory: &GuestMemory, ptr: u32) -> u32 {
    u32::from_le_bytes(memory_read(memory, RV32_REGISTER_AS, ptr))
}
