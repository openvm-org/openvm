use std::ops::Mul;

use openvm_circuit::{
    arch::{execution_mode::ExecutionCtxTrait, VmStateMut},
    system::memory::{
        merkle::public_values::PUBLIC_VALUES_AS,
        online::{GuestMemory, TracingMemory},
    },
};
use openvm_instructions::{
    program::MAX_ALLOWED_PC,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    DEFERRAL_AS,
};
use openvm_stark_backend::p3_field::{PrimeCharacteristicRing, PrimeField32};

mod alu;
mod alu_w;
mod branch;
mod jalr;
mod loadstore;
mod mul;
mod mul_w;
mod rdwrite;

pub use alu::*;
pub use alu_w::*;
pub use branch::*;
pub use jalr::*;
pub use loadstore::*;
pub use mul::*;
pub use mul_w::*;
pub use openvm_instructions::riscv::{
    RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS, RV64_CELL_BITS, RV64_REGISTER_NUM_LIMBS,
    RV64_WORD_NUM_LIMBS,
};
pub use rdwrite::*;

/// 256-bit heap integer stored as 32 bytes (32 limbs of 8-bits)
pub const INT256_NUM_LIMBS: usize = 32;

// For soundness, should be <= 16
pub const RV_IS_TYPE_IMM_BITS: usize = 12;

// Branch immediate value is in [-2^12, 2^12)
pub const RV_B_TYPE_IMM_BITS: usize = 13;

pub const RV_J_TYPE_IMM_BITS: usize = 21;

#[inline(always)]
pub fn debug_assert_valid_pc_target(to_pc: u32) {
    debug_assert!(
        to_pc <= MAX_ALLOWED_PC,
        "PC target 0x{to_pc:08x} exceeds maximum PC 0x{MAX_ALLOWED_PC:08x}"
    );
}

#[inline(always)]
pub fn debug_assert_valid_pointer(ptr: u64, pointer_max_bits: usize) {
    debug_assert!(
        pointer_max_bits < u64::BITS as usize,
        "pointer_max_bits={pointer_max_bits} is too large for u64 pointer checks"
    );
    debug_assert!(
        ptr < (1u64 << pointer_max_bits),
        "pointer {ptr} must be less than 2^{pointer_max_bits}"
    );
}

/// Convert the RISC-V register data (64 bits represented as 8 bytes, where each byte is represented
/// as a field element) back into its value as u64.
pub fn compose<F: PrimeField32>(ptr_data: [F; RV64_REGISTER_NUM_LIMBS]) -> u64 {
    let mut val: u64 = 0;
    for (i, limb) in ptr_data.map(|x| x.as_canonical_u32()).iter().enumerate() {
        val += (*limb as u64) << (i * RV64_CELL_BITS);
    }
    val
}

/// inverse of `compose`
pub fn decompose<F: PrimeField32>(value: u64) -> [F; RV64_REGISTER_NUM_LIMBS] {
    std::array::from_fn(|i| {
        F::from_u32(((value >> (RV64_CELL_BITS * i)) & ((1 << RV64_CELL_BITS) - 1)) as u32)
    })
}

/// Convert a 24-bit immediate to sign-extended RV64 bytes.
/// The immediate is a 12-bit signed value encoded into 24 bits with byte 2
/// carrying the sign. This function sign-extends it to 8 bytes.
#[inline(always)]
pub fn imm_to_bytes(imm: u32) -> [u8; RV64_REGISTER_NUM_LIMBS] {
    debug_assert_eq!(imm >> 24, 0);
    let mut imm_le = (imm as u64).to_le_bytes();
    // Sign-extend: byte 2 carries the sign, replicate to bytes 3-7
    imm_le[3] = imm_le[2];
    imm_le[4] = imm_le[2];
    imm_le[5] = imm_le[2];
    imm_le[6] = imm_le[2];
    imm_le[7] = imm_le[2];
    imm_le
}

/// Convert a 24-bit immediate (as stored in the instruction) to a sign-extended u64.
/// The immediate is a 12-bit signed value that was encoded into 24 bits with byte 2
/// carrying the sign. This function sign-extends it to 64 bits for RV64 execution.
#[inline(always)]
pub fn imm_to_u64(imm: u32) -> u64 {
    debug_assert_eq!(imm >> 24, 0);
    // The immediate is 12-bit sign-extended to 24 bits.
    // Sign-extend from 24 bits to 64 bits:
    let sign_extended = ((imm as i32) << 8) >> 8;
    sign_extended as i64 as u64
}

#[inline(always)]
pub fn memory_read_deferral<F, const N: usize>(memory: &GuestMemory, ptr: u32) -> [F; N]
where
    F: PrimeField32,
{
    // SAFETY: address space `DEFERRAL_AS` has cell type `F`
    unsafe { memory.read::<F, N>(DEFERRAL_AS, ptr) }
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
    // SAFETY: deferral address space has cell type `F`
    unsafe { memory.write::<F, BLOCK_SIZE>(DEFERRAL_AS, ptr, vals) }
}

#[inline(always)]
pub fn memory_read<const N: usize>(memory: &GuestMemory, address_space: u32, ptr: u32) -> [u8; N] {
    debug_assert!(
        address_space == RV64_REGISTER_AS
            || address_space == RV64_MEMORY_AS
            || address_space == PUBLIC_VALUES_AS,
    );

    // SAFETY:
    // - address space `RV64_REGISTER_AS` and `RV64_MEMORY_AS` will always have cell type `u8` and
    //   minimum alignment of `RV64_REGISTER_NUM_LIMBS`
    unsafe { memory.read::<u8, N>(address_space, ptr) }
}

#[inline(always)]
pub fn memory_write<const N: usize>(
    memory: &mut GuestMemory,
    address_space: u32,
    ptr: u32,
    data: [u8; N],
) {
    debug_assert!(
        address_space == RV64_REGISTER_AS
            || address_space == RV64_MEMORY_AS
            || address_space == PUBLIC_VALUES_AS
    );

    // SAFETY:
    // - address space `RV64_REGISTER_AS` and `RV64_MEMORY_AS` will always have cell type `u8` and
    //   minimum alignment of `RV64_REGISTER_NUM_LIMBS`
    unsafe { memory.write::<u8, N>(address_space, ptr, data) }
}

/// Atomic read operation which increments the timestamp by 1.
/// Returns `(t_prev, [ptr:N]_{address_space})` where `t_prev` is the timestamp of the last memory
/// access.
#[inline(always)]
pub fn timed_read<const N: usize>(
    memory: &mut TracingMemory,
    address_space: u32,
    ptr: u32,
) -> (u32, [u8; N]) {
    debug_assert!(
        address_space == RV64_REGISTER_AS
            || address_space == RV64_MEMORY_AS
            || address_space == PUBLIC_VALUES_AS
    );

    // SAFETY:
    // - address space `RV64_REGISTER_AS` and `RV64_MEMORY_AS` will always have cell type `u8` and
    //   minimum alignment of `RV64_REGISTER_NUM_LIMBS`
    #[cfg(feature = "legacy-v1-3-mem-align")]
    if address_space == RV64_MEMORY_AS {
        unsafe { memory.read::<u8, N>(address_space, ptr) }
    } else {
        unsafe { memory.read::<u8, N>(address_space, ptr) }
    }
    #[cfg(not(feature = "legacy-v1-3-mem-align"))]
    unsafe {
        memory.read::<u8, N>(address_space, ptr)
    }
}

#[inline(always)]
pub fn timed_write<const N: usize>(
    memory: &mut TracingMemory,
    address_space: u32,
    ptr: u32,
    data: [u8; N],
) -> (u32, [u8; N]) {
    debug_assert!(
        address_space == RV64_REGISTER_AS
            || address_space == RV64_MEMORY_AS
            || address_space == PUBLIC_VALUES_AS
    );

    // SAFETY:
    // - address space `RV64_REGISTER_AS` and `RV64_MEMORY_AS` will always have cell type `u8` and
    //   minimum alignment of `RV64_REGISTER_NUM_LIMBS`
    #[cfg(feature = "legacy-v1-3-mem-align")]
    if address_space == RV64_MEMORY_AS {
        unsafe { memory.write::<u8, N>(address_space, ptr, data) }
    } else {
        unsafe { memory.write::<u8, N>(address_space, ptr, data) }
    }
    #[cfg(not(feature = "legacy-v1-3-mem-align"))]
    unsafe {
        memory.write::<u8, N>(address_space, ptr, data)
    }
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

/// Reads an RV64 register, records the memory access, and returns the low 32 bits. Debug-asserts
/// the returned value fits in `pointer_max_bits` (which, for `pointer_max_bits <= 32`, also
/// implies the upper 32 bits are zero).
#[inline(always)]
pub fn tracing_read_reg_ptr(
    memory: &mut TracingMemory,
    ptr: u32,
    prev_timestamp: &mut u32,
    pointer_max_bits: usize,
) -> u32 {
    let bytes = tracing_read(memory, RV64_REGISTER_AS, ptr, prev_timestamp);
    let val = rv64_bytes_to_u32(bytes);
    debug_assert_valid_pointer(val as u64, pointer_max_bits);
    val
}

#[inline(always)]
pub fn tracing_read_imm(
    memory: &mut TracingMemory,
    imm: u32,
    imm_mut: &mut u32,
) -> [u8; RV64_REGISTER_NUM_LIMBS] {
    *imm_mut = imm;
    memory.increment_timestamp();
    imm_to_bytes(imm)
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
pub fn memory_read_from_state<F, Ctx, const N: usize>(
    state: &mut VmStateMut<F, GuestMemory, Ctx>,
    address_space: u32,
    ptr: u32,
) -> [u8; N]
where
    Ctx: ExecutionCtxTrait,
{
    state.ctx.on_memory_operation(address_space, ptr, N as u32);

    memory_read(state.memory, address_space, ptr)
}

#[inline(always)]
pub fn memory_write_from_state<F, Ctx, const N: usize>(
    state: &mut VmStateMut<F, GuestMemory, Ctx>,
    address_space: u32,
    ptr: u32,
    data: [u8; N],
) where
    Ctx: ExecutionCtxTrait,
{
    state.ctx.on_memory_operation(address_space, ptr, N as u32);

    memory_write(state.memory, address_space, ptr, data)
}

#[inline(always)]
pub fn read_rv64_register_from_state<F, Ctx>(
    state: &mut VmStateMut<F, GuestMemory, Ctx>,
    ptr: u32,
) -> u64
where
    Ctx: ExecutionCtxTrait,
{
    u64::from_le_bytes(memory_read_from_state(state, RV64_REGISTER_AS, ptr))
}

#[inline(always)]
pub fn read_rv64_register(memory: &GuestMemory, ptr: u32) -> u64 {
    u64::from_le_bytes(memory_read(memory, RV64_REGISTER_AS, ptr))
}

/// Read an RV64 register and return its value as u32, asserting (in debug) that the upper
/// 32 bits are zero.
#[inline(always)]
pub fn read_rv64_register_as_u32(memory: &GuestMemory, ptr: u32) -> u32 {
    rv64_u64_to_u32(read_rv64_register(memory, ptr))
}

/// Convert a u64 value to u32, asserting (in debug) that the upper 32 bits are zero.
#[inline(always)]
pub fn rv64_u64_to_u32(val: u64) -> u32 {
    debug_assert_eq!(val >> 32, 0, "upper 4 bytes must be zero");
    val as u32
}

/// Convert an 8-byte RV64 register value to a u32 pointer, asserting (in debug) that the upper 4
/// bytes are zero.
#[inline(always)]
pub fn rv64_bytes_to_u32(bytes: [u8; RV64_REGISTER_NUM_LIMBS]) -> u32 {
    rv64_u64_to_u32(u64::from_le_bytes(bytes))
}

/// Expand `N` limbs to `RV64_REGISTER_NUM_LIMBS` (8) by zero-padding the upper limbs. Used for
/// register bus reads where the register holds a value in fewer than 8 bytes.
pub fn expand_to_rv64_register<V: Clone + Into<T>, T: PrimeCharacteristicRing, const N: usize>(
    limbs: &[V; N],
) -> [T; RV64_REGISTER_NUM_LIMBS] {
    const { assert!(N <= RV64_REGISTER_NUM_LIMBS) }
    std::array::from_fn(|i| {
        if i < N {
            limbs[i].clone().into()
        } else {
            T::ZERO
        }
    })
}

pub fn abstract_compose<T: PrimeCharacteristicRing, V: Mul<T, Output = T>, const N: usize>(
    data: [V; N],
) -> T {
    data.into_iter()
        .enumerate()
        .fold(T::ZERO, |acc, (i, limb)| {
            acc + limb * T::from_u64(1u64 << (i * RV64_CELL_BITS))
        })
}

// TEMP[jpw]
pub fn tmp_convert_to_u8s<F: PrimeField32, const N: usize>(data: [F; N]) -> [u8; N] {
    data.map(|x| x.as_canonical_u32() as u8)
}
