use std::ops::Mul;

use openvm_circuit::{
    arch::{execution_mode::ExecutionCtxTrait, VmStateMut, BLOCK_FE_WIDTH},
    system::memory::online::{GuestMemory, TracingMemory},
};
use openvm_circuit_primitives::encoder::Encoder;
pub use openvm_circuit_primitives::U16_BITS;
use openvm_instructions::{
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    DEFERRAL_AS, PUBLIC_VALUES_AS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};

mod alu_imm;
mod alu_imm_u16;
mod alu_reg;
mod alu_reg_u16;
mod alu_w_imm_u16;
mod alu_w_reg_u16;
mod branch;
mod jalr;
mod load;
mod mul;
mod mul_w;
mod rdwrite;
mod store;

pub use alu_imm::*;
pub use alu_imm_u16::*;
pub use alu_reg::*;
pub use alu_reg_u16::*;
pub use alu_w_imm_u16::*;
pub use alu_w_reg_u16::*;
pub use branch::*;
pub use jalr::*;
pub use load::*;
pub use mul::*;
pub use mul_w::*;
pub use openvm_instructions::riscv::{
    RV64_BYTE_BITS, RV64_REGISTER_NUM_LIMBS, RV64_WORD_NUM_LIMBS,
};
pub use rdwrite::*;
pub use store::*;

/// Number of u16 limbs needed for a low-32-bit RV64 pointer.
pub const RV64_PTR_U16_LIMBS: usize = RV64_WORD_NUM_LIMBS / 2;
/// Bit width covered by [`RV64_PTR_U16_LIMBS`].
pub const RV64_PTR_BITS: usize = U16_BITS * RV64_PTR_U16_LIMBS;
/// Number of u16 limbs in a 32-bit RV64 word (e.g. an `ADDW`/`SUBW` operand, or one half of a
/// register). Numerically equal to [`RV64_PTR_U16_LIMBS`], but named for arithmetic-word use.
pub const RV64_WORD_U16_LIMBS: usize = RV64_WORD_NUM_LIMBS / 2;

/// Supported load/store access widths in bytes.
pub(crate) const BYTE_ACCESS_WIDTH: usize = 1;
pub(crate) const HALFWORD_ACCESS_WIDTH: usize = 2;
pub(crate) const WORD_ACCESS_WIDTH: usize = 4;
pub(crate) const DOUBLEWORD_ACCESS_WIDTH: usize = 8;

pub(crate) const fn is_multi_byte_access_width(width: usize) -> bool {
    width == HALFWORD_ACCESS_WIDTH || width == WORD_ACCESS_WIDTH || width == DOUBLEWORD_ACCESS_WIDTH
}

pub(crate) const fn is_signed_multi_byte_access_width(width: usize) -> bool {
    width == HALFWORD_ACCESS_WIDTH || width == WORD_ACCESS_WIDTH
}

/// Byte shifts of an effective pointer inside an 8-byte memory block. Every load/store core
/// encodes shift `i` as selector case `i`.
pub(crate) const NUM_BYTE_SHIFTS: usize = 2 * BLOCK_FE_WIDTH;
/// Number of columns in the byte-shift selector encoding.
pub(crate) const BYTE_SHIFT_SELECTOR_WIDTH: usize = 3;
const SHIFT_SELECTOR_MAX_DEGREE: u32 = 2;

#[inline(always)]
pub(crate) fn rv64_register_pointer(pointer: u32) -> u8 {
    debug_assert!(pointer <= u32::from(u8::MAX));
    debug_assert_eq!(pointer as usize % RV64_REGISTER_NUM_LIMBS, 0);
    pointer as u8
}

/// Encodes one selector case for each byte shift, reserving the zero point for invalid rows.
pub(crate) fn shift_encoder() -> Encoder {
    let encoder = Encoder::new(NUM_BYTE_SHIFTS, SHIFT_SELECTOR_MAX_DEGREE, true);
    assert_eq!(encoder.width(), BYTE_SHIFT_SELECTOR_WIDTH);
    encoder
}

/// Packs two little-endian u8 limbs into one u16-shaped field element.
#[inline(always)]
pub fn pack_u8_pair<T: PrimeCharacteristicRing>(lo: T, hi: T) -> T {
    lo + hi * T::from_u32(1 << RV64_BYTE_BITS)
}

#[inline(always)]
pub fn pack_u8_pair_u32<T: PrimeCharacteristicRing>(lo: u32, hi: u32) -> T {
    pack_u8_pair(T::from_u32(lo), T::from_u32(hi))
}

#[inline(always)]
pub fn pack_rv64_u16_block<L, H, T>(
    low_word: &[L; RV64_WORD_NUM_LIMBS],
    high: &[H; RV64_PTR_U16_LIMBS],
) -> [T; BLOCK_FE_WIDTH]
where
    L: Clone + Into<T>,
    H: Clone + Into<T>,
    T: PrimeCharacteristicRing,
{
    [
        pack_u8_pair(low_word[0].clone().into(), low_word[1].clone().into()),
        pack_u8_pair(low_word[2].clone().into(), low_word[3].clone().into()),
        high[0].clone().into(),
        high[1].clone().into(),
    ]
}

/// Concatenates the low-word u16 limbs with the upper u16 limbs into a full RV64 register block.
/// Unlike [`pack_rv64_u16_block`], the low word is already u16-celled, so no byte packing occurs.
#[inline(always)]
pub fn concat_rv64_u16_block<L, H, T>(
    low_word: &[L; RV64_WORD_U16_LIMBS],
    high: &[H; RV64_WORD_U16_LIMBS],
) -> [T; BLOCK_FE_WIDTH]
where
    L: Clone + Into<T>,
    H: Clone + Into<T>,
    T: PrimeCharacteristicRing,
{
    std::array::from_fn(|i| {
        if i < RV64_WORD_U16_LIMBS {
            low_word[i].clone().into()
        } else {
            high[i - RV64_WORD_U16_LIMBS].clone().into()
        }
    })
}

#[inline(always)]
pub(crate) fn pack_high_u16<T, B>(
    bytes: &[B; RV64_REGISTER_NUM_LIMBS - RV64_WORD_NUM_LIMBS],
) -> [T; RV64_PTR_U16_LIMBS]
where
    T: PrimeCharacteristicRing,
    B: Copy + Into<u32>,
{
    std::array::from_fn(|i| pack_u8_pair_u32(bytes[2 * i].into(), bytes[2 * i + 1].into()))
}

/// Sign-extends a 16-bit immediate represented by `(imm, sign)` into a u32.
#[inline(always)]
pub fn sign_extend_imm16(imm: u32, sign: u32) -> u32 {
    imm + sign * (u32::MAX << U16_BITS)
}

/// Sign-extends a 32-bit value into RV64 register arithmetic form.
#[inline(always)]
pub fn sext32_to_u64(value: u32) -> u64 {
    value as i32 as i64 as u64
}

// For soundness, should be <= 16
pub const RV_IS_TYPE_IMM_BITS: usize = 12;

// Branch immediate value is in [-2^12, 2^12)
pub const RV_B_TYPE_IMM_BITS: usize = 13;

pub const RV_J_TYPE_IMM_BITS: usize = 21;

/// Composes an RV64 register byte-limb array into a `u64`.
pub fn rv64_limbs_to_u64<F: PrimeField32>(limbs: [F; RV64_REGISTER_NUM_LIMBS]) -> u64 {
    let mut val: u64 = 0;
    for (i, limb) in limbs.map(|x| x.as_canonical_u32()).iter().enumerate() {
        val += (*limb as u64) << (i * RV64_BYTE_BITS);
    }
    val
}

/// Decomposes a `u64` into RV64 register byte limbs.
pub fn u64_to_rv64_limbs<F: PrimeField32>(value: u64) -> [F; RV64_REGISTER_NUM_LIMBS] {
    std::array::from_fn(|i| {
        F::from_u32(((value >> (RV64_BYTE_BITS * i)) & ((1 << RV64_BYTE_BITS) - 1)) as u32)
    })
}

/// Converts a 24-bit instruction immediate to sign-extended RV64 register bytes.
/// The immediate is a 12-bit signed value encoded into 24 bits with byte 2
/// carrying the sign.
#[inline(always)]
pub fn imm_to_rv64_bytes(imm: u32) -> [u8; RV64_REGISTER_NUM_LIMBS] {
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

/// Converts a 24-bit instruction immediate to a sign-extended RV64 value.
/// The immediate is a 12-bit signed value that was encoded into 24 bits with byte 2
/// carrying the sign.
#[inline(always)]
pub fn imm_to_rv64_u64(imm: u32) -> u64 {
    debug_assert_eq!(imm >> 24, 0);
    // The immediate is 12-bit sign-extended to 24 bits.
    // Sign-extend from 24 bits to 64 bits:
    let sign_extended = ((imm as i32) << 8) >> 8;
    sign_extended as i64 as u64
}

/// Returns whether `imm` is the canonical 24-bit sign extension of a signed 12-bit immediate.
#[inline(always)]
pub fn is_canonical_i12(imm: u32) -> bool {
    let low11 = imm & ((1 << 11) - 1);
    let sign = (imm >> 11) & 1;
    imm == low11 + sign * 0xff_f800
}

#[inline(always)]
pub fn byte_ptr_to_u16_ptr<AB: InteractionBuilder>(byte_ptr: impl Into<AB::Expr>) -> AB::Expr {
    byte_ptr.into() * AB::F::TWO.inverse()
}

/// Concrete-value form of [`byte_ptr_to_u16_ptr`].
#[inline(always)]
pub fn byte_ptr_to_u16_ptr_value(byte_ptr: u32) -> u32 {
    debug_assert_eq!(byte_ptr & 1, 0, "u16 pointer conversion requires alignment");
    byte_ptr >> 1
}

/// Converts a `u64` to `u32`, requiring the upper 32 bits to be zero.
#[inline(always)]
pub fn u64_to_u32_checked(value: u64) -> u32 {
    u32::try_from(value).expect("upper 4 bytes must be zero")
}

/// Converts RV64 register bytes to a `u32`, requiring the upper 4 bytes to be zero.
#[inline(always)]
pub fn rv64_bytes_to_u32(bytes: [u8; RV64_REGISTER_NUM_LIMBS]) -> u32 {
    u64_to_u32_checked(u64::from_le_bytes(bytes))
}

/// Attempts to convert RV64 register bytes to a `u32`, requiring the upper 4 bytes to be zero.
#[inline(always)]
pub fn try_rv64_bytes_to_u32(bytes: [u8; RV64_REGISTER_NUM_LIMBS]) -> Option<u32> {
    u32::try_from(u64::from_le_bytes(bytes)).ok()
}

/// Adds an already-sign-extended 16-bit RV64 immediate to an implemented low-32-bit address.
#[inline(always)]
pub fn rv64_address_add_imm(base: u32, imm_extended: u32) -> u64 {
    u64::from(base).wrapping_add(sext32_to_u64(imm_extended))
}

#[inline(always)]
pub fn rv64_bytes_to_u16_block(bytes: [u8; RV64_REGISTER_NUM_LIMBS]) -> [u16; BLOCK_FE_WIDTH] {
    std::array::from_fn(|i| u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]))
}

pub(crate) const RV64_BYTE_MASK: u16 = (1 << RV64_BYTE_BITS) - 1;
pub(crate) const RV64_BYTE_SIGN_BIT: u16 = 1 << (RV64_BYTE_BITS - 1);
pub(crate) const RV64_U16_SIGN_BIT: u16 = 1 << (U16_BITS - 1);

#[inline(always)]
pub(crate) fn u16_cell_byte(cell: u16, byte_idx: usize) -> u16 {
    u16::from(cell.to_le_bytes()[byte_idx])
}

#[inline(always)]
pub(crate) fn set_u16_cell_byte(cell: u16, byte_idx: usize, byte: u16) -> u16 {
    debug_assert!(byte <= RV64_BYTE_MASK);
    let mut bytes = cell.to_le_bytes();
    bytes[byte_idx] = byte as u8;
    u16::from_le_bytes(bytes)
}

/// Converts a low-32-bit value to one zero-extended RV64 u16 block.
#[inline(always)]
pub fn rv64_u32_to_u16_block(value: u32) -> [u16; BLOCK_FE_WIDTH] {
    std::array::from_fn(|i| {
        if i < RV64_PTR_U16_LIMBS {
            (value >> (U16_BITS * i)) as u16
        } else {
            0
        }
    })
}

/// Splits a 32-bit RV64 pointer into low-to-high u16 limbs.
#[inline(always)]
pub fn ptr_to_u16_limbs(ptr: u32) -> [u16; RV64_PTR_U16_LIMBS] {
    std::array::from_fn(|i| (ptr >> (U16_BITS * i)) as u16)
}

/// Field-element form of [`ptr_to_u16_limbs`].
#[inline(always)]
pub fn ptr_to_field_u16_limbs<F: PrimeCharacteristicRing>(value: u32) -> [F; RV64_PTR_U16_LIMBS] {
    ptr_to_u16_limbs(value).map(F::from_u16)
}

#[inline(always)]
pub fn rv64_u16_block_to_bytes(block: [u16; BLOCK_FE_WIDTH]) -> [u8; RV64_REGISTER_NUM_LIMBS] {
    let mut out = [0u8; RV64_REGISTER_NUM_LIMBS];
    for (i, cell) in block.into_iter().enumerate() {
        let [lo, hi] = cell.to_le_bytes();
        out[2 * i] = lo;
        out[2 * i + 1] = hi;
    }
    out
}

/// Left shift applied to the high u16 limb for the pointer-width range check.
#[inline(always)]
pub fn ptr_max_bits_shift(ptr_max_bits: usize) -> usize {
    assert!(
        (U16_BITS..=RV64_PTR_BITS).contains(&ptr_max_bits),
        "ptr_max_bits must be in [U16_BITS, RV64_PTR_BITS]"
    );
    RV64_PTR_BITS - ptr_max_bits
}

/// Range-check value for a high u16 pointer limb.
#[inline(always)]
pub fn ptr_bound_from_high_u16(high_u16: u16, ptr_max_bits: usize) -> u32 {
    u32::from(high_u16) << ptr_max_bits_shift(ptr_max_bits)
}

/// Range-check value for the high u16 limb of a low-32-bit pointer.
#[inline(always)]
pub fn ptr_bound_from_ptr(ptr: u32, ptr_max_bits: usize) -> u32 {
    let high_u16 = ptr_to_u16_limbs(ptr)[RV64_PTR_U16_LIMBS - 1];
    ptr_bound_from_high_u16(high_u16, ptr_max_bits)
}

/// Expression form of [`ptr_bound_from_high_u16`].
#[inline(always)]
pub fn ptr_bound_from_high_u16_expr<T, V>(high_u16: V, ptr_max_bits: usize) -> T
where
    T: PrimeCharacteristicRing,
    V: Into<T>,
{
    high_u16.into() * T::from_u64(1u64 << ptr_max_bits_shift(ptr_max_bits))
}

/// Composes low-to-high u16 pointer limbs into one field expression/value.
#[inline(always)]
pub fn u16_limbs_to_ptr<T, V>(limbs: &[V; RV64_PTR_U16_LIMBS]) -> T
where
    T: PrimeCharacteristicRing,
    V: Copy + Into<T>,
{
    limbs.iter().enumerate().fold(T::ZERO, |acc, (i, limb)| {
        acc + (*limb).into() * T::from_u64(1u64 << (i * U16_BITS))
    })
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

/// Expand `N` u16 limbs to one RV64 register bus block by zero-padding.
pub fn expand_to_rv64_block<V, T, const N: usize>(limbs: &[V; N]) -> [T; BLOCK_FE_WIDTH]
where
    V: Clone + Into<T>,
    T: PrimeCharacteristicRing,
{
    const { assert!(N <= BLOCK_FE_WIDTH) }
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
            acc + limb * T::from_u64(1u64 << (i * RV64_BYTE_BITS))
        })
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

    // SAFETY: reads raw storage bytes at VM byte pointers.
    unsafe { memory.read_bytes::<N>(address_space, ptr) }
}

#[inline(always)]
pub fn memory_read_u16<const N: usize>(
    memory: &GuestMemory,
    address_space: u32,
    ptr: u32,
) -> [u16; N] {
    debug_assert!(
        address_space == RV64_REGISTER_AS
            || address_space == RV64_MEMORY_AS
            || address_space == PUBLIC_VALUES_AS,
    );

    // SAFETY: these address spaces are u16-celled and `ptr` is an AS-native cell pointer.
    unsafe { memory.read::<u16, N>(address_space, ptr) }
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

    // SAFETY: writes raw storage bytes at VM byte pointers.
    unsafe { memory.write_bytes::<N>(address_space, ptr, data) }
}

/// Timestamped raw-byte read at VM byte pointer `ptr`.
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

    // SAFETY: reads raw storage bytes at VM byte pointers.
    unsafe { memory.read_bytes::<N>(address_space, ptr) }
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

    // SAFETY: writes raw storage bytes at VM byte pointers.
    unsafe { memory.write_bytes::<N>(address_space, ptr, data) }
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

/// Timestamped u16-cell read at an AS-native pointer.
#[inline(always)]
pub fn timed_read_u16<const N: usize>(
    memory: &mut TracingMemory,
    address_space: u32,
    ptr: u32,
) -> (u32, [u16; N]) {
    debug_assert!(
        address_space == RV64_REGISTER_AS
            || address_space == RV64_MEMORY_AS
            || address_space == PUBLIC_VALUES_AS
    );

    // SAFETY: these address spaces are u16-celled.
    unsafe { memory.read::<u16, N>(address_space, ptr) }
}

/// u16-typed counterpart to [`tracing_read`].
#[inline(always)]
pub fn tracing_read_u16<const N: usize>(
    memory: &mut TracingMemory,
    address_space: u32,
    ptr: u32,
    prev_timestamp: &mut u32,
) -> [u16; N] {
    let (t_prev, data) = timed_read_u16(memory, address_space, ptr);
    *prev_timestamp = t_prev;
    data
}

/// Timestamped u16-cell write at an AS-native pointer.
#[inline(always)]
pub fn timed_write_u16<const N: usize>(
    memory: &mut TracingMemory,
    address_space: u32,
    ptr: u32,
    data: [u16; N],
) -> (u32, [u16; N]) {
    debug_assert!(
        address_space == RV64_REGISTER_AS
            || address_space == RV64_MEMORY_AS
            || address_space == PUBLIC_VALUES_AS
    );
    // SAFETY: see `timed_read_u16`.
    unsafe { memory.write::<u16, N>(address_space, ptr, data) }
}

/// u16-typed counterpart to [`tracing_write`].
#[inline(always)]
pub fn tracing_write_u16<const N: usize>(
    memory: &mut TracingMemory,
    address_space: u32,
    ptr: u32,
    data: [u16; N],
    prev_timestamp: &mut u32,
    prev_data: &mut [u16; N],
) {
    let (t_prev, data_prev) = timed_write_u16(memory, address_space, ptr, data);
    *prev_timestamp = t_prev;
    *prev_data = data_prev;
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
    debug_assert!((val as u64) < (1u64 << pointer_max_bits));
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
    imm_to_rv64_bytes(imm)
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
pub fn memory_read_from_state<Ctx, const N: usize>(
    state: &mut VmStateMut<GuestMemory, Ctx>,
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
pub fn memory_write_from_state<Ctx, const N: usize>(
    state: &mut VmStateMut<GuestMemory, Ctx>,
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
pub fn read_rv64_register_from_state<Ctx>(state: &mut VmStateMut<GuestMemory, Ctx>, ptr: u32) -> u64
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
    u64_to_u32_checked(read_rv64_register(memory, ptr))
}
