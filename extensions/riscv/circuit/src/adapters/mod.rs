use std::ops::Mul;

use openvm_circuit::{
    arch::{
        execution_mode::ExecutionCtxTrait, ExecutionError, PostflightError, VmStateMut,
        BLOCK_FE_WIDTH, DEFAULT_RV64_MEMORY_BYTE_CAPACITY, MEMORY_BLOCK_BYTES, U16_CELL_SIZE_BITS,
    },
    system::memory::online::GuestMemory,
};
pub use openvm_circuit_primitives::U16_BITS;
use openvm_circuit_primitives::{
    encoder::Encoder,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
};
use openvm_instructions::{
    instruction::InstructionOperand,
    riscv::{MEMORY_AS, REGISTER_AS},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::AirBuilder,
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
pub use openvm_instructions::riscv::{BYTE_BITS, REGISTER_NUM_LIMBS, WORD_NUM_LIMBS};
pub use rdwrite::*;
pub use store::*;

/// Number of u16 limbs needed for a low-32-bit RV64 pointer.
pub const PTR_U16_LIMBS: usize = WORD_NUM_LIMBS / 2;
/// Bit width covered by [`PTR_U16_LIMBS`].
pub const PTR_BITS: usize = U16_BITS * PTR_U16_LIMBS;
/// Number of u16 limbs in a 32-bit RV64 word (e.g. an `ADDW`/`SUBW` operand, or one half of a
/// register). Numerically equal to [`PTR_U16_LIMBS`], but named for arithmetic-word use.
pub const WORD_U16_LIMBS: usize = WORD_NUM_LIMBS / 2;

#[inline(always)]
pub(crate) fn checked_register_pointer(pointer: u32) -> Result<u8, PostflightError> {
    if pointer > u8::MAX as u32 || !pointer.is_multiple_of(REGISTER_NUM_LIMBS as u32) {
        return Err(PostflightError::new(
            "RV64 register pointer is outside the register domain",
        ));
    }
    Ok(pointer as u8)
}

#[inline(always)]
pub(crate) fn checked_register_u16_pointer(pointer: u32) -> Result<u32, PostflightError> {
    checked_register_pointer(pointer)?;
    Ok(pointer >> 1)
}

pub(crate) struct ReplayComputation<const NUM_LIMBS: usize, M> {
    pub output: [u8; NUM_LIMBS],
    pub metadata: M,
}

pub(crate) struct ReplayResult<const NUM_LIMBS: usize, M> {
    pub inputs: [[u8; NUM_LIMBS]; 2],
    pub output: [u8; NUM_LIMBS],
    pub metadata: M,
}

/// Validate a guest byte pointer used as the start of a memory-bus block.
///
/// OpenVM memory is an equipartition into [`BLOCK_FE_WIDTH`]-cell blocks. In
/// the RV64 u16 address spaces that makes every proof-visible block start
/// [`MEMORY_BLOCK_BYTES`]-byte aligned.
#[inline(always)]
pub fn validate_memory_block_byte_ptr(pc: u32, ptr: u32) -> Result<u32, ExecutionError> {
    if !ptr.is_multiple_of(MEMORY_BLOCK_BYTES as u32) {
        return Err(ExecutionError::Fail {
            pc,
            msg: "memory block pointer must be eight-byte aligned",
        });
    }
    Ok(ptr)
}

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

/// Encodes one selector case for each byte shift, reserving the zero point for invalid rows.
pub(crate) fn shift_encoder() -> Encoder {
    let encoder = Encoder::new(NUM_BYTE_SHIFTS, SHIFT_SELECTOR_MAX_DEGREE, true);
    assert_eq!(encoder.width(), BYTE_SHIFT_SELECTOR_WIDTH);
    encoder
}

/// Packs two little-endian u8 limbs into one u16-shaped field element.
#[inline(always)]
pub fn pack_u8_pair<T: PrimeCharacteristicRing>(lo: T, hi: T) -> T {
    lo + hi * T::from_u32(1 << BYTE_BITS)
}

#[inline(always)]
pub fn pack_u8_pair_u32<T: PrimeCharacteristicRing>(lo: u32, hi: u32) -> T {
    pack_u8_pair(T::from_u32(lo), T::from_u32(hi))
}

#[inline(always)]
pub fn pack_u16_block<L, H, T>(
    low_word: &[L; WORD_NUM_LIMBS],
    high: &[H; PTR_U16_LIMBS],
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
/// Unlike [`pack_u16_block`], the low word is already u16-celled, so no byte packing occurs.
#[inline(always)]
pub fn concat_u16_block<L, H, T>(
    low_word: &[L; WORD_U16_LIMBS],
    high: &[H; WORD_U16_LIMBS],
) -> [T; BLOCK_FE_WIDTH]
where
    L: Clone + Into<T>,
    H: Clone + Into<T>,
    T: PrimeCharacteristicRing,
{
    std::array::from_fn(|i| {
        if i < WORD_U16_LIMBS {
            low_word[i].clone().into()
        } else {
            high[i - WORD_U16_LIMBS].clone().into()
        }
    })
}

#[inline(always)]
pub(crate) fn pack_high_u16<T, B>(
    bytes: &[B; REGISTER_NUM_LIMBS - WORD_NUM_LIMBS],
) -> [T; PTR_U16_LIMBS]
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

/// Decodes a signed instruction immediate and checks that it fits the requested
/// RISC-V immediate width.
#[inline]
pub fn decode_signed_instruction_imm(operand: InstructionOperand, bits: usize) -> Option<i32> {
    let value = operand.as_i32();
    let shift = u32::try_from(bits.checked_sub(1)?).ok()?;
    let bound = 1i32.checked_shl(shift)?;
    (-bound..bound).contains(&value).then_some(value)
}

/// Composes an RV64 register byte-limb array into a `u64`.
pub fn limbs_to_u64<F: PrimeField32>(limbs: [F; REGISTER_NUM_LIMBS]) -> u64 {
    let mut val: u64 = 0;
    for (i, limb) in limbs.map(|x| x.as_canonical_u32()).iter().enumerate() {
        val += (*limb as u64) << (i * BYTE_BITS);
    }
    val
}

/// Decomposes a `u64` into RV64 register byte limbs.
pub fn u64_to_limbs<F: PrimeField32>(value: u64) -> [F; REGISTER_NUM_LIMBS] {
    std::array::from_fn(|i| {
        F::from_u32(((value >> (BYTE_BITS * i)) & ((1 << BYTE_BITS) - 1)) as u32)
    })
}

/// Converts a 24-bit instruction immediate to sign-extended RV64 register bytes.
/// The immediate is a 12-bit signed value encoded into 24 bits with byte 2
/// carrying the sign.
#[inline(always)]
pub fn imm_to_bytes(imm: u32) -> [u8; REGISTER_NUM_LIMBS] {
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
pub fn imm_to_u64(imm: u32) -> u64 {
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

#[inline(always)]
pub(crate) fn checked_byte_ptr_to_u16_ptr_value(
    byte_ptr: u32,
) -> Result<u32, openvm_circuit::arch::PostflightError> {
    if byte_ptr & 1 != 0 {
        return Err(openvm_circuit::arch::PostflightError::new(
            "u16 memory-bus pointer must be two-byte aligned",
        ));
    }
    Ok(byte_ptr >> 1)
}

/// Converts a `u64` to `u32`, requiring the upper 32 bits to be zero.
#[inline(always)]
pub fn u64_to_u32_checked(value: u64) -> u32 {
    u32::try_from(value).expect("upper 4 bytes must be zero")
}

/// Converts RV64 register bytes to a `u32`, requiring the upper 4 bytes to be zero.
#[inline(always)]
pub fn bytes_to_u32(bytes: [u8; REGISTER_NUM_LIMBS]) -> u32 {
    u64_to_u32_checked(u64::from_le_bytes(bytes))
}

/// Attempts to convert RV64 register bytes to a `u32`, requiring the upper 4 bytes to be zero.
#[inline(always)]
pub fn try_bytes_to_u32(bytes: [u8; REGISTER_NUM_LIMBS]) -> Option<u32> {
    u32::try_from(u64::from_le_bytes(bytes)).ok()
}

/// Adds an already-sign-extended 16-bit RV64 immediate to an implemented low-32-bit address.
#[inline(always)]
pub fn address_add_imm(base: u32, imm_extended: u32) -> u64 {
    u64::from(base).wrapping_add(sext32_to_u64(imm_extended))
}

#[inline(always)]
pub(crate) fn checked_memory_address(
    pc: u32,
    base: u32,
    imm_extended: u32,
    access_width: usize,
) -> Result<u32, ExecutionError> {
    // Bound by the *circuit* memory capacity (2^32 bytes), not the guest platform's
    // `MEM_SIZE`: the RV64 memory address space supports the full 2^32-byte range.
    // TODO: use `MemoryConfig::pointer_max_bits` once execution state carries the memory config.
    debug_assert!(access_width <= DEFAULT_RV64_MEMORY_BYTE_CAPACITY);
    let address = address_add_imm(base, imm_extended);
    if address > (DEFAULT_RV64_MEMORY_BYTE_CAPACITY - access_width) as u64 {
        return Err(ExecutionError::Fail {
            pc,
            msg: "effective address exceeds implemented memory address space",
        });
    }
    Ok(address as u32)
}

#[inline(always)]
pub fn bytes_to_u16_block(bytes: [u8; REGISTER_NUM_LIMBS]) -> [u16; BLOCK_FE_WIDTH] {
    std::array::from_fn(|i| u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]))
}

pub(crate) const BYTE_SIGN_BIT: u16 = 1 << (BYTE_BITS - 1);
pub(crate) const U16_SIGN_BIT: u16 = 1 << (U16_BITS - 1);

#[inline(always)]
pub(crate) fn u16_cell_byte(cell: u16, byte_idx: usize) -> u16 {
    u16::from(cell.to_le_bytes()[byte_idx])
}

/// Converts a low-32-bit value to one zero-extended RV64 u16 block.
#[inline(always)]
pub fn u32_to_u16_block(value: u32) -> [u16; BLOCK_FE_WIDTH] {
    std::array::from_fn(|i| {
        if i < PTR_U16_LIMBS {
            (value >> (U16_BITS * i)) as u16
        } else {
            0
        }
    })
}

/// Splits a 32-bit RV64 pointer into low-to-high u16 limbs.
#[inline(always)]
pub fn ptr_to_u16_limbs(ptr: u32) -> [u16; PTR_U16_LIMBS] {
    std::array::from_fn(|i| (ptr >> (U16_BITS * i)) as u16)
}

/// Field-element form of [`ptr_to_u16_limbs`].
#[inline(always)]
pub fn ptr_to_field_u16_limbs<F: PrimeCharacteristicRing>(value: u32) -> [F; PTR_U16_LIMBS] {
    ptr_to_u16_limbs(value).map(F::from_u16)
}

#[inline(always)]
pub fn u16_block_to_bytes(block: [u16; BLOCK_FE_WIDTH]) -> [u8; REGISTER_NUM_LIMBS] {
    let mut out = [0u8; REGISTER_NUM_LIMBS];
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
        (U16_BITS..=PTR_BITS).contains(&ptr_max_bits),
        "ptr_max_bits must be in [U16_BITS, PTR_BITS]"
    );
    PTR_BITS - ptr_max_bits
}

/// Range-check value for a high u16 pointer limb.
#[inline(always)]
pub fn ptr_bound_from_high_u16(high_u16: u16, ptr_max_bits: usize) -> u32 {
    u32::from(high_u16) << ptr_max_bits_shift(ptr_max_bits)
}

/// Range-check value for the high u16 limb of a low-32-bit pointer.
#[inline(always)]
pub fn ptr_bound_from_ptr(ptr: u32, ptr_max_bits: usize) -> u32 {
    let high_u16 = ptr_to_u16_limbs(ptr)[PTR_U16_LIMBS - 1];
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
pub fn u16_limbs_to_ptr<T, V>(limbs: &[V; PTR_U16_LIMBS]) -> T
where
    T: PrimeCharacteristicRing,
    V: Copy + Into<T>,
{
    limbs.iter().enumerate().fold(T::ZERO, |acc, (i, limb)| {
        acc + (*limb).into() * T::from_u64(1u64 << (i * U16_BITS))
    })
}

// ----------------------------------------------------------------------------
// AS-native pointer-limb helpers.
//
// Every memory-bus pointer is two little-endian 16-bit *AS-native cell* pointer limbs
// `[lo16, hi16]` (see `openvm_circuit::system::memory::MemoryAddress`). These helpers convert
// between RV64 *byte* pointers (read from registers) and AS-native *cell* pointer limbs without
// composing a full (up to 31-bit) pointer into one field element.
// ----------------------------------------------------------------------------

/// AS-native memory pointer represented as little-endian 16-bit limbs `[lo16, hi16]`.
pub type PtrLimbs<T> = [T; 2];

/// Splits a concrete pointer into little-endian 16-bit limb *values* `[lo16, hi16]`.
#[inline(always)]
pub fn u32_to_ptr_limbs(ptr: u32) -> PtrLimbs<u32> {
    [ptr & 0xffff, ptr >> U16_BITS]
}

/// Recomposes little-endian 16-bit limb values into a `u32`.
#[inline(always)]
pub fn ptr_limbs_to_u32(limbs: PtrLimbs<u32>) -> u32 {
    limbs[0] | (limbs[1] << U16_BITS)
}

/// AS-native cell-pointer limbs for a byte pointer in the register address space
/// ([`REGISTER_AS`]).
///
/// The register file holds at most `NUM_REGISTERS * 8` bytes, so a register byte pointer's
/// cell pointer `ptr / 2` is far below `2^16`: it fits entirely in the low 16-bit limb and the
/// high limb is always zero. This lets us skip the carry/decomposition columns and range checks
/// that a general (up to `DEFAULT_POINTER_MAX_BITS`-bit) memory pointer requires. Only use this for
/// register-AS pointers; for the memory address space use the range-checked decomposition helpers.
#[inline(always)]
pub fn reg_byte_ptr_to_cell_ptr_limbs<AB: InteractionBuilder>(
    byte_ptr: impl Into<AB::Expr>,
) -> PtrLimbs<AB::Expr> {
    [byte_ptr_to_u16_ptr::<AB>(byte_ptr), AB::Expr::ZERO]
}

/// Value form of [`reg_byte_ptr_to_cell_ptr_limbs`].
#[inline(always)]
pub fn reg_byte_ptr_to_cell_ptr_limbs_value(byte_ptr: u32) -> PtrLimbs<u32> {
    [byte_ptr_to_u16_ptr_value(byte_ptr), 0]
}

/// Converts an aligned RV64 byte pointer given as little-endian 16-bit limbs `[byte_lo, byte_hi]`
/// into AS-native u16 *cell* pointer limbs `[cell_lo, cell_hi]` (cell = byte / 2).
///
/// `carry` is a witness boolean intended to equal `byte_hi & 1`. The returned limbs are the
/// expressions
///   cell_lo = (byte_lo + carry * 2^16) / 2,   cell_hi = (byte_hi - carry) / 2.
/// The composed cell pointer `cell_lo + 2^16 * cell_hi` equals `byte_ptr / 2` as a field identity
/// for *any* `carry`. The caller must already constrain `byte_lo` to be a canonical 16-bit value
/// divisible by 8; this makes `cell_lo < 2^16` for either boolean carry. This function range-checks
/// only `cell_hi < 2^cell_hi_bits`, where the high-limb bound is derived from `byte_ptr_max_bits`
/// (the guest *byte* pointer width): a u16 cell is two bytes, so
/// `cell_max_bits = byte_ptr_max_bits - U16_CELL_SIZE_BITS` and
/// `cell_hi_bits = cell_max_bits - U16_BITS`. Since `cell_hi` is a bounded integer expression, this
/// also forces `carry = byte_hi & 1`.
#[allow(clippy::too_many_arguments)]
pub fn eval_byte_ptr_limbs_to_u16_cell_ptr_limbs<AB: InteractionBuilder>(
    builder: &mut AB,
    range_bus: VariableRangeCheckerBus,
    byte_limbs: [AB::Expr; 2],
    carry: impl Into<AB::Expr>,
    byte_ptr_max_bits: usize,
    enabled: AB::Expr,
) -> PtrLimbs<AB::Expr> {
    let cell_hi_bits = byte_ptr_max_bits - U16_CELL_SIZE_BITS - U16_BITS;
    let carry_e: AB::Expr = carry.into();
    builder.when(enabled.clone()).assert_bool(carry_e.clone());
    let inv2 = AB::F::TWO.inverse();
    let [byte_lo, byte_hi] = byte_limbs;
    let cell_lo = (byte_lo + carry_e.clone() * AB::F::from_u32(1 << U16_BITS)) * inv2;
    let cell_hi = (byte_hi - carry_e) * inv2;
    range_bus
        .range_check(cell_hi.clone(), cell_hi_bits)
        .eval(builder, enabled);
    [cell_lo, cell_hi]
}

/// Cell high-limb range-check bit width corresponding to a guest `byte_ptr_max_bits`.
#[inline(always)]
pub fn cell_ptr_hi_bits(byte_ptr_max_bits: usize) -> usize {
    byte_ptr_max_bits - U16_CELL_SIZE_BITS - U16_BITS
}

/// Adds a small constant `constant` (`< 2^16`) to a pointer given as little-endian 16-bit limbs
/// `[lo, hi]`, carrying into the high limb:
///   new_lo = lo + constant - carry * 2^16,   new_hi = hi + carry.
/// `carry` is a witness boolean. Only `new_lo` is range-checked (to 16 bits): this forces `carry`
/// to be the correct carry bit (given `lo` canonical), so `new_hi = hi + carry` is canonical
/// whenever `hi` is. Use to add a per-block cell offset to an already-converted base cell pointer.
#[allow(clippy::too_many_arguments)]
pub fn eval_add_const_u16_limbs<AB: InteractionBuilder>(
    builder: &mut AB,
    range_bus: VariableRangeCheckerBus,
    limbs: [AB::Expr; 2],
    constant: u32,
    carry: AB::Var,
    enabled: AB::Expr,
) -> PtrLimbs<AB::Expr> {
    let carry_e: AB::Expr = carry.into();
    builder.when(enabled.clone()).assert_bool(carry_e.clone());
    let [lo, hi] = limbs;
    let new_lo =
        lo + AB::Expr::from_u32(constant) - carry_e.clone() * AB::F::from_u32(1 << U16_BITS);
    let new_hi = hi + carry_e;
    range_bus
        .range_check(new_lo.clone(), U16_BITS)
        .eval(builder, enabled);
    [new_lo, new_hi]
}

/// Value form of [`eval_add_const_u16_limbs`]: returns `(carry, [new_lo, new_hi])`.
#[inline(always)]
pub fn add_const_u16_limbs_value(limbs: PtrLimbs<u32>, constant: u32) -> (u32, PtrLimbs<u32>) {
    let sum_lo = limbs[0] + constant;
    let carry = sum_lo >> U16_BITS;
    (carry, [sum_lo & 0xffff, limbs[1] + carry])
}

/// Computes one add-carry per memory block from an already-converted base cell pointer,
/// registering the matching range checks for each block's new low limb.
pub fn compute_block_add_carries(
    range_checker: &SharedVariableRangeCheckerChip,
    base_cell: [u16; 2],
    num_blocks: usize,
    cell_stride: u32,
) -> Vec<u32> {
    let base_cell = base_cell.map(u32::from);
    (0..num_blocks)
        .map(|i| {
            let (add_carry, block_cell_ptr) =
                add_const_u16_limbs_value(base_cell, i as u32 * cell_stride);
            range_checker.add_count(block_cell_ptr[0], U16_BITS);
            add_carry
        })
        .collect()
}

/// Value form of [`eval_byte_ptr_limbs_to_u16_cell_ptr_limbs`]. Returns
/// `(carry, [cell_lo, cell_hi])` for an aligned byte pointer given as little-endian 16-bit limb
/// values. The caller is responsible for registering the matching range-check for `cell_hi`
/// to `hi_bits`.
#[inline(always)]
pub fn byte_ptr_limbs_to_cell_ptr_limbs_value(byte_limbs: PtrLimbs<u32>) -> (u32, PtrLimbs<u32>) {
    let carry = byte_limbs[1] & 1;
    let cell_lo = (byte_limbs[0] + (carry << U16_BITS)) >> 1;
    let cell_hi = byte_limbs[1] >> 1;
    (carry, [cell_lo, cell_hi])
}

/// Computes the byte->cell conversion carry and one add-carry per block for a heap
/// access group, registering the matching range checks.
///
/// Returns `(conv_carry, add_carries)`.
///
/// Column writes are left to the caller because vec_heap-family fillers must buffer
/// carries before overwriting their records.
pub fn compute_pointer_carries(
    range_checker: &SharedVariableRangeCheckerChip,
    byte_ptr: u32,
    num_blocks: usize,
    cell_stride: u32,
    byte_ptr_max_bits: usize,
) -> (u32, Vec<u32>) {
    let byte_limbs = u32_to_ptr_limbs(byte_ptr);
    let (conv_carry, base_cell) = byte_ptr_limbs_to_cell_ptr_limbs_value(byte_limbs);
    range_checker.add_count(base_cell[1], cell_ptr_hi_bits(byte_ptr_max_bits));
    let add_carries = compute_block_add_carries(
        range_checker,
        base_cell.map(|limb| limb as u16),
        num_blocks,
        cell_stride,
    );
    (conv_carry, add_carries)
}

/// Expand `N` limbs to `REGISTER_NUM_LIMBS` (8) by zero-padding the upper limbs. Used for
/// register bus reads where the register holds a value in fewer than 8 bytes.
pub fn expand_to_register<V: Clone + Into<T>, T: PrimeCharacteristicRing, const N: usize>(
    limbs: &[V; N],
) -> [T; REGISTER_NUM_LIMBS] {
    const { assert!(N <= REGISTER_NUM_LIMBS) }
    std::array::from_fn(|i| {
        if i < N {
            limbs[i].clone().into()
        } else {
            T::ZERO
        }
    })
}

/// Expand `N` u16 limbs to one RV64 register bus block by zero-padding.
pub fn expand_to_block<V, T, const N: usize>(limbs: &[V; N]) -> [T; BLOCK_FE_WIDTH]
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
            acc + limb * T::from_u64(1u64 << (i * BYTE_BITS))
        })
}

#[inline(always)]
pub fn memory_read<const N: usize>(memory: &GuestMemory, address_space: u32, ptr: u32) -> [u8; N] {
    debug_assert!(address_space == REGISTER_AS || address_space == MEMORY_AS,);

    // SAFETY: reads raw storage bytes at VM byte pointers.
    unsafe { memory.read_bytes::<N>(address_space, ptr) }
}

#[inline(always)]
pub fn memory_write<const N: usize>(
    memory: &mut GuestMemory,
    address_space: u32,
    ptr: u32,
    data: [u8; N],
) {
    debug_assert!(address_space == REGISTER_AS || address_space == MEMORY_AS);

    // SAFETY: writes raw storage bytes at VM byte pointers.
    unsafe { memory.write_bytes::<N>(address_space, ptr, data) }
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
pub fn read_register_from_state<Ctx>(state: &mut VmStateMut<GuestMemory, Ctx>, ptr: u32) -> u64
where
    Ctx: ExecutionCtxTrait,
{
    u64::from_le_bytes(memory_read_from_state(state, REGISTER_AS, ptr))
}

#[inline(always)]
pub fn read_register(memory: &GuestMemory, ptr: u32) -> u64 {
    u64::from_le_bytes(memory_read(memory, REGISTER_AS, ptr))
}

/// Read an RV64 register and return its value as u32, asserting (in debug) that the upper
/// 32 bits are zero.
#[inline(always)]
pub fn read_register_as_u32(memory: &GuestMemory, ptr: u32) -> u32 {
    u64_to_u32_checked(read_register(memory, ptr))
}

#[cfg(test)]
mod tests {
    use openvm_instructions::instruction::InstructionOperand;

    use super::{
        checked_register_u16_pointer, decode_signed_instruction_imm,
        validate_memory_block_byte_ptr, RV_B_TYPE_IMM_BITS,
    };

    #[test]
    fn signed_branch_immediate_rejects_out_of_range_values() {
        let bound = 1i32 << (RV_B_TYPE_IMM_BITS - 1);
        assert_eq!(
            decode_signed_instruction_imm(InstructionOperand::from_i32(-bound), RV_B_TYPE_IMM_BITS,),
            Some(-bound)
        );
        assert_eq!(
            decode_signed_instruction_imm(
                InstructionOperand::from_i32(bound - 1),
                RV_B_TYPE_IMM_BITS,
            ),
            Some(bound - 1)
        );
        assert_eq!(
            decode_signed_instruction_imm(InstructionOperand::from_i32(bound), RV_B_TYPE_IMM_BITS,),
            None
        );
        assert_eq!(
            decode_signed_instruction_imm(
                InstructionOperand::from_i32(InstructionOperand::MAX),
                RV_B_TYPE_IMM_BITS,
            ),
            None
        );
    }

    #[test]
    fn memory_block_pointer_uses_the_eight_byte_equipartition() {
        for pointer in [0, 8] {
            assert_eq!(
                validate_memory_block_byte_ptr(12, pointer).unwrap(),
                pointer
            );
        }
        for pointer in [2, 4, 6] {
            let error = validate_memory_block_byte_ptr(12, pointer).unwrap_err();
            assert!(error.to_string().contains("eight-byte aligned"), "{error}");
        }
    }

    #[test]
    fn register_pointer_uses_the_register_domain() {
        assert_eq!(checked_register_u16_pointer(0).unwrap(), 0);
        assert_eq!(checked_register_u16_pointer(31 * 8).unwrap(), 31 * 4);

        for pointer in [2, 32 * 8] {
            let error = checked_register_u16_pointer(pointer).unwrap_err();
            assert!(
                error.to_string().contains("outside the register domain"),
                "{error}"
            );
        }
    }
}
