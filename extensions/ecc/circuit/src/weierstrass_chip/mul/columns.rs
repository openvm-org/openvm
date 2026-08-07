//! Column layout for the `EC_MUL` chip.
//!
//! ```text
//!  offset 0            header_width         header_width + expr_width
//!  +------------------+--------------------+--------------------------+
//!  | EcMulHeaderCols  | FieldExpr sub-row  | EcMulDigestCols          |
//!  | every row        | compute rows only  | digest row only          |
//!  +------------------+--------------------+--------------------------+
//! ```
//!
//! The `FieldExpr` sub-row width is known only at runtime, since it depends on the curve's
//! `num_limbs`, so the digest region's offset is computed rather than expressed as a struct field.
//! The header and digest regions are fixed-size and are `AlignedBorrow`ed out of the row slice.
//!
//! The three regions do not overlap, and the digest region cannot be overlaid onto the expression
//! region to reclaim its width. Several `FieldExpr` constraints are not gated by `is_valid`:
//! `check_carry_to_zero`'s carry recurrence, its final zero-carry assertion, and `assert_bool` on
//! each flag. An inactive expression region must therefore still hold a consistent witness for some
//! input, which arbitrary digest data is not. Trace generation instead fills it on digest and
//! padding rows with a witness for zero inputs and clears `is_valid`.

use openvm_circuit::{
    arch::{ExecutionState, BLOCK_FE_WIDTH},
    system::memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols},
};
use openvm_circuit_primitives::AlignedBorrow;
use openvm_riscv_circuit::adapters::PTR_U16_LIMBS;

use super::{EC_MUL_COMPUTE_ROWS, SCALAR_ACC_LIMBS, SCALAR_BLOCKS, SCALAR_LIMBS};

/// Register reads per instruction: `rd`, `rs1` (base point pointer), `rs2` (scalar pointer).
pub const EC_MUL_REGISTER_READS: usize = 3;

/// The digest row sits at this `row_idx`.
pub const EC_MUL_DIGEST_ROW_IDX: usize = EC_MUL_COMPUTE_ROWS;

/// Columns present on every row.
///
/// `is_compute` doubles as the `FieldExpr`'s `is_valid`: the AIR constrains
/// `row[header_width] == is_compute`, switching the expression off on digest and padding rows.
#[repr(C)]
#[derive(Copy, Clone, Debug, AlignedBorrow)]
pub struct EcMulHeaderCols<T> {
    /// 1 on ladder rows (`row_idx ∈ [0, EC_MUL_COMPUTE_ROWS)`).
    pub is_compute: T,
    /// 1 on the digest row (`row_idx == EC_MUL_COMPUTE_ROWS`).
    pub is_digest: T,
    /// 1 on the first compute row of an instruction. Derived from `row_idx`, but stored so that
    /// the initial-accumulator constraint does not need an is-zero gadget.
    pub is_first_compute: T,
    /// 1 for `SETUP_EC_MUL`, constant across the instruction's rows.
    ///
    /// Mirrors the value `FieldExpr` derives as `is_valid - sum(flags)`. Stored as a column
    /// because the digest row has no expression sub-row to derive it from and needs it to
    /// decide whether to check the scalar.
    pub is_setup: T,
    /// 1 on a ladder row that continues a non-setup instruction, i.e.
    /// `is_compute AND NOT is_setup AND NOT is_first_compute`.
    ///
    /// Derived, but stored so the accumulator and base-point links can be gated by a degree-1
    /// selector. Computing the conjunction inline would put those constraints at degree 4, above
    /// the budget every other AIR in the configuration meets.
    pub is_ladder: T,
    /// 1 on the digest row of a non-setup instruction, i.e. `is_digest AND NOT is_setup`. Stored
    /// for the same reason as [`EcMulHeaderCols::is_ladder`]: it gates the handoff links at
    /// degree 1.
    pub is_real_digest: T,
    /// Position within the instruction, `0..=EC_MUL_COMPUTE_ROWS`.
    ///
    /// A plain incrementing counter rather than an `Encoder`: the increment is a degree-3
    /// transition constraint in one column, where an `Encoder` over this many rows would raise the
    /// AIR's maximum constraint degree and with it the application's `log_blowup`.
    pub row_idx: T,
    /// The bit accumulator `B`, MSB-first, in limbs of [`EC_MUL_STEPS_PER_ROW`] bits:
    /// `B' = 2^EC_MUL_STEPS_PER_ROW * B + digits`.
    ///
    /// Holds the value entering the row, so it is zero on the first compute row and complete on
    /// the digest row, where `2B + 1` is checked against the scalar. That check is what binds
    /// the rows' sign flags to the operand.
    ///
    /// The limb size makes the recurrence a shift, so there is nothing to carry and nothing to
    /// range check. `B'[0]` holds this row's digits, already in range as a degree-1 form in
    /// the flags, and every other limb copies a neighbour.
    pub scalar_acc: [T; SCALAR_ACC_LIMBS],
}

/// Columns used only on the digest row, which carries all of the instruction's memory I/O.
///
/// `NUM_LIMBS` is the coordinate width in 8-bit limbs; `BLOCKS` is the number of
/// `MEMORY_BLOCK_BYTES` blocks spanned by one point (both coordinates together).
#[repr(C)]
#[derive(Copy, Clone, Debug, AlignedBorrow)]
pub struct EcMulDigestCols<T, const NUM_LIMBS: usize, const BLOCKS: usize> {
    pub from_state: ExecutionState<T>,

    /// Register operand addresses.
    pub rd_ptr: T,
    pub rs1_ptr: T,
    pub rs2_ptr: T,

    /// Heap pointers materialised from those registers, as u16 cells.
    pub rd_val: [T; PTR_U16_LIMBS],
    pub rs1_val: [T; PTR_U16_LIMBS],
    pub rs2_val: [T; PTR_U16_LIMBS],

    pub rs_read_aux: [MemoryReadAuxCols<T>; EC_MUL_REGISTER_READS],

    /// Base point `P`, read from `rs1`, linked to the compute rows' `FieldExpr` inputs across the
    /// compute-to-digest transition. `P` is constrained constant across compute rows, so the
    /// single link propagates to all of them.
    pub point_x: [T; NUM_LIMBS],
    pub point_y: [T; NUM_LIMBS],
    pub point_read_aux: [MemoryReadAuxCols<T>; BLOCKS],

    /// Scalar bytes read from `rs2`, checked against `2B + 1` for the accumulated
    /// [`EcMulHeaderCols::scalar_acc`].
    pub scalar_data: [T; SCALAR_LIMBS],
    pub scalar_read_aux: [MemoryReadAuxCols<T>; SCALAR_BLOCKS],
    /// Carries for that check, one per byte. Boolean, since `2*B[i] + c <= 511`.
    ///
    /// The top carry is constrained to zero, which pins `2B + 1 < 2^256` and so the scalar's
    /// width. The tighter bound below the curve order stays a caller precondition.
    pub scalar_carry: [T; SCALAR_LIMBS],

    /// Result `k·P`, written to `rd`. Linked to the last compute row's outputs.
    pub result_x: [T; NUM_LIMBS],
    pub result_y: [T; NUM_LIMBS],
    pub write_aux: [MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>; BLOCKS],
}

/// Width of [`EcMulHeaderCols`], i.e. the offset of the `FieldExpr` sub-row.
pub const fn ec_mul_header_width() -> usize {
    EcMulHeaderCols::<u8>::width()
}

/// Width of [`EcMulDigestCols`] for a given coordinate size.
pub const fn ec_mul_digest_width<const NUM_LIMBS: usize, const BLOCKS: usize>() -> usize {
    EcMulDigestCols::<u8, NUM_LIMBS, BLOCKS>::width()
}

/// Offset of the digest region within a row.
pub fn ec_mul_digest_offset(expr_width: usize) -> usize {
    ec_mul_header_width() + expr_width
}

/// Total row width: header, then the runtime-sized `FieldExpr` sub-row, then the digest region.
pub fn ec_mul_width<const NUM_LIMBS: usize, const BLOCKS: usize>(expr_width: usize) -> usize {
    ec_mul_digest_offset(expr_width) + ec_mul_digest_width::<NUM_LIMBS, BLOCKS>()
}
