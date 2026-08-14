//! Column layout for the `EC_MUL` chip.
//!
//! ```text
//!  offset 0            header_width         header_width + expr_width
//!  +------------------+--------------------+--------------------------+
//!  | EcMulHeaderCols  | FieldExpr sub-row  | EcMulIoCols              |
//!  | every row        | compute rows only  | final compute row only   |
//!  +------------------+--------------------+--------------------------+
//! ```
//!
//! The `FieldExpr` sub-row width is only known at runtime, so the I/O region's offset is computed
//! rather than expressed as a struct field. All memory and execution activity lives on the final
//! compute row, whose expression already holds the base point as inputs and the product as
//! outputs; an instruction packs into a fixed power-of-two number of rows.

use openvm_circuit::{
    arch::{ExecutionState, BLOCK_FE_WIDTH},
    system::memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols},
};
use openvm_circuit_primitives::AlignedBorrow;
use openvm_riscv_circuit::adapters::PTR_U16_LIMBS;

use super::{EC_MUL_COMPUTE_ROWS, SCALAR_ACC_LIMBS, SCALAR_BLOCKS, SCALAR_LIMBS};

/// Register reads per instruction: `rd`, `rs1` (base point pointer), `rs2` (scalar pointer).
pub const EC_MUL_REGISTER_READS: usize = 3;

/// The final compute row, which carries the instruction's I/O, sits at this `row_idx`.
pub const EC_MUL_FINAL_ROW_IDX: usize = EC_MUL_COMPUTE_ROWS - 1;

/// Columns present on every row.
#[repr(C)]
#[derive(Copy, Clone, Debug, AlignedBorrow)]
pub struct EcMulHeaderCols<T> {
    /// 1 on ladder rows. Doubles as the `FieldExpr`'s `is_valid`.
    pub is_compute: T,
    /// 1 on the final compute row, where every memory and execution interaction fires.
    pub is_final: T,
    /// 1 on the first compute row of an instruction.
    pub is_first_compute: T,
    /// 1 for `SETUP_EC_MUL`, constant across the instruction's rows.
    pub is_setup: T,
    /// `is_compute AND NOT is_setup AND NOT is_first_compute`, stored so the data links can be
    /// gated at degree 1.
    pub is_ladder: T,
    /// `is_final AND NOT is_setup`, stored so the scalar binding can be gated at degree 1.
    pub is_real_final: T,
    /// Position within the instruction, `0..EC_MUL_COMPUTE_ROWS`.
    pub row_idx: T,
    /// The bit accumulator entering the row, MSB-first, in limbs of one row's digits so the
    /// recurrence is a pure shift.
    pub scalar_acc: [T; SCALAR_ACC_LIMBS],
}

/// Columns used only on the final compute row, which carries all of the instruction's memory I/O.
/// The base point and the result have no stored copies: the memory bridge reads the final row's
/// `FieldExpr` inputs and writes its outputs directly.
///
/// `NUM_LIMBS` is the coordinate width in 8-bit limbs; `BLOCKS` is the number of
/// `MEMORY_BLOCK_BYTES` blocks spanned by one point.
#[repr(C)]
#[derive(Copy, Clone, Debug, AlignedBorrow)]
pub struct EcMulIoCols<T, const NUM_LIMBS: usize, const BLOCKS: usize> {
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

    pub point_read_aux: [MemoryReadAuxCols<T>; BLOCKS],

    /// Scalar bytes read from rs2, checked against `2B + 1` for the completed accumulator.
    pub scalar_data: [T; SCALAR_LIMBS],
    pub scalar_read_aux: [MemoryReadAuxCols<T>; SCALAR_BLOCKS],
    /// Carries for that check, one per byte. Boolean, since `2*B[i] + c <= 511`.
    pub scalar_carry: [T; SCALAR_LIMBS],

    pub write_aux: [MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>; BLOCKS],
}

/// Width of the header, i.e. the offset of the expression sub-row.
pub const fn ec_mul_header_width() -> usize {
    EcMulHeaderCols::<u8>::width()
}

/// Width of the I/O region for a given coordinate size.
pub const fn ec_mul_io_width<const NUM_LIMBS: usize, const BLOCKS: usize>() -> usize {
    EcMulIoCols::<u8, NUM_LIMBS, BLOCKS>::width()
}

/// Offset of the I/O region within a row.
pub fn ec_mul_io_offset(expr_width: usize) -> usize {
    ec_mul_header_width() + expr_width
}

/// Total row width: header, then the runtime-sized `FieldExpr` sub-row, then the I/O region.
pub fn ec_mul_width<const NUM_LIMBS: usize, const BLOCKS: usize>(expr_width: usize) -> usize {
    ec_mul_io_offset(expr_width) + ec_mul_io_width::<NUM_LIMBS, BLOCKS>()
}
