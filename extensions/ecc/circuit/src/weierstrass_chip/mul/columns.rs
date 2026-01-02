//! Column structures for the EC_MUL multirow chip
//!
//! ## Trace Layout Overview
//!
//! The EC_MUL chip uses 257 rows per instruction:
//! - Rows 0-255: Compute rows (each processes one scalar bit)
//! - Row 256: Digest row (handles all memory I/O)
//!
//! ## Column Layout
//!
//! ```text
//! Compute Row:
//! [EcMulFlagsCols | EcMulControlCols | FieldExpr columns... ]
//!                                     ^
//!                                     |
//!                              offset = base_width
//!
//! Digest Row:
//! [EcMulFlagsCols | EcMulControlCols | from_state | pointers | aux | data... ]
//!                                     ^
//!                                     |
//!                              offset = base_width
//! ```
//!
//! ## KNOWN ISSUE: Column Layout Mismatch
//!
//! The AIR evaluates FieldExpr on ALL rows, reading `is_valid` from offset `base_width`.
//! For digest rows, this offset contains `from_state.pc` (non-zero), causing:
//! - `assert_bool(is_valid)` to fail
//! - Wrong interaction counts
//!
//! **FIX**: Add a padding field before `from_state` in `EcMulDigestCols`:
//! ```rust,ignore
//! pub struct EcMulDigestCols<...> {
//!     pub flags: ...,
//!     pub control: ...,
//!     pub field_expr_gate: T,  // <- Add this, always 0
//!     pub from_state: ...,
//!     // ...
//! }
//! ```
//! This ensures `local[base_width] = 0` for digest rows, gating FieldExpr constraints.

use openvm_circuit::{
    arch::ExecutionState,
    system::memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols},
};
use openvm_circuit_primitives::AlignedBorrow;
use openvm_instructions::riscv::RV32_REGISTER_NUM_LIMBS;

use super::{EC_MUL_REGISTER_READS, EC_MUL_SCALAR_BITS, EC_MUL_SCALAR_BYTES};

/// Compute row columns (rows 0-255)
///
/// Each compute row performs one double-and-add step using FieldExpr.
/// The FieldExpr columns are NOT stored here - they come after these base columns
/// in the trace and are accessed dynamically via slicing at offset `base_width`.
///
/// ## Memory Layout
/// ```text
/// [flags | control | FieldExpr.is_valid | FieldExpr.inputs | FieldExpr.vars | ...]
/// |<-- base_width -->|<---------------- expr_width ----------------------->|
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, AlignedBorrow)]
pub struct EcMulComputeCols<T, const NUM_LIMBS: usize> {
    pub flags: EcMulFlagsCols<T>,
    pub control: EcMulControlCols<T, NUM_LIMBS>,
    // FieldExpr columns come after these in the trace, accessed dynamically via slicing
}

/// Digest row columns (row 256)
///
/// Handles ALL memory I/O operations for the EC_MUL instruction:
/// - 3 register reads (rd, rs1, rs2)
/// - 1 scalar read (32 bytes)
/// - NUM_BLOCKS base point reads
/// - NUM_BLOCKS result writes
///
/// ## KNOWN ISSUE
/// The `from_state` field starts at offset `base_width`, which overlaps with
/// FieldExpr's `is_valid` column position. This causes FieldExpr constraints
/// to fail on digest rows because `from_state.pc` is non-zero.
///
/// **FIX**: Add `pub field_expr_gate: T` before `from_state` and never write to it.
/// It will remain 0 from trace zeroing, acting as FieldExpr's `is_valid = 0`.
#[repr(C)]
#[derive(Clone, Copy, Debug, AlignedBorrow)]
pub struct EcMulDigestCols<
    T,
    const NUM_LIMBS: usize,
    const NUM_BLOCKS: usize,
    const BLOCK_SIZE: usize,
> {
    pub flags: EcMulFlagsCols<T>,
    pub control: EcMulControlCols<T, NUM_LIMBS>,

    // TODO: Add `pub field_expr_gate: T` here to fix the column layout mismatch.
    // This field should always be 0, gating FieldExpr constraints for digest rows.

    // VM execution state (pc and timestamp)
    pub from_state: ExecutionState<T>,

    // Register pointer reads (rd, rs1, rs2)
    pub rd_ptr: T,
    pub rs1_ptr: T,
    pub rs2_ptr: T,

    // Resolved addresses (from register reads)
    pub dst_ptr: [T; RV32_REGISTER_NUM_LIMBS],
    pub scalar_ptr: [T; RV32_REGISTER_NUM_LIMBS],
    pub basepoint_ptr: [T; RV32_REGISTER_NUM_LIMBS],

    pub rs_read_aux: [MemoryReadAuxCols<T>; EC_MUL_REGISTER_READS],

    // Scalar: 1 block of 32 bytes
    pub scalar_data: [T; EC_MUL_SCALAR_BYTES],
    pub reads_scalar_aux: MemoryReadAuxCols<T>,

    // Base point: NUM_BLOCKS blocks of BLOCK_SIZE bytes each
    pub basepoint_data: [[T; BLOCK_SIZE]; NUM_BLOCKS],
    pub reads_point_aux: [MemoryReadAuxCols<T>; NUM_BLOCKS],

    // Result point: NUM_BLOCKS blocks of BLOCK_SIZE bytes each
    pub result_data: [[T; BLOCK_SIZE]; NUM_BLOCKS],
    pub writes_aux: [MemoryWriteAuxCols<T, BLOCK_SIZE>; NUM_BLOCKS],
}

/// Control columns (shared by both compute and digest rows)
/// Contains data that remains constant after initialization
#[repr(C)]
#[derive(Clone, Copy, Debug, AlignedBorrow)]
pub struct EcMulControlCols<T, const NUM_LIMBS: usize> {
    // Scalar bits (constant after digest row sets them)
    pub scalar_bits: [T; EC_MUL_SCALAR_BITS],

    // Base point limbs (constant after digest row sets them)
    pub base_px_limbs: [T; NUM_LIMBS],
    pub base_py_limbs: [T; NUM_LIMBS],

    // Current accumulator limbs (updated each compute row)
    pub rx_limbs: [T; NUM_LIMBS],
    pub ry_limbs: [T; NUM_LIMBS],
}

/// Flag columns for row type identification
/// Mutually exclusive: is_compute_row + is_digest_row = 1
#[repr(C)]
#[derive(Clone, Copy, Debug, AlignedBorrow)]
pub struct EcMulFlagsCols<T> {
    /// 1 for compute rows (rows 0-255 of an EC_MUL instruction)
    pub is_compute_row: T,
    /// 1 for digest row (row 256 of an EC_MUL instruction, handles memory I/O)
    pub is_digest_row: T,
    /// 1 for the first compute row (row_idx = 0 AND is_compute_row = 1)
    /// This is a derived flag but stored for convenience
    pub is_first_compute_row: T,
    /// 1 for SETUP_EC_MUL opcode, 0 for EC_MUL opcode
    /// This flag is constant across all 257 rows of an instruction
    pub is_setup: T,
    /// 1 when the accumulator R is at infinity (point at infinity).
    /// `is_inf = 1` on the first compute row (R starts at infinity).
    /// `is_inf_next = is_inf * (1 - bit)` (stays at infinity only if bit = 0).
    /// Used to determine when to use safe denominators in FieldExpr.
    pub is_inf: T,
    /// Encodes row index 0..257 within an EC_MUL instruction
    /// Using Encoder with max_degree=2: var_cnt=22 (since C(24,2)=276 >= 257)
    pub row_idx: [T; 22],
}

/// Width constants for column structures
/// Note: Compute row width is NOT calculated here - it's dynamic based on FieldExpr.width()
/// The total width is calculated in the AIR's BaseAir::width() implementation
pub const fn ec_mul_compute_base_width<const NUM_LIMBS: usize>() -> usize {
    EcMulComputeCols::<u8, NUM_LIMBS>::width()
}

pub const fn ec_mul_digest_width<
    const NUM_LIMBS: usize,
    const NUM_BLOCKS: usize,
    const BLOCK_SIZE: usize,
>() -> usize {
    EcMulDigestCols::<u8, NUM_LIMBS, NUM_BLOCKS, BLOCK_SIZE>::width()
}
