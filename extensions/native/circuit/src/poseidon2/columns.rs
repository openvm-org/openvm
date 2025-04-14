use openvm_circuit::system::memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_poseidon2_air::Poseidon2SubCols;

use crate::{poseidon2::CHUNK, utils::const_max};

/// A valid trace is composed of four types of contiguous blocks:
///
/// 1. **Disabled Block:** A single row marked as disabled.
/// 2. **Simple Block:** A single row handling permutation/compression operations.
/// 3. **Inside-Row Block:** A sequence of rows that compute the row-hash for all input matrix
///    columns corresponding to an `MmcsVerifyBatch` input of the same height.
/// 4. **Top-Level Block:** A sequence of rows that perform Merkle tree compression on the row
///    hashes produced from an `MmcsVerifyBatch` input.
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct NativePoseidon2Cols<T, const SBOX_REGISTERS: usize> {
    /// Columns required to compute Poseidon2 permutation.
    pub inner: Poseidon2SubCols<T, SBOX_REGISTERS>,

    // Mode-of-operation flags. Each flag is boolean, and at most one may be true.
    // If none are true, the row is disabled.
    /// Indicates that this top-level block row is in incorporate-row mode.
    pub incorporate_row: T,
    /// Indicates that this top-level block row is in incorporate-sibling mode.
    pub incorporate_sibling: T,
    /// Indicates that this row is part of an inside-row block.
    pub inside_row: T,
    /// Indicates that this row is a simple row.
    pub simple: T,

    /// Indicates the last row in an inside-row block.
    pub end_inside_row: T,
    /// Indicates the last row in a top-level block.
    pub end_top_level: T,
    /// Indicates the first row in a top-level block.
    pub start_top_level: T,

    /// The initial timestamp of the instruction, which must be identical for all top-level and
    /// inside-row rows associated with the same instruction.
    pub very_first_timestamp: T,
    /// The starting timestamp of this row.
    pub start_timestamp: T,

    /// Operand `g` from the instruction. The multiplicative inverse of the size of an opened value
    /// in the opened values array. Must be consistent for all top-level and inside-row rows
    /// associated with the same instruction.
    pub opened_element_size_inv: T,

    /// On an `incorporate_row` row, this is the first matrix index `i` for which `log_heights[i]`
    /// equals `log_height`. On an `incorporate_sibling` row, this holds the initial index
    /// corresponding to the `log_height` for the next `incorporate_row` row, or
    /// `opened_length` if none exists.
    pub initial_opened_index: T,

    /// Pointer to the beginning of the `opened_values` array.
    pub opened_base_pointer: T,

    /// For rows that are not inside-row, this field should be 0. Otherwise, `is_exhausted[i]`
    /// indicates that cell `i + 1` inside a chunk is exhausted.
    pub is_exhausted: [T; CHUNK - 1],

    pub specific: [T; max3(
        TopLevelSpecificCols::<usize>::width(),
        InsideRowSpecificCols::<usize>::width(),
        SimplePoseidonSpecificCols::<usize>::width(),
    )],
}

const fn max3(a: usize, b: usize, c: usize) -> usize {
    const_max(a, const_max(b, c))
}
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct TopLevelSpecificCols<T> {
    /// The program counter for the VERIFY_BATCH instruction being processed.
    pub pc: T,

    /// The timestamp marking the end of processing this top-level row. For an
    /// `incorporate_sibling` row, it increases by a fixed amount. For an `incorporate_row`
    /// row, its increase depends on the row's length and the number of matrices involved, with
    /// additional constraints imposed by the internal bus.
    pub end_timestamp: T,

    /// Operand `a` from the instruction. Pointer to the `dimensions` array.
    pub dim_register: T,
    /// Operand `b` from the instruction. Pointer to the pointer of the `opened_values` array.
    pub opened_register: T,
    /// Operand `c` from the instruction. Pointer to the length of the `opened_values` array.
    pub opened_length_register: T,
    /// Operand `d` from the instruction. Provided as a hint to the run-time and (otherwise
    /// unconstrained).
    pub proof_id: T,
    /// Operand `e` from the instruction. Pointer to the pointer of the `index_bits` array, which
    /// indicates the direction (left/right) of Merkle tree siblings.
    pub index_register: T,
    /// Operand `f` from the instruction. Pointer to the pointer of the expected Merkle root.
    pub commit_register: T,

    /// For an `incorporate_row` row, the largest matrix index `i` such that `log_heights[i]`
    /// equals `log_height`. For an `incorporate_sibling` row, this is set to
    /// `initial_opened_index - 1` for bookkeeping.
    pub final_opened_index: T,

    /// The log height of the matrices currently being incorporated. Remains fixed on
    /// `incorporate_row` rows and decreases by one on `incorporate_sibling` rows.
    pub log_height: T,
    /// The length of the `opened_values` array, i.e., the number of non-empty traces.
    /// Equal to the value read in `opened_length_register`. Also constrained on the final row of
    /// the top-level block (constraint depends on if we end with an `incorporate_row` or an
    /// `incorporate_sibling`).
    pub opened_length: T,

    /// Pointer to the array of log heights.
    pub dim_base_pointer: T,
    /// Pointer to the array indicating Merkle proof directions.
    pub index_base_pointer: T,

    /// Memory aux columns for `dim_base_pointer` read.
    pub dim_base_pointer_read: MemoryReadAuxCols<T>,
    /// Memory aux columns for `opened_base_pointer` read.
    pub opened_base_pointer_read: MemoryReadAuxCols<T>,
    /// Memory aux columns for `opened_length` read.
    pub opened_length_read: MemoryReadAuxCols<T>,
    /// Memory aux columns for `index_base_pointer` read.
    pub index_base_pointer_read: MemoryReadAuxCols<T>,
    /// Memory aux columns for `commit_pointer` read.
    pub commit_pointer_read: MemoryReadAuxCols<T>,

    /// Index into the Merkle proof for the next sibling to incorporate.
    /// Starts at zero in a top-level block and increments by one after each `incorporate_sibling`
    /// row.
    pub proof_index: T,

    /// Memory aux columns for reading either `initial_height` or `sibling_is_on_right`. On an
    /// `incorporate_row` row, aux columns for reading `dims[initial_opened_index]`, and otherwise
    /// aux columns for `index_bits[proof_index]`.
    pub read_initial_height_or_sibling_is_on_right: MemoryReadAuxCols<T>,
    /// Memory aux columns for reading `dims[final_opened_index]`.
    pub read_final_height: MemoryReadAuxCols<T>,

    /// Indicator for whether the sibling being incorporated (if any) is on the right. Constrained
    /// to equal `index_bits[proof_index]` on `incorporate_sibling` rows. Unconstrained on other
    /// rows.
    pub sibling_is_on_right: T,
    /// Pointer to the Merkle root.
    pub commit_pointer: T,
    /// Memory aux columns for reading the Merkle root.
    pub commit_read: MemoryReadAuxCols<T>,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct InsideRowSpecificCols<T> {
    /// The columns to constrain a sequence of consecutive opened values (possibly cross-matrix).
    /// For an inside-row row, if `i == 0` or `is_exhausted[i - 1] != 0`, then `cells[i]` contains
    /// the information about which array the opened-value came from.
    pub cells: [VerifyBatchCellCols<T>; CHUNK],
}

/// Information about an opened value. We refer to `opened_values` as the two-dimensional array of
/// opened values; `opened_values[idx]` is an array of opened values.
#[repr(C)]
#[derive(AlignedBorrow, Copy, Clone)]
pub struct VerifyBatchCellCols<T> {
    /// Memory aux columns for the opened value.
    pub read: MemoryReadAuxCols<T>,
    /// The index into the `opened_values` array that this opened value came from.
    pub opened_index: T,
    /// Memory aux columns for reading `row_pointer` and length determining `row_end`; only used
    /// when `is_first_in_row = 1`.
    pub read_row_pointer_and_length: MemoryReadAuxCols<T>,
    /// Pointer to the opened value itself.
    pub row_pointer: T,
    /// Pointer just after the row given by `opened_values[opened_index]`.
    pub row_end: T,
    /// Whether this cell corresponds to `opened_values[opened_index][0]`.
    pub is_first_in_row: T,
}

#[repr(C)]
#[derive(AlignedBorrow, Copy, Clone)]
pub struct SimplePoseidonSpecificCols<T> {
    pub pc: T,
    pub is_compress: T,
    // instruction (a, b, c)
    pub output_register: T,
    pub input_register_1: T,
    pub input_register_2: T,

    pub output_pointer: T,
    pub input_pointer_1: T,
    pub input_pointer_2: T,

    pub read_output_pointer: MemoryReadAuxCols<T>,
    pub read_input_pointer_1: MemoryReadAuxCols<T>,
    pub read_input_pointer_2: MemoryReadAuxCols<T>,
    pub read_data_1: MemoryReadAuxCols<T>,
    pub read_data_2: MemoryReadAuxCols<T>,
    pub write_data_1: MemoryWriteAuxCols<T, { CHUNK }>,
    pub write_data_2: MemoryWriteAuxCols<T, { CHUNK }>,
}
