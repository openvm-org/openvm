use openvm_circuit::system::memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_poseidon2_air::Poseidon2SubCols;

use crate::poseidon2::CHUNK;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct NativePoseidon2Cols<T, const SBOX_REGISTERS: usize> {
    // poseidon2
    pub inner: Poseidon2SubCols<T, SBOX_REGISTERS>,

    // flags - at most 1 is true, if none is true then row is disabled
    pub incorporate_row: T,
    pub incorporate_sibling: T,
    pub inside_row: T,
    pub simple: T,

    pub end_inside_row: T,
    pub end_top_level: T,
    pub start_top_level: T,

    pub very_first_timestamp: T,
    pub start_timestamp: T,

    // instruction (g)
    pub opened_element_size_inv: T,

    // initial/final opened index for a subsegment with same height
    pub initial_opened_index: T,

    pub opened_base_pointer: T,

    // cannot be shared, should be 0 on rows that are not inside row
    pub is_exhausted: [T; CHUNK],

    pub specific: [T; max3(
        TopLevelSpecificCols::<usize>::width(),
        InsideRowSpecificCols::<usize>::width(),
        SimplePoseidonSpecificCols::<usize>::width(),
    )],
}

const fn max(a: usize, b: usize) -> usize {
    [a, b][(a < b) as usize]
}
const fn max3(a: usize, b: usize, c: usize) -> usize {
    max(a, max(b, c))
}
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct TopLevelSpecificCols<T> {
    pub pc: T,
    pub end_timestamp: T,

    // instruction (a, b, c, d, e, f)
    pub dim_register: T,
    pub opened_register: T,
    pub opened_length_register: T,
    pub proof_id: T,
    pub index_register: T,
    pub commit_register: T,

    pub final_opened_index: T,

    pub log_height: T,
    pub opened_length: T,

    pub dim_base_pointer: T,
    pub index_base_pointer: T,

    pub dim_base_pointer_read: MemoryReadAuxCols<T>,
    pub opened_base_pointer_read: MemoryReadAuxCols<T>,
    pub opened_length_read: MemoryReadAuxCols<T>,
    pub index_base_pointer_read: MemoryReadAuxCols<T>,
    pub commit_pointer_read: MemoryReadAuxCols<T>,

    pub proof_index: T,

    pub read_initial_height_or_root_is_on_right: MemoryReadAuxCols<T>,
    pub read_final_height: MemoryReadAuxCols<T>,

    // incorporate sibling only
    pub root_is_on_right: T,
    pub commit_pointer: T,
    pub commit_read: MemoryReadAuxCols<T>,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct InsideRowSpecificCols<T> {
    pub cells: [VerifyBatchCellCols<T>; CHUNK],
}

#[repr(C)]
#[derive(AlignedBorrow, Copy, Clone)]
pub struct VerifyBatchCellCols<T> {
    pub read: MemoryReadAuxCols<T>,
    pub opened_index: T,
    pub read_row_pointer_and_length: MemoryReadAuxCols<T>,
    pub row_pointer: T,
    pub row_end: T,
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
