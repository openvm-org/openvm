use core::mem::size_of;

use afs_derive::AlignedBorrow;

#[derive(Default, AlignedBorrow, Copy, Clone)]
#[repr(C)]
pub struct VariableRangeCols<T> {
    pub mult: T,
}

#[derive(Default, AlignedBorrow, Copy, Clone)]
#[repr(C)]
pub struct VariableRangePreprocessedCols<T> {
    pub value: T,
    pub max_bits: T,
}

pub const NUM_VARIABLE_RANGE_COLS: usize = size_of::<VariableRangeCols<u8>>();
pub const NUM_VARIABLE_RANGE_PREPROCESSED_COLS: usize =
    size_of::<VariableRangePreprocessedCols<u8>>();
