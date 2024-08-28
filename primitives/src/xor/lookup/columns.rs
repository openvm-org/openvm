use core::mem::{size_of, transmute};

use afs_derive::AlignedBorrow;
use p3_util::indices_arr;

#[derive(Copy, Clone, Debug, AlignedBorrow)]
pub struct XorLookupCols<T> {
    pub mult: T,
}

#[derive(Copy, Clone, Debug, AlignedBorrow)]
pub struct XorLookupPreprocessedCols<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

pub const NUM_XOR_LOOKUP_COLS: usize = size_of::<XorLookupCols<u8>>();
pub const XOR_LOOKUP_COL_MAP: XorLookupCols<usize> = make_col_map();

pub const NUM_XOR_LOOKUP_PREPROCESSED_COLS: usize = size_of::<XorLookupPreprocessedCols<u8>>();
pub const XOR_LOOKUP_PREPROCESSED_COL_MAP: XorLookupPreprocessedCols<usize> =
    make_preprocessed_col_map();

const fn make_col_map() -> XorLookupCols<usize> {
    let indices_arr = indices_arr::<NUM_XOR_LOOKUP_COLS>();
    unsafe { transmute::<[usize; NUM_XOR_LOOKUP_COLS], XorLookupCols<usize>>(indices_arr) }
}

const fn make_preprocessed_col_map() -> XorLookupPreprocessedCols<usize> {
    let indices_arr = indices_arr::<NUM_XOR_LOOKUP_PREPROCESSED_COLS>();
    unsafe {
        transmute::<[usize; NUM_XOR_LOOKUP_PREPROCESSED_COLS], XorLookupPreprocessedCols<usize>>(
            indices_arr,
        )
    }
}
