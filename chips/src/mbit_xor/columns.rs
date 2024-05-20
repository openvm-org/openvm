use afs_derive::AlignedBorrow;
use core::mem::{size_of, transmute};
use p3_util::indices_arr;

#[derive(Default, AlignedBorrow)]
pub struct MBitXorCols<T> {
    pub mult: T,
}

#[derive(Default, AlignedBorrow)]
pub struct MBitXorPreprocessedCols<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

pub const NUM_MBIT_XOR_COLS: usize = size_of::<MBitXorCols<u8>>();
pub const MBIT_XOR_COL_MAP: MBitXorCols<usize> = make_col_map();

pub const NUM_MBIT_XOR_PREPROCESSED_COLS: usize = size_of::<MBitXorPreprocessedCols<u8>>();
pub const MBIT_XOR_PREPROCESSED_COL_MAP: MBitXorPreprocessedCols<usize> =
    make_preprocessed_col_map();

const fn make_col_map() -> MBitXorCols<usize> {
    let indices_arr = indices_arr::<NUM_MBIT_XOR_COLS>();
    unsafe { transmute::<[usize; NUM_MBIT_XOR_COLS], MBitXorCols<usize>>(indices_arr) }
}

const fn make_preprocessed_col_map() -> MBitXorPreprocessedCols<usize> {
    let indices_arr = indices_arr::<NUM_MBIT_XOR_PREPROCESSED_COLS>();
    unsafe {
        transmute::<[usize; NUM_MBIT_XOR_PREPROCESSED_COLS], MBitXorPreprocessedCols<usize>>(
            indices_arr,
        )
    }
}
