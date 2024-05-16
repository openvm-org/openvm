use afs_derive::AlignedBorrow;
use core::mem::{size_of, transmute};
use p3_util::indices_arr;

#[derive(Default, AlignedBorrow)]
pub struct XorRequesterCols<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

pub const NUM_XOR_REQUESTER_COLS: usize = size_of::<XorRequesterCols<u8>>();
pub const XOR_REQUESTER_COL_MAP: XorRequesterCols<usize> = make_col_map();

const fn make_col_map() -> XorRequesterCols<usize> {
    let indices_arr = indices_arr::<NUM_XOR_REQUESTER_COLS>();
    unsafe { transmute::<[usize; NUM_XOR_REQUESTER_COLS], XorRequesterCols<usize>>(indices_arr) }
}
