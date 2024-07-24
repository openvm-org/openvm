use afs_derive::AlignedBorrow;
use core::mem::{size_of, transmute};
use p3_util::indices_arr;

#[derive(Copy, Clone, Default, AlignedBorrow)]
pub struct RangeGateCols<T> {
    pub counter: T,
    pub mult: T,
}

impl<T: Clone> RangeGateCols<T> {
    pub fn from_slice(slice: &[T]) -> Self {
        let counter = slice[0].clone();
        let mult = slice[1].clone();

        Self { counter, mult }
    }
}

pub const NUM_RANGE_GATE_COLS: usize = size_of::<RangeGateCols<u8>>();
pub const RANGE_GATE_COL_MAP: RangeGateCols<usize> = make_col_map();

const fn make_col_map() -> RangeGateCols<usize> {
    let indices_arr = indices_arr::<NUM_RANGE_GATE_COLS>();
    unsafe { transmute::<[usize; NUM_RANGE_GATE_COLS], RangeGateCols<usize>>(indices_arr) }
}
