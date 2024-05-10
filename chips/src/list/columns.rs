use afs_middleware_derive::AlignedBorrow;
use core::mem::{size_of, transmute};
use p3_util::indices_arr;

#[cfg(feature = "debug-trace")]
use p3_test_derive::Headers;

#[derive(Default, AlignedBorrow)]
#[cfg_attr(feature = "debug-trace", derive(Headers))]
pub struct ListCols<T> {
    pub val: T,
}

pub const NUM_LIST_COLS: usize = size_of::<ListCols<u8>>();
pub const LIST_COL_MAP: ListCols<usize> = make_col_map();

const fn make_col_map() -> ListCols<usize> {
    let indices_arr = indices_arr::<NUM_LIST_COLS>();
    unsafe { transmute::<[usize; NUM_LIST_COLS], ListCols<usize>>(indices_arr) }
}
