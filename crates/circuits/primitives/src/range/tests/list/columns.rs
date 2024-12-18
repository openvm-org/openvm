use core::mem::{size_of, transmute};

use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_stark_backend::p3_util::indices_arr;

#[derive(Default, AlignedBorrow)]
pub struct ListCols<T> {
    pub val: T,
}

pub const NUM_LIST_COLS: usize = size_of::<ListCols<u8>>();
pub const LIST_COL_MAP: ListCols<usize> = make_col_map();

const fn make_col_map() -> ListCols<usize> {
    let indices_arr = indices_arr::<NUM_LIST_COLS>();
    unsafe { transmute::<[usize; NUM_LIST_COLS], ListCols<usize>>(indices_arr) }
}
