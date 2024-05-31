use core::mem::{size_of, transmute};

use afs_derive::AlignedBorrow;
use p3_keccak_air::KeccakCols;
use p3_util::indices_arr;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct KeccakPermuteCols<T> {
    pub keccak: KeccakCols<T>,

    pub is_real: T,

    pub is_real_input: T,

    pub is_real_output: T,
}

pub const NUM_KECCAK_PERMUTE_COLS: usize = size_of::<KeccakPermuteCols<u8>>();
pub(crate) const KECCAK_PERMUTE_COL_MAP: KeccakPermuteCols<usize> = make_col_map();

const fn make_col_map() -> KeccakPermuteCols<usize> {
    let indices_arr = indices_arr::<NUM_KECCAK_PERMUTE_COLS>();
    unsafe { transmute::<[usize; NUM_KECCAK_PERMUTE_COLS], KeccakPermuteCols<usize>>(indices_arr) }
}
