use core::mem::{size_of, transmute};

use afs_derive::AlignedBorrow;
use p3_keccak_air::KeccakCols;
use p3_util::indices_arr;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct KeccakPermuteCols<T> {
    pub inner: KeccakCols<T>,
    pub io: KeccakPermuteIoCols<T>,
    pub aux: KeccakPermuteAuxCols<T>,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct KeccakPermuteIoCols<T> {
    /// Whether row corresponds to an opcode (PERMUTE)
    pub is_opcode: T,
    /// The clock cycle (NOT timestamp)
    pub clk: T,
    pub a: T,
    // pub b: T, // b = offset = 0
    pub c: T,
    pub d: T,
    pub e: T,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct KeccakPermuteAuxCols<T> {
    pub dst: T,
    pub src: T,
}

pub const NUM_KECCAK_PERMUTE_COLS: usize = size_of::<KeccakPermuteCols<u8>>();
pub(crate) const KECCAK_PERMUTE_COL_MAP: KeccakPermuteCols<usize> = make_col_map();

const fn make_col_map() -> KeccakPermuteCols<usize> {
    let indices_arr = indices_arr::<NUM_KECCAK_PERMUTE_COLS>();
    unsafe { transmute::<[usize; NUM_KECCAK_PERMUTE_COLS], KeccakPermuteCols<usize>>(indices_arr) }
}
