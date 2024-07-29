use core::mem::{size_of, transmute};

use afs_derive::AlignedBorrow;
use p3_keccak_air::KeccakCols;
use p3_util::indices_arr;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct KeccakPermuteCols<T> {
    pub keccak: KeccakCols<T>,

    /// Whether row corresponds to an opcode (PERMUTE)
    // pub is_opcode: T,
    /// Whether row is for a direct inter-chip permute interaction
    pub is_direct: T,
    /// The clock cycle (NOT timestamp)
    pub clk: T,
}

pub const NUM_KECCAK_PERMUTE_COLS: usize = size_of::<KeccakPermuteCols<u8>>();
pub(crate) const KECCAK_PERMUTE_COL_MAP: KeccakPermuteCols<usize> = make_col_map();

const fn make_col_map() -> KeccakPermuteCols<usize> {
    let indices_arr = indices_arr::<NUM_KECCAK_PERMUTE_COLS>();
    unsafe { transmute::<[usize; NUM_KECCAK_PERMUTE_COLS], KeccakPermuteCols<usize>>(indices_arr) }
}
