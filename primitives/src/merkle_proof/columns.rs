use afs_derive::AlignedBorrow;
use core::mem::size_of;
use std::mem::MaybeUninit;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct MerkleProofCols<T, const DEPTH: usize, const DIGEST_WIDTH: usize> {
    // TODO: Add clk/timestamp
    pub is_real: T,

    pub step_flags: [T; DEPTH],

    pub node: [T; DIGEST_WIDTH],

    pub sibling: [T; DIGEST_WIDTH],

    pub is_right_child: T,

    pub accumulated_index: T,

    pub index: T,

    pub left_node: [T; DIGEST_WIDTH],

    pub right_node: [T; DIGEST_WIDTH],

    pub output: [T; DIGEST_WIDTH],
}

pub(crate) const fn num_merkle_proof_cols<const DEPTH: usize, const DIGEST_WIDTH: usize>() -> usize
{
    size_of::<MerkleProofCols<u8, DEPTH, DIGEST_WIDTH>>()
}

pub(crate) fn merkle_proof_col_map<const DEPTH: usize, const DIGEST_WIDTH: usize>(
) -> MerkleProofCols<usize, DEPTH, DIGEST_WIDTH> {
    let num_cols = num_merkle_proof_cols::<DEPTH, DIGEST_WIDTH>();
    let indices_arr = (0..num_cols).collect::<Vec<usize>>();

    let mut cols = MaybeUninit::<MerkleProofCols<usize, DEPTH, DIGEST_WIDTH>>::uninit();
    let ptr = cols.as_mut_ptr() as *mut usize;
    unsafe {
        ptr.copy_from_nonoverlapping(indices_arr.as_ptr(), num_cols);
        cols.assume_init()
    }
}
