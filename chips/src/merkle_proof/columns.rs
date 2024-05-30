use afs_derive::AlignedBorrow;
use core::mem::size_of;
use std::mem::MaybeUninit;

pub const NUM_U64_HASH_ELEMS: usize = 4;
pub const NUM_U16_LIMBS: usize = 4;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct MerkleProofCols<T, const DEPTH: usize> {
    pub is_real: T,

    pub step_flags: [T; DEPTH],

    pub node: [[T; NUM_U16_LIMBS]; NUM_U64_HASH_ELEMS],

    pub sibling: [[T; NUM_U16_LIMBS]; NUM_U64_HASH_ELEMS],

    pub is_right_child: T,

    pub accumulated_index: T,

    pub index: T,

    pub left_node: [[T; NUM_U16_LIMBS]; NUM_U64_HASH_ELEMS],

    pub right_node: [[T; NUM_U16_LIMBS]; NUM_U64_HASH_ELEMS],

    pub output: [[T; NUM_U16_LIMBS]; NUM_U64_HASH_ELEMS],
}

impl<T: Copy, const DEPTH: usize> MerkleProofCols<T, DEPTH> {}

pub(crate) const fn num_merkle_proof_cols<const DEPTH: usize>() -> usize {
    size_of::<MerkleProofCols<u8, DEPTH>>()
}

pub(crate) fn merkle_proof_col_map<const DEPTH: usize>() -> MerkleProofCols<usize, DEPTH> {
    let num_cols = num_merkle_proof_cols::<DEPTH>();
    let indices_arr = (0..num_cols).collect::<Vec<usize>>();

    unsafe {
        let mut uninit = MaybeUninit::<MerkleProofCols<usize, DEPTH>>::uninit();
        let ptr = uninit.as_mut_ptr() as *mut usize;
        ptr.copy_from_nonoverlapping(indices_arr.as_ptr(), num_cols);
        uninit.assume_init()
    }
}
