use afs_derive::AlignedBorrow;
use core::mem::{size_of, transmute};
use p3_util::indices_arr;

// pub const MERKLE_PROOF_DEPTH: usize = 4;
pub const MERKLE_PROOF_DEPTH: usize = 8; // PAGE_SIZE_BITS
pub const NUM_U64_HASH_ELEMS: usize = 4;
pub const NUM_U16_LIMBS: usize = 4;

#[repr(C)]
#[derive(Default, AlignedBorrow)]
pub struct MerkleProofCols<T> {
    pub is_real: T,

    pub step_flags: [T; MERKLE_PROOF_DEPTH],

    pub node: [[T; NUM_U16_LIMBS]; NUM_U64_HASH_ELEMS],

    pub sibling: [[T; NUM_U16_LIMBS]; NUM_U64_HASH_ELEMS],

    pub is_right_child: T,

    pub accumulated_index: T,

    pub index: T,

    pub left_node: [[T; NUM_U16_LIMBS]; NUM_U64_HASH_ELEMS],

    pub right_node: [[T; NUM_U16_LIMBS]; NUM_U64_HASH_ELEMS],

    pub output: [[T; NUM_U16_LIMBS]; NUM_U64_HASH_ELEMS],
}

impl<T: Copy> MerkleProofCols<T> {}

pub(crate) const NUM_MERKLE_PROOF_COLS: usize = size_of::<MerkleProofCols<u8>>();
pub(crate) const MERKLE_PROOF_COL_MAP: MerkleProofCols<usize> = make_col_map();

const fn make_col_map() -> MerkleProofCols<usize> {
    let indices_arr = indices_arr::<NUM_MERKLE_PROOF_COLS>();
    unsafe { transmute::<[usize; NUM_MERKLE_PROOF_COLS], MerkleProofCols<usize>>(indices_arr) }
}
