use afs_middleware_derive::AlignedBorrow;
use core::mem::{size_of, transmute};
use p3_util::indices_arr;

// pub const MERKLE_TREE_DEPTH: usize = 4;
pub const MERKLE_TREE_DEPTH: usize = 8; // PAGE_SIZE_BITS
pub const NUM_U64_HASH_ELEMS: usize = 4;
pub const NUM_U16_LIMBS: usize = 4;

#[repr(C)]
#[derive(Default, AlignedBorrow)]
pub struct MerkleTreeCols<T> {
    pub is_real: T,

    // Ideally preprocessed or with periodic constraint
    pub is_first_step: T,

    pub is_final_step: T,

    pub node: [[T; NUM_U16_LIMBS]; NUM_U64_HASH_ELEMS],

    pub sibling: [[T; NUM_U16_LIMBS]; NUM_U64_HASH_ELEMS],

    pub is_right_child: T,

    pub bit_factor: T,

    pub accumulated_index: T,

    pub index: T,

    pub left_node: [[T; NUM_U16_LIMBS]; NUM_U64_HASH_ELEMS],

    pub right_node: [[T; NUM_U16_LIMBS]; NUM_U64_HASH_ELEMS],

    pub output: [[T; NUM_U16_LIMBS]; NUM_U64_HASH_ELEMS],
}

impl<T: Copy> MerkleTreeCols<T> {}

pub(crate) const NUM_MERKLE_TREE_COLS: usize = size_of::<MerkleTreeCols<u8>>();
pub(crate) const MERKLE_TREE_COL_MAP: MerkleTreeCols<usize> = make_col_map();

const fn make_col_map() -> MerkleTreeCols<usize> {
    let indices_arr = indices_arr::<NUM_MERKLE_TREE_COLS>();
    unsafe { transmute::<[usize; NUM_MERKLE_TREE_COLS], MerkleTreeCols<usize>>(indices_arr) }
}
