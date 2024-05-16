mod air;
mod chip;
pub mod columns;
mod trace;

use self::columns::{MERKLE_TREE_DEPTH, NUM_MERKLE_TREE_COLS};

pub(crate) const NUM_U8_HASH_ELEMS: usize = 32;

#[derive(Default, Clone)]
pub struct MerkleTreeChip {
    pub leaves: Vec<[u8; NUM_U8_HASH_ELEMS]>,
    pub leaf_indices: Vec<usize>,
    pub siblings: Vec<[[u8; NUM_U8_HASH_ELEMS]; MERKLE_TREE_DEPTH]>,
}
