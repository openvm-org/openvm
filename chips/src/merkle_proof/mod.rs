mod air;
mod chip;
pub mod columns;
mod round_flags;
mod trace;

use self::columns::{MERKLE_PROOF_DEPTH, NUM_MERKLE_PROOF_COLS};

pub(crate) const NUM_U8_HASH_ELEMS: usize = 32;

#[derive(Default, Clone)]
pub struct MerkleProofChip {
    pub bus_hash_input: usize,
    pub bus_hash_output: usize,

    pub leaves: Vec<[u8; NUM_U8_HASH_ELEMS]>,
    pub leaf_indices: Vec<usize>,
    pub siblings: Vec<[[u8; NUM_U8_HASH_ELEMS]; MERKLE_PROOF_DEPTH]>,
}
