mod air;
mod chip;
pub mod columns;
mod round_flags;
mod trace;

pub(crate) const NUM_U8_HASH_ELEMS: usize = 32;

#[derive(Default, Clone)]
pub struct MerkleProofChip<const DEPTH: usize> {
    pub bus_hash_input: usize,
    pub bus_hash_output: usize,

    pub leaves: Vec<[u8; NUM_U8_HASH_ELEMS]>,
    pub leaf_indices: Vec<usize>,
    pub siblings: Vec<[[u8; NUM_U8_HASH_ELEMS]; DEPTH]>,
}
