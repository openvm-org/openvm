mod air;
mod chip;
pub mod columns;
mod round_flags;
mod trace;

#[derive(Clone)]
pub struct MerkleProofOp<T, const DEPTH: usize, const DIGEST_WIDTH: usize>
where
    T: Default + Copy,
{
    pub leaf_index: usize,
    pub leaf_hash: [T; DIGEST_WIDTH],
    pub siblings: [[T; DIGEST_WIDTH]; DEPTH],
}

impl<T, const DEPTH: usize, const DIGEST_WIDTH: usize> Default
    for MerkleProofOp<T, DEPTH, DIGEST_WIDTH>
where
    T: Default + Copy,
{
    fn default() -> Self {
        Self {
            leaf_index: 0,
            leaf_hash: [T::default(); DIGEST_WIDTH],
            siblings: [[T::default(); DIGEST_WIDTH]; DEPTH],
        }
    }
}

#[derive(Clone)]
pub struct MerkleProofChip<const DEPTH: usize, const DIGEST_WIDTH: usize> {
    pub bus_hash_input: usize,
    pub bus_hash_output: usize,
}
