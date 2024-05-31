mod air;
mod chip;
mod columns;
mod trace;

/// Number of u64 elements in a Keccak hash.
pub const NUM_U64_HASH_ELEMS: usize = 4;

#[derive(Clone)]
pub struct KeccakPermuteChip {
    pub bus_input: usize,
    pub bus_output: usize,
}
