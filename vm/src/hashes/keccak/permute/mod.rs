pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

/// Number of u64 elements in a Keccak hash.
pub const NUM_U64_HASH_ELEMS: usize = 4;

pub use air::KeccakPermuteAir;

#[derive(Clone, Copy, Debug)]
pub struct KeccakPermuteChip;
