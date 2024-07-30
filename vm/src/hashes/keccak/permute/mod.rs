pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

/// Number of u64 elements in a Keccak hash.
pub const NUM_U64_HASH_ELEMS: usize = 4;

pub use air::KeccakPermuteAir;
use columns::{KeccakPermuteAuxCols, KeccakPermuteIoCols};
use p3_field::PrimeField32;
use p3_keccak_air::U64_LIMBS;

#[derive(Clone, Debug)]
pub struct KeccakPermuteChip<F: PrimeField32> {
    pub air: KeccakPermuteAir,
    /// IO and memory data necessary for each opcode call
    pub requests: Vec<(KeccakPermuteIoCols<F>, KeccakPermuteAuxCols<F>)>,
    /// The input state of each keccak-f permutation corresponding to `requests`.
    /// Must have same length as `requests`.
    pub inputs: Vec<[u64; 25]>,
}

impl<F: PrimeField32> KeccakPermuteChip<F> {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            air: KeccakPermuteAir {},
            requests: Vec::new(),
            inputs: Vec::new(),
        }
    }

    pub fn max_accesses_per_instruction() -> usize {
        // 2 for reading dst, src
        // U64_LIMBS * 25 to read input
        // U64_LIMBS * 25 to write output
        2 + U64_LIMBS * 25 + U64_LIMBS * 25
    }
}
