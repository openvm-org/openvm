use columns::KeccakOpcodeCols;
use p3_field::PrimeField32;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub use air::KeccakVmAir;

/// Number of u64 elements in a Keccak hash.
pub const NUM_U64_HASH_ELEMS: usize = 4;
/// Total number of sponge bytes: number of rate bytes + number of capacity
/// bytes.
pub const KECCAK_WIDTH_BYTES: usize = 200;
/// Total number of 16-bit limbs in the sponge.
pub const KECCAK_WIDTH_U16S: usize = KECCAK_WIDTH_BYTES / 2;
/// Number of non-digest bytes.
pub const KECCAK_WIDTH_MINUS_DIGEST_U16S: usize = (KECCAK_WIDTH_BYTES - KECCAK_DIGEST_BYTES) / 2;
/// Number of rate bytes.
pub const KECCAK_RATE_BYTES: usize = 136;
/// Number of 16-bit rate limbs.
pub const KECCAK_RATE_U16S: usize = KECCAK_RATE_BYTES / 2;
/// Number of absorb rounds, equal to rate in u64s.
pub const NUM_ABSORB_ROUNDS: usize = KECCAK_RATE_BYTES / 8;
/// Number of capacity bytes.
pub const KECCAK_CAPACITY_BYTES: usize = 64;
/// Number of 16-bit capacity limbs.
pub const KECCAK_CAPACITY_U16S: usize = KECCAK_CAPACITY_BYTES / 2;
/// Number of output digest bytes used during the squeezing phase.
pub const KECCAK_DIGEST_BYTES: usize = 32;
/// Number of 16-bit digest limbs.
pub const KECCAK_DIGEST_U16S: usize = KECCAK_DIGEST_BYTES / 2;

#[derive(Clone, Debug)]
pub struct KeccakVmChip<F: PrimeField32> {
    pub air: KeccakVmAir,
    /// IO and memory data necessary for each opcode call
    pub requests: Vec<KeccakOpcodeCols<F>>,
    /// The input state of each keccak-f permutation corresponding to `requests`.
    /// Must have same length as `requests`.
    pub inputs: Vec<[u64; 25]>,
}

impl<F: PrimeField32> KeccakVmChip<F> {
    #[allow(clippy::new_without_default)]
    pub fn new(xor_bus_index: usize) -> Self {
        Self {
            air: KeccakVmAir::new(xor_bus_index),
            requests: Vec::new(),
            inputs: Vec::new(),
        }
    }
}
