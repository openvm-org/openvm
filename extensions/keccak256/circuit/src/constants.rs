use openvm_circuit::arch::DEFAULT_BLOCK_SIZE;

/// Number of memory operations for the full keccakf state (200 / 8 = 25).
pub const KECCAK_WIDTH_MEM_OPS: usize = KECCAK_WIDTH_BYTES / DEFAULT_BLOCK_SIZE;
/// Number of memory operations for the keccak rate portion (136 / 8 = 17).
pub const KECCAK_RATE_MEM_OPS: usize = KECCAK_RATE_BYTES / DEFAULT_BLOCK_SIZE;

// ==== Do not change these constants! ====
/// Total number of sponge bytes: number of rate bytes + number of capacity
/// bytes.
pub const KECCAK_WIDTH_BYTES: usize = 200;
/// Total number of 16-bit limbs in the sponge.
pub const KECCAK_WIDTH_U16S: usize = KECCAK_WIDTH_BYTES / 2;
/// Total number of 64-bit limbs in the sponge.
pub const KECCAK_WIDTH_U64S: usize = KECCAK_WIDTH_BYTES / 8;
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
/// Number of 64-bit digest limbs.
pub const KECCAK_DIGEST_U64S: usize = KECCAK_DIGEST_BYTES / 8;
