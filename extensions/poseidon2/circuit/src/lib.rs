#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]
#![cfg_attr(feature = "tco", allow(internal_features))]
#![cfg_attr(feature = "tco", feature(core_intrinsics))]

/// Sub-AIR constraining that a 4-byte decomposition canonically represents a field element.
pub mod canonicity;
/// Poseidon2 extension, config, builder and prover extension.
pub mod extension;
/// Wrapper around the `openvm-poseidon2-air` permutation AIR with a direct lookup bus for
/// interaction with `Poseidon2PermuteAir`.
pub mod periphery;
/// The AIR that handles interactions with the VM ExecutionBus and MemoryBus for handling of the
/// `PERMUTE` opcode.
pub mod permute;

// ==== VM-specific constants ====
/// Number of cells to read/write in a single memory access
pub const POSEIDON2_WORD_SIZE: usize = 4;
/// Total number of bytes of the Poseidon2 state.
pub const POSEIDON2_STATE_BYTES: usize = 64;

/// Number of Poseidon2 S-box registers. Affects the max constraint degree of the AIR; the
/// periphery AIR must satisfy `DEFAULT_POSEIDON2_MAX_CONSTRAINT_DEGREE`.
pub(crate) const SBOX_REGISTERS: usize = 1;
