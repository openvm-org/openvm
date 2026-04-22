#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]
#![cfg_attr(feature = "tco", allow(internal_features))]
#![cfg_attr(feature = "tco", feature(core_intrinsics))]

/// The AIR that handles interactions with the VM ExecutionBus and MemoryBus for handling of the
/// keccakf opcode.
pub mod keccakf_op;
/// Wrapper around the Plonky3 keccakf permutation AIR with a direct lookup bus for interaction with
/// `KeccakfOpAir`.
mod keccakf_perm;
/// AIR that handles the `xorin` opcode.
pub mod xorin;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

mod constants;
mod extension;
pub use constants::*;
pub use extension::*;
