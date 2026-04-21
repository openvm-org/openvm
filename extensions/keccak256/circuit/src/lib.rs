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
use openvm_instructions::riscv::{RV64_REGISTER_NUM_LIMBS, RV64_WORD_NUM_LIMBS};
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;

/// Expand RV64_WORD_NUM_LIMBS (4) limbs to RV64_REGISTER_NUM_LIMBS (8) by zero-padding the upper
/// limbs. Used for register bus reads where the register holds a 32-bit value in the low 4 bytes.
pub fn expand_rv64_limbs<V: Copy + Into<T>, T: PrimeCharacteristicRing>(
    limbs: &[V; RV64_WORD_NUM_LIMBS],
) -> [T; RV64_REGISTER_NUM_LIMBS] {
    std::array::from_fn(|i| {
        if i < RV64_WORD_NUM_LIMBS {
            limbs[i].into()
        } else {
            T::ZERO
        }
    })
}
