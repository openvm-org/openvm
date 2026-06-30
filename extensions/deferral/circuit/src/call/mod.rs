use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_deferral_guest::MAX_DEF_CIRCUITS;
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

/// Number of accumulator digests stored for each `deferral_idx`.
pub(in crate::call) const NUM_ACCUMULATORS_PER_IDX: usize = 2;

// The accumulator cell pointers live entirely in the low 16-bit pointer limb. The count bus
// constrains `deferral_idx < num_deferral_circuits <= MAX_DEF_CIRCUITS`, so every accumulator cell
// pointer (`NUM_ACCUMULATORS_PER_IDX * deferral_idx * DIGEST_SIZE` plus an offset
// `< NUM_ACCUMULATORS_PER_IDX * DIGEST_SIZE`) is strictly below
// `NUM_ACCUMULATORS_PER_IDX * MAX_DEF_CIRCUITS * DIGEST_SIZE`. As long as that bound fits in 2^16,
// the high pointer limb is identically zero.
const _: () =
    assert!(NUM_ACCUMULATORS_PER_IDX * (MAX_DEF_CIRCUITS as usize) * DIGEST_SIZE <= (1 << 16));

#[inline(always)]
pub(in crate::call) const fn accumulator_ptrs(deferral_idx: u32) -> (u32, u32) {
    let input_acc_ptr = (NUM_ACCUMULATORS_PER_IDX as u32) * deferral_idx * DIGEST_SIZE as u32;
    (input_acc_ptr, input_acc_ptr + DIGEST_SIZE as u32)
}

mod air;
pub use air::*;
mod trace;
pub use trace::*;
mod execution;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type DeferralCallAir = VmAirWrapper<DeferralCallAdapterAir, DeferralCallCoreAir>;
pub type DeferralCallExecutor = DeferralCallCoreExecutor<DeferralCallAdapterExecutor>;
pub type DeferralCallChip<F> =
    VmChipWrapper<F, DeferralCallCoreFiller<DeferralCallAdapterFiller, F>>;
