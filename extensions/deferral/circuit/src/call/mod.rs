use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

/// Number of accumulator digests stored for each `deferral_idx`.
pub(in crate::call) const NUM_ACCUMULATORS_PER_IDX: usize = 2;

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
