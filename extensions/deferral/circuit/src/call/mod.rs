use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

/// A deferral call manipulates two accumulators per `deferral_idx`: input + output.
const NUM_ACCUMULATORS_PER_IDX: usize = 2;

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
