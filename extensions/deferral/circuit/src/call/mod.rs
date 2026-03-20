use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

mod air;
pub use air::*;
mod trace;
pub use trace::*;
mod execution;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

pub type DeferralCallAir = VmAirWrapper<DeferralCallAdapterAir, DeferralCallCoreAir>;
pub type DeferralCallExecutor = DeferralCallCoreExecutor<DeferralCallAdapterExecutor>;
pub type DeferralCallChip<F> =
    VmChipWrapper<F, DeferralCallCoreFiller<DeferralCallAdapterFiller, F>>;
