use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

mod air;
pub use air::*;
mod trace;
pub use trace::*;
mod execution;

pub type DeferralCallAir = VmAirWrapper<DeferralCallAdapterAir, DeferralCallCoreAir>;
pub type DeferralCallExecutor = DeferralCallCoreExecutor<DeferralCallAdapterExecutor>;
pub type DeferralCallChip<F> =
    VmChipWrapper<F, DeferralCallCoreFiller<DeferralCallAdapterFiller, F>>;
