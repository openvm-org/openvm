use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

mod adapter;
pub use adapter::*;
mod core;
pub use core::*;
mod execution;

pub type DeferralCallAir = VmAirWrapper<DeferralCallAdapterAir, DeferralCallCoreAir>;
pub type DeferralCallExecutor = DeferralCallCoreExecutor<DeferralCallAdapterExecutor>;
pub type DeferralCallChip<F> =
    VmChipWrapper<F, DeferralCallCoreFiller<DeferralCallAdapterFiller, F>>;
