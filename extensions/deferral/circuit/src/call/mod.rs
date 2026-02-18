use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

mod adapter;
mod core;
mod execution;

pub use adapter::*;
pub use core::*;

pub type DeferralCallAir = VmAirWrapper<DeferralCallAdapterAir, DeferralCallCoreAir>;
pub type DeferralCallExecutor<F> = DeferralCallCoreExecutor<DeferralCallAdapterExecutor, F>;
pub type DeferralCallChip<F> = VmChipWrapper<F, DeferralCallCoreFiller<DeferralCallAdapterFiller>>;
