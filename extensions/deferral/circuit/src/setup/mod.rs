use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

mod adapter;
mod core;
mod execution;

pub use adapter::*;
pub use core::*;

pub type DeferralSetupAir<F> = VmAirWrapper<DeferralSetupAdapterAir, DeferralSetupCoreAir<F>>;
pub type DeferralSetupExecutor<F> = DeferralSetupCoreExecutor<F, DeferralSetupAdapterExecutor>;
pub type DeferralSetupChip<F> =
    VmChipWrapper<F, DeferralSetupCoreFiller<DeferralSetupAdapterFiller>>;
