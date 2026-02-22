use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

mod adapter;
pub use adapter::*;
mod core;
pub use core::*;
mod execution;

pub type DeferralSetupAir<F> = VmAirWrapper<DeferralSetupAdapterAir, DeferralSetupCoreAir<F>>;
pub type DeferralSetupExecutor<F> = DeferralSetupCoreExecutor<F, DeferralSetupAdapterExecutor>;
pub type DeferralSetupChip<F> =
    VmChipWrapper<F, DeferralSetupCoreFiller<DeferralSetupAdapterFiller>>;
