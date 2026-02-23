use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

mod air;
pub use air::*;
mod trace;
pub use trace::*;
mod execution;

pub type DeferralSetupAir<F> = VmAirWrapper<DeferralSetupAdapterAir, DeferralSetupCoreAir<F>>;
pub type DeferralSetupExecutor<F> = DeferralSetupCoreExecutor<F, DeferralSetupAdapterExecutor>;
pub type DeferralSetupChip<F> =
    VmChipWrapper<F, DeferralSetupCoreFiller<DeferralSetupAdapterFiller>>;
