use derive_more::derive::From;
use openvm_circuit::{
    arch::{
        AirInventory, AirInventoryError, ChipInventory, ChipInventoryError,
        ExecutorInventoryBuilder, ExecutorInventoryError, VmCircuitExtension, VmExecutionExtension,
        VmProverExtension,
    },
    system::phantom::PhantomExecutor,
};
use openvm_circuit_derive::{AnyEnum, Executor, MeteredExecutor, PreflightExecutor};
use openvm_stark_backend::{p3_field::Field, StarkEngine, StarkProtocolConfig};
use serde::{Deserialize, Serialize};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        mod cuda;
        pub use self::cuda::DeferralGpuProverExt as DeferralProverExt;
    } else {
        pub use self::DeferralCpuProverExt as DeferralProverExt;
    }
}

// =================================== VM Extension Implementation =================================
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct DeferralExtension;

#[derive(Clone, From, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
#[cfg_attr(
    feature = "aot",
    derive(
        openvm_circuit_derive::AotExecutor,
        openvm_circuit_derive::AotMeteredExecutor
    )
)]
pub enum DeferralExecutor<F: Field> {
    Phantom(PhantomExecutor<F>),
}

impl<F: Field> VmExecutionExtension<F> for DeferralExtension {
    type Executor = DeferralExecutor<F>;

    fn extend_execution(
        &self,
        _inventory: &mut ExecutorInventoryBuilder<F, DeferralExecutor<F>>,
    ) -> Result<(), ExecutorInventoryError> {
        Ok(())
    }
}

impl<SC: StarkProtocolConfig> VmCircuitExtension<SC> for DeferralExtension {
    fn extend_circuit(&self, _inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        Ok(())
    }
}

pub struct DeferralCpuProverExt;
impl<E, RA> VmProverExtension<E, RA, DeferralExtension> for DeferralCpuProverExt
where
    E: StarkEngine,
{
    fn extend_prover(
        &self,
        _: &DeferralExtension,
        _inventory: &mut ChipInventory<E::SC, RA, E::PB>,
    ) -> Result<(), ChipInventoryError> {
        Ok(())
    }
}
