use std::sync::Arc;

use derive_more::derive::From;
use openvm_circuit::{
    arch::{
        AirInventory, AirInventoryError, ChipInventory, ChipInventoryError, ExecutionBridge,
        ExecutorInventoryBuilder, ExecutorInventoryError, RowMajorMatrixArena, VmCircuitExtension,
        VmExecutionExtension, VmField, VmProverExtension,
    },
    system::memory::SharedMemoryHelper,
};
use openvm_circuit_derive::{AnyEnum, Executor, MeteredExecutor, PreflightExecutor};
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::LocalOpcode;
use openvm_stark_backend::{
    p3_field::PrimeCharacteristicRing,
    prover::{CpuBackend, CpuDevice},
    StarkEngine, StarkProtocolConfig, Val,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use serde::{Deserialize, Serialize};

use crate::{
    call::{
        DeferralCallAdapterAir, DeferralCallAdapterExecutor, DeferralCallAdapterFiller,
        DeferralCallAir, DeferralCallChip, DeferralCallCoreAir, DeferralCallCoreFiller,
        DeferralCallExecutor,
    },
    count::{bus::DeferralCircuitCountBus, DeferralCircuitCountAir, DeferralCircuitCountChip},
    output::{DeferralOutputAir, DeferralOutputChip, DeferralOutputExecutor, DeferralOutputFiller},
    poseidon2::{
        bus::DeferralPoseidon2Bus, deferral_poseidon2_air, deferral_poseidon2_chip,
        DeferralPoseidon2Air,
    },
    setup::{
        DeferralSetupAdapterAir, DeferralSetupAdapterExecutor, DeferralSetupAdapterFiller,
        DeferralSetupAir, DeferralSetupChip, DeferralSetupCoreAir, DeferralSetupCoreFiller,
        DeferralSetupExecutor,
    },
};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        mod cuda;
        pub use self::cuda::DeferralGpuProverExt as DeferralProverExt;
    } else {
        pub use self::DeferralCpuProverExt as DeferralProverExt;
    }
}

// =================================== VM Extension Implementation =================================
#[derive(Clone, Copy, Debug, derive_new::new, Serialize, Deserialize)]
pub struct DeferralExtension {
    pub native_start_ptr: u32,
    pub expected_def_vks_commit: [u32; DIGEST_SIZE],
    pub num_deferral_circuits: usize,
}

impl Default for DeferralExtension {
    fn default() -> Self {
        Self {
            native_start_ptr: 0,
            expected_def_vks_commit: [0; DIGEST_SIZE],
            num_deferral_circuits: 0,
        }
    }
}

#[derive(Clone, From, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
#[cfg_attr(
    feature = "aot",
    derive(
        openvm_circuit_derive::AotExecutor,
        openvm_circuit_derive::AotMeteredExecutor
    )
)]
pub enum DeferralExecutor<F: VmField> {
    Setup(DeferralSetupExecutor<F>),
    Call(DeferralCallExecutor<F>),
    Output(DeferralOutputExecutor),
}

impl<F: VmField> VmExecutionExtension<F> for DeferralExtension {
    type Executor = DeferralExecutor<F>;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, DeferralExecutor<F>>,
    ) -> Result<(), ExecutorInventoryError> {
        let expected_def_vks_commit = self.expected_def_vks_commit.map(F::from_u32);
        let setup = DeferralSetupExecutor::new(
            DeferralSetupAdapterExecutor::new(self.native_start_ptr),
            expected_def_vks_commit,
        );
        inventory.add_executor(setup, [DeferralOpcode::SETUP.global_opcode()])?;

        let call = DeferralCallExecutor::new(
            DeferralCallAdapterExecutor::new(self.native_start_ptr),
            Arc::new(deferral_poseidon2_chip()),
        );
        inventory.add_executor(call, [DeferralOpcode::CALL.global_opcode()])?;

        inventory.add_executor(
            DeferralOutputExecutor::new(),
            [DeferralOpcode::OUTPUT.global_opcode()],
        )?;

        Ok(())
    }
}

impl<SC: StarkProtocolConfig> VmCircuitExtension<SC> for DeferralExtension
where
    Val<SC>: VmField,
{
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        let memory_bridge = inventory.system().port().memory_bridge;
        let execution_bridge = ExecutionBridge::new(
            inventory.system().port().execution_bus,
            inventory.system().port().program_bus,
        );

        let count_bus = DeferralCircuitCountBus::new(inventory.new_bus_idx());
        let poseidon2_bus_idx = inventory.new_bus_idx();
        let poseidon2_bus = DeferralPoseidon2Bus::new(poseidon2_bus_idx);

        let expected_def_vks_commit = self.expected_def_vks_commit.map(Val::<SC>::from_u32);
        inventory.add_air(DeferralSetupAir::new(
            DeferralSetupAdapterAir::new(execution_bridge, memory_bridge, self.native_start_ptr),
            DeferralSetupCoreAir::new(expected_def_vks_commit),
        ));

        inventory.add_air(DeferralCallAir::new(
            DeferralCallAdapterAir::new(execution_bridge, memory_bridge, self.native_start_ptr),
            DeferralCallCoreAir::new(count_bus, poseidon2_bus),
        ));

        inventory.add_air(DeferralOutputAir::new(
            execution_bridge,
            memory_bridge,
            count_bus,
            poseidon2_bus,
        ));

        inventory.add_air(DeferralCircuitCountAir::new(
            count_bus,
            self.num_deferral_circuits,
        ));
        inventory.add_air_ref(Arc::new(deferral_poseidon2_air(poseidon2_bus.0)));

        Ok(())
    }
}

pub struct DeferralCpuProverExt;

impl<SC, E, RA> VmProverExtension<E, RA, DeferralExtension> for DeferralCpuProverExt
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: VmField,
    SC::EF: Ord,
{
    fn extend_prover(
        &self,
        extension: &DeferralExtension,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let mem_helper = SharedMemoryHelper::new(range_checker, timestamp_max_bits);

        let count_chip = Arc::new(DeferralCircuitCountChip::new(
            extension.num_deferral_circuits,
        ));
        let poseidon2_chip = Arc::new(deferral_poseidon2_chip());

        inventory.next_air::<DeferralSetupAir<Val<SC>>>()?;
        inventory.add_executor_chip(DeferralSetupChip::new(
            DeferralSetupCoreFiller::new(DeferralSetupAdapterFiller::new()),
            mem_helper.clone(),
        ));

        inventory.next_air::<DeferralCallAir>()?;
        inventory.add_executor_chip(DeferralCallChip::new(
            DeferralCallCoreFiller::new(
                DeferralCallAdapterFiller::new(),
                count_chip.clone(),
                poseidon2_chip.clone(),
            ),
            mem_helper.clone(),
        ));

        inventory.next_air::<DeferralOutputAir>()?;
        inventory.add_executor_chip(DeferralOutputChip::new(
            DeferralOutputFiller::new(count_chip.clone(), poseidon2_chip.clone()),
            mem_helper,
        ));

        inventory.next_air::<DeferralCircuitCountAir>()?;
        inventory.add_periphery_chip(count_chip);

        inventory.next_air::<DeferralPoseidon2Air<Val<SC>>>()?;
        inventory.add_periphery_chip(poseidon2_chip);

        Ok(())
    }
}
