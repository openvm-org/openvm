use std::sync::Arc;

use derive_more::derive::From;
use openvm_circuit::{
    arch::{
        hasher::Hasher, AirInventory, AirInventoryError, ChipInventory, ChipInventoryError,
        ExecutionBridge, ExecutorInventoryBuilder, ExecutorInventoryError, InitFileGenerator,
        MatrixRecordArena, RowMajorMatrixArena, SystemConfig, VmBuilder, VmChipComplex,
        VmCircuitExtension, VmExecutionExtension, VmField, VmProverExtension,
    },
    system::{memory::SharedMemoryHelper, SystemChipInventory, SystemCpuBuilder, SystemExecutor},
};
use openvm_circuit_derive::{AnyEnum, Executor, MeteredExecutor, PreflightExecutor, VmConfig};
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::LocalOpcode;
use openvm_rv32im_circuit::{
    Rv32I, Rv32IExecutor, Rv32ImCpuProverExt, Rv32Io, Rv32IoExecutor, Rv32M, Rv32MExecutor,
};
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
    utils::{byte_commit_to_f, f_commit_to_bytes, COMMIT_NUM_BYTES},
    DeferralFn,
};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        mod cuda;
        pub use self::cuda::DeferralGpuProverExt as DeferralProverExt;
    } else {
        pub use self::DeferralCpuProverExt as DeferralProverExt;
    }
}

// SAFETY: These deferral AIRs must be at these indices within the extension
pub(crate) const POSEIDON2_AIR_IDX: usize = 1;
pub(crate) const CALL_AIR_IDX: usize = 3;
pub(crate) const OUTPUT_AIR_IDX: usize = 4;

// =================================== VM Extension Implementation =================================

#[derive(Clone, Default)]
pub struct DeferralExtension {
    pub fns: Vec<Arc<DeferralFn>>,
    pub expected_def_vks_commit: [u8; COMMIT_NUM_BYTES],
    pub native_start_ptr: u32,
}

impl DeferralExtension {
    pub fn new<F: VmField>(fns: Vec<Arc<DeferralFn>>, commits: &[[F; DIGEST_SIZE]]) -> Self {
        assert_eq!(fns.len(), commits.len());
        let mut expected_commit = [F::ZERO; DIGEST_SIZE];
        let hasher = deferral_poseidon2_chip();
        for commit in commits {
            expected_commit = hasher.compress(&expected_commit, commit);
        }
        Self {
            fns,
            expected_def_vks_commit: f_commit_to_bytes(&expected_commit),
            native_start_ptr: 0,
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
    Call(DeferralCallExecutor),
    Output(DeferralOutputExecutor),
}

impl<F: VmField> VmExecutionExtension<F> for DeferralExtension {
    type Executor = DeferralExecutor<F>;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, DeferralExecutor<F>>,
    ) -> Result<(), ExecutorInventoryError> {
        let setup = DeferralSetupExecutor::new(
            DeferralSetupAdapterExecutor::new(self.native_start_ptr),
            byte_commit_to_f(&self.expected_def_vks_commit.map(F::from_u8)),
        );
        inventory.add_executor(setup, [DeferralOpcode::SETUP.global_opcode()])?;

        let call = DeferralCallExecutor::new(
            DeferralCallAdapterExecutor::new(self.native_start_ptr),
            self.fns.clone(),
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
        let poseidon2_bus = DeferralPoseidon2Bus::new(inventory.new_bus_idx());
        let base_num_airs = inventory.num_airs();

        inventory.add_air(DeferralCircuitCountAir::new(count_bus, self.fns.len()));
        assert_eq!(inventory.num_airs() - base_num_airs, POSEIDON2_AIR_IDX);
        inventory.add_air_ref(Arc::new(deferral_poseidon2_air(poseidon2_bus.0)));

        inventory.add_air(DeferralSetupAir::new(
            DeferralSetupAdapterAir::new(execution_bridge, memory_bridge, self.native_start_ptr),
            DeferralSetupCoreAir::new(byte_commit_to_f(
                &self.expected_def_vks_commit.map(Val::<SC>::from_u8),
            )),
        ));
        assert_eq!(inventory.num_airs() - base_num_airs, CALL_AIR_IDX);
        inventory.add_air(DeferralCallAir::new(
            DeferralCallAdapterAir::new(execution_bridge, memory_bridge, self.native_start_ptr),
            DeferralCallCoreAir::new(count_bus, poseidon2_bus),
        ));
        assert_eq!(inventory.num_airs() - base_num_airs, OUTPUT_AIR_IDX);
        inventory.add_air(DeferralOutputAir::new(
            execution_bridge,
            memory_bridge,
            count_bus,
            poseidon2_bus,
        ));

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

        let count_chip = Arc::new(DeferralCircuitCountChip::new(extension.fns.len()));
        let poseidon2_chip = Arc::new(deferral_poseidon2_chip());

        inventory.next_air::<DeferralCircuitCountAir>()?;
        inventory.add_periphery_chip(count_chip.clone());

        inventory.next_air::<DeferralPoseidon2Air<Val<SC>>>()?;
        inventory.add_periphery_chip(poseidon2_chip.clone());

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

        Ok(())
    }
}

// =================================== VM Rv32 Config and Builder =================================

#[derive(Clone, VmConfig, Serialize, Deserialize)]
pub struct Rv32DeferralConfig {
    #[config(executor = "SystemExecutor<F>")]
    pub system: SystemConfig,
    #[extension]
    pub rv32i: Rv32I,
    #[extension]
    pub rv32m: Rv32M,
    #[extension]
    pub io: Rv32Io,
    #[serde(skip)]
    #[extension(executor = "DeferralExecutor<F>")]
    pub deferral: DeferralExtension,
}

impl InitFileGenerator for Rv32DeferralConfig {}

#[derive(Clone)]
pub struct DeferralCpuBuilder;

impl<SC, E> VmBuilder<E> for DeferralCpuBuilder
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: VmField,
    SC::EF: Ord,
{
    type VmConfig = Rv32DeferralConfig;
    type SystemChipInventory = SystemChipInventory<SC>;
    type RecordArena = MatrixRecordArena<Val<SC>>;

    fn create_chip_complex(
        &self,
        config: &Self::VmConfig,
        circuit: AirInventory<SC>,
    ) -> Result<
        VmChipComplex<SC, Self::RecordArena, E::PB, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let mut chip_complex =
            VmBuilder::<E>::create_chip_complex(&SystemCpuBuilder, &config.system, circuit)?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImCpuProverExt, &config.rv32i, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImCpuProverExt, &config.rv32m, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImCpuProverExt, &config.io, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(
            &DeferralCpuProverExt,
            &config.deferral,
            inventory,
        )?;
        Ok(chip_complex)
    }
}
