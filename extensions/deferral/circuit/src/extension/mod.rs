use std::sync::Arc;

use derive_more::derive::From;
use openvm_circuit::{
    arch::{
        AirInventory, AirInventoryError, ChipInventory, ChipInventoryError, ExecutionBridge,
        ExecutorInventoryBuilder, ExecutorInventoryError, InitFileGenerator, MatrixRecordArena,
        RowMajorMatrixArena, SystemConfig, VmBuilder, VmChipComplex, VmCircuitExtension,
        VmExecutionExtension, VmField, VmProverExtension,
    },
    system::{memory::SharedMemoryHelper, SystemChipInventory, SystemCpuBuilder, SystemExecutor},
};
use openvm_circuit_derive::{AnyEnum, Executor, MeteredExecutor, PreflightExecutor, VmConfig};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_cpu_backend::{CpuBackend, CpuDevice};
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::LocalOpcode;
use openvm_riscv_circuit::{
    Rv64I, Rv64IExecutor, Rv64ImCpuProverExt, Rv64Io, Rv64IoExecutor, Rv64M, Rv64MExecutor,
};
use openvm_stark_backend::{StarkEngine, StarkProtocolConfig, Val};
use serde::{Deserialize, Serialize};

use crate::{
    call::{
        DeferralCallAdapterAir, DeferralCallAdapterExecutor, DeferralCallAdapterFiller,
        DeferralCallAir, DeferralCallChip, DeferralCallCoreAir, DeferralCallCoreFiller,
        DeferralCallExecutor,
    },
    count::{DeferralCircuitCountAir, DeferralCircuitCountBus, DeferralCircuitCountChip},
    output::{DeferralOutputAir, DeferralOutputChip, DeferralOutputExecutor, DeferralOutputFiller},
    poseidon2::{
        deferral_poseidon2_air, deferral_poseidon2_chip, DeferralPoseidon2Air, DeferralPoseidon2Bus,
    },
    utils::COMMIT_NUM_BYTES,
    DeferralFn,
};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        mod cuda;
        pub use self::cuda::DeferralGpuProverExt as DeferralProverExt;
        pub use self::cuda::Rv64DeferralGpuBuilder as Rv64DeferralBuilder;

    } else {
        pub use self::DeferralCpuProverExt as DeferralProverExt;
        pub use self::Rv64DeferralCpuBuilder as Rv64DeferralBuilder;
    }
}

// SAFETY: These deferral AIRs must be at these indices within the extension
pub(crate) const POSEIDON2_AIR_REL_IDX: usize = 1;
pub(crate) const CALL_AIR_REL_IDX: usize = 2;
pub(crate) const OUTPUT_AIR_REL_IDX: usize = 3;

// =================================== VM Extension Implementation =================================

#[derive(Clone, Debug, Default, Serialize, Deserialize, derive_new::new)]
pub struct DeferralExtension {
    #[serde(skip)]
    pub fns: Vec<Arc<DeferralFn>>,
    pub def_circuit_commits: Vec<[u8; COMMIT_NUM_BYTES]>,
}

#[derive(Clone, From, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
#[cfg_attr(
    feature = "aot",
    derive(
        openvm_circuit_derive::AotExecutor,
        openvm_circuit_derive::AotMeteredExecutor
    )
)]
pub enum DeferralExecutor {
    Call(DeferralCallExecutor),
    Output(DeferralOutputExecutor),
}

impl<F: VmField> VmExecutionExtension<F> for DeferralExtension {
    type Executor = DeferralExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, DeferralExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        let call = DeferralCallExecutor::new(DeferralCallAdapterExecutor, self.fns.clone());
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
        let bitwise_bus = {
            let existing_air = inventory.find_air::<BitwiseOperationLookupAir<8>>().next();
            if let Some(air) = existing_air {
                air.bus
            } else {
                let bus = BitwiseOperationLookupBus::new(inventory.new_bus_idx());
                let air = BitwiseOperationLookupAir::<8>::new(bus);
                inventory.add_air(air);
                air.bus
            }
        };

        let base_num_airs = inventory.num_airs();
        let address_bits = inventory.pointer_max_bits();

        inventory.add_air(DeferralCircuitCountAir::new(count_bus, self.fns.len()));

        assert_eq!(inventory.num_airs() - base_num_airs, POSEIDON2_AIR_REL_IDX);
        inventory.add_air_ref(Arc::new(deferral_poseidon2_air(poseidon2_bus.0)));

        assert_eq!(inventory.num_airs() - base_num_airs, CALL_AIR_REL_IDX);
        inventory.add_air(DeferralCallAir::new(
            DeferralCallAdapterAir::new(execution_bridge, memory_bridge, bitwise_bus, address_bits),
            DeferralCallCoreAir::new(count_bus, poseidon2_bus, bitwise_bus),
        ));

        assert_eq!(inventory.num_airs() - base_num_airs, OUTPUT_AIR_REL_IDX);
        inventory.add_air(DeferralOutputAir::new(
            execution_bridge,
            memory_bridge,
            count_bus,
            poseidon2_bus,
            bitwise_bus,
            address_bits,
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
        let address_bits = inventory.airs().pointer_max_bits();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);
        let bitwise_lu = {
            let existing_chip = inventory
                .find_chip::<SharedBitwiseOperationLookupChip<8>>()
                .next();
            if let Some(chip) = existing_chip {
                chip.clone()
            } else {
                let air: &BitwiseOperationLookupAir<8> = inventory.next_air()?;
                let chip = Arc::new(BitwiseOperationLookupChip::new(air.bus));
                inventory.add_periphery_chip(chip.clone());
                chip
            }
        };
        let count_chip = Arc::new(DeferralCircuitCountChip::new(extension.fns.len()));
        let poseidon2_chip = Arc::new(deferral_poseidon2_chip());

        inventory.next_air::<DeferralCircuitCountAir>()?;
        inventory.add_periphery_chip(count_chip.clone());

        inventory.next_air::<DeferralPoseidon2Air<Val<SC>>>()?;
        inventory.add_periphery_chip(poseidon2_chip.clone());

        inventory.next_air::<DeferralCallAir>()?;
        inventory.add_executor_chip(DeferralCallChip::new(
            DeferralCallCoreFiller::new(
                DeferralCallAdapterFiller::new(bitwise_lu.clone(), address_bits),
                count_chip.clone(),
                poseidon2_chip.clone(),
                bitwise_lu.clone(),
                address_bits,
            ),
            mem_helper.clone(),
        ));

        inventory.next_air::<DeferralOutputAir>()?;
        inventory.add_executor_chip(DeferralOutputChip::new(
            DeferralOutputFiller::new(
                count_chip.clone(),
                poseidon2_chip.clone(),
                bitwise_lu,
                address_bits,
            ),
            mem_helper,
        ));

        Ok(())
    }
}

// =================================== VM Rv64 Config and Builder =================================

#[derive(Clone, VmConfig, Serialize, Deserialize)]
pub struct Rv64DeferralConfig {
    #[config(executor = "SystemExecutor<F>")]
    pub system: SystemConfig,
    #[extension]
    pub rv64i: Rv64I,
    #[extension]
    pub rv64m: Rv64M,
    #[extension]
    pub io: Rv64Io,
    #[serde(skip)]
    #[extension(executor = "DeferralExecutor")]
    pub deferral: DeferralExtension,
}

impl InitFileGenerator for Rv64DeferralConfig {}

#[derive(Clone)]
pub struct Rv64DeferralCpuBuilder;

impl<SC, E> VmBuilder<E> for Rv64DeferralCpuBuilder
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: VmField,
    SC::EF: Ord,
{
    type VmConfig = Rv64DeferralConfig;
    type SystemChipInventory = SystemChipInventory<SC>;
    type RecordArena = MatrixRecordArena<Val<SC>>;

    fn create_chip_complex(
        &self,
        config: &Self::VmConfig,
        circuit: AirInventory<SC>,
        device_ctx: &openvm_stark_backend::EngineDeviceCtx<E>,
    ) -> Result<
        VmChipComplex<SC, Self::RecordArena, E::PB, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let mut chip_complex = VmBuilder::<E>::create_chip_complex(
            &SystemCpuBuilder,
            &config.system,
            circuit,
            device_ctx,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(&Rv64ImCpuProverExt, &config.rv64i, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv64ImCpuProverExt, &config.rv64m, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv64ImCpuProverExt, &config.io, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(
            &DeferralCpuProverExt,
            &config.deferral,
            inventory,
        )?;
        Ok(chip_complex)
    }
}
