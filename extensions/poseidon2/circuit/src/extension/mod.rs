use std::{result::Result, sync::Arc};

use derive_more::derive::From;
use openvm_circuit::{
    arch::{
        AirInventory, AirInventoryError, ChipInventory, ChipInventoryError, ExecutionBridge,
        ExecutorInventoryBuilder, ExecutorInventoryError, InitFileGenerator, MatrixRecordArena,
        RowMajorMatrixArena, SystemConfig, VmBuilder, VmChipComplex, VmCircuitExtension,
        VmExecutionExtension, VmField, VmProverExtension,
    },
    system::{
        memory::SharedMemoryHelper, SystemChipInventory, SystemCpuBuilder, SystemExecutor,
        SystemPort,
    },
};
use openvm_circuit_derive::{AnyEnum, Executor, MeteredExecutor, PreflightExecutor, VmConfig};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_cpu_backend::{CpuBackend, CpuDevice};
use openvm_instructions::*;
use openvm_poseidon2_air::Poseidon2Config;
use openvm_poseidon2_transpiler::Poseidon2Opcode;
use openvm_rv32im_circuit::{
    Rv32I, Rv32IExecutor, Rv32ImCpuProverExt, Rv32Io, Rv32IoExecutor, Rv32M, Rv32MExecutor,
};
use openvm_stark_backend::{
    interaction::LookupBus, p3_field::PrimeField32, StarkEngine, StarkProtocolConfig, Val,
};
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use crate::{
    periphery::Poseidon2PeripheryAir,
    permute::{Poseidon2PermuteAir, Poseidon2PermuteChip, Poseidon2PermuteExecutor},
};

#[derive(Clone, Debug, VmConfig, derive_new::new, Serialize, Deserialize)]
pub struct Poseidon2Rv32Config {
    #[config(executor = "SystemExecutor<F>")]
    pub system: SystemConfig,
    #[extension]
    pub rv32i: Rv32I,
    #[extension]
    pub rv32m: Rv32M,
    #[extension]
    pub io: Rv32Io,
    #[extension]
    pub poseidon2: Poseidon2,
}

impl Default for Poseidon2Rv32Config {
    fn default() -> Self {
        Self {
            system: SystemConfig::default(),
            rv32i: Rv32I,
            rv32m: Rv32M::default(),
            io: Rv32Io,
            poseidon2: Poseidon2,
        }
    }
}

// Default implementation uses no init file
impl InitFileGenerator for Poseidon2Rv32Config {}

#[derive(Clone)]
pub struct Poseidon2Rv32CpuBuilder;

impl<SC, E> VmBuilder<E> for Poseidon2Rv32CpuBuilder
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: VmField,
    SC::EF: Ord,
{
    type VmConfig = Poseidon2Rv32Config;
    type SystemChipInventory = SystemChipInventory<SC>;
    type RecordArena = MatrixRecordArena<Val<SC>>;

    fn create_chip_complex(
        &self,
        config: &Poseidon2Rv32Config,
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
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImCpuProverExt, &config.rv32i, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImCpuProverExt, &config.rv32m, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImCpuProverExt, &config.io, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(
            &Poseidon2CpuProverExt,
            &config.poseidon2,
            inventory,
        )?;
        Ok(chip_complex)
    }
}

// =================================== VM Extension Implementation =================================
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Poseidon2;

#[derive(Clone, Copy, From, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
#[cfg_attr(
    feature = "aot",
    derive(
        openvm_circuit_derive::AotExecutor,
        openvm_circuit_derive::AotMeteredExecutor
    )
)]
pub enum Poseidon2Executor {
    Permute(Poseidon2PermuteExecutor),
}

impl<F> VmExecutionExtension<F> for Poseidon2 {
    type Executor = Poseidon2Executor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, Poseidon2Executor>,
    ) -> Result<(), ExecutorInventoryError> {
        let pointer_max_bits = inventory.pointer_max_bits();

        let permute_executor =
            Poseidon2PermuteExecutor::new(Poseidon2Opcode::CLASS_OFFSET, pointer_max_bits);
        inventory.add_executor(
            permute_executor,
            Poseidon2Opcode::iter().map(|x| x.global_opcode()),
        )?;

        Ok(())
    }
}

impl<SC: StarkProtocolConfig> VmCircuitExtension<SC> for Poseidon2
where
    SC::F: PrimeField32,
{
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = inventory.system().port();

        let exec_bridge = ExecutionBridge::new(execution_bus, program_bus);
        let pointer_max_bits = inventory.pointer_max_bits();

        let bitwise_lu = {
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

        let poseidon2_bus = LookupBus::new(inventory.new_bus_idx());
        // The periphery AIR is added before the adapter AIR (and its chip before the adapter chip
        // in `extend_prover`) so that the VK AIR index of the periphery AIR is `op_air_idx + 1`,
        // which the metered executor relies on for height tracking (see `execute_e2_impl`).
        let periphery_air =
            Poseidon2PeripheryAir::<Val<SC>>::new(Poseidon2Config::default(), poseidon2_bus);
        inventory.add_air(periphery_air);

        let op_air = Poseidon2PermuteAir::new(
            exec_bridge,
            memory_bridge,
            bitwise_lu,
            poseidon2_bus,
            pointer_max_bits,
            Poseidon2Opcode::CLASS_OFFSET,
        );
        inventory.add_air(op_air);

        Ok(())
    }
}

pub struct Poseidon2CpuProverExt;
// This implementation is specific to CpuBackend because the lookup chips (VariableRangeChecker,
// BitwiseOperationLookupChip) are specific to CpuBackend.
impl<SC, E, RA> VmProverExtension<E, RA, Poseidon2> for Poseidon2CpuProverExt
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: VmField,
    SC::EF: Ord,
{
    fn extend_prover(
        &self,
        _: &Poseidon2,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);
        let pointer_max_bits = inventory.airs().pointer_max_bits();

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

        inventory.next_air::<Poseidon2PeripheryAir<Val<SC>>>()?;
        // WARNING: the periphery chip must be added _before_ the adapter chip so that its tracegen
        // is done _last_: chips are trace-generated in reverse insertion order, and the adapter
        // chip's tracegen calls `perm_and_record` to record the states the periphery chip then
        // needs to generate its own trace.
        let periphery = Arc::new(crate::periphery::poseidon2_periphery_chip::<Val<SC>>());
        inventory.add_periphery_chip(periphery.clone());

        inventory.next_air::<Poseidon2PermuteAir>()?;
        let op_chip =
            Poseidon2PermuteChip::new(bitwise_lu, pointer_max_bits, mem_helper, periphery);
        inventory.add_executor_chip(op_chip);

        Ok(())
    }
}
