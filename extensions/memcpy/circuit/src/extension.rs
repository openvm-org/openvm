use std::{result::Result, sync::Arc};

use bus::MemcpyBus;
use derive_more::derive::From;
use openvm_circuit::{
    arch::{
        AirInventory, AirInventoryError, ChipInventory, ChipInventoryError, ExecutionBridge,
        ExecutorInventoryBuilder, ExecutorInventoryError, RowMajorMatrixArena, VmCircuitExtension,
        VmExecutionExtension, VmProverExtension,
    },
    system::{memory::SharedMemoryHelper, SystemPort},
};
use openvm_circuit_derive::{AnyEnum, Executor, MeteredExecutor, PreflightExecutor};
use openvm_instructions::*;
use openvm_memcpy_transpiler::Rv32MemcpyOpcode;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    engine::StarkEngine,
    p3_field::PrimeField32,
    prover::cpu::{CpuBackend, CpuDevice},
};
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use crate::*;

// =================================== VM Extension Implementation =================================
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Memcpy;

#[derive(Clone, From, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
pub enum MemcpyExecutor {
    MemcpyLoop(MemcpyIterExecutor),
}

impl<F> VmExecutionExtension<F> for Memcpy {
    type Executor = MemcpyExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, MemcpyExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        let memcpy_iter = MemcpyIterExecutor::new(Rv32MemcpyOpcode::CLASS_OFFSET);

        inventory.add_executor(
            memcpy_iter,
            Rv32MemcpyOpcode::iter().map(|x| x.global_opcode()),
        )?;

        Ok(())
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for Memcpy {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = inventory.system().port();

        let execution_bridge = ExecutionBridge::new(execution_bus, program_bus);
        let range_bus = inventory.range_checker().bus;
        let pointer_max_bits = inventory.pointer_max_bits();

        let memcpy_bus = MemcpyBus::new(inventory.new_bus_idx());

        let memcpy_loop = MemcpyLoopAir::new(
            memory_bridge,
            execution_bridge,
            range_bus,
            memcpy_bus,
            pointer_max_bits,
            Rv32MemcpyOpcode::CLASS_OFFSET,
        );
        inventory.add_air(memcpy_loop);

        let memcpy_iter =
            MemcpyIterAir::new(memory_bridge, range_bus, memcpy_bus, pointer_max_bits);
        inventory.add_air(memcpy_iter);

        Ok(())
    }
}

pub struct MemcpyCpuProverExt;
// This implementation is specific to CpuBackend because the lookup chips (VariableRangeChecker)
// are specific to CpuBackend.
impl<E, SC, RA> VmProverExtension<E, RA, Memcpy> for MemcpyCpuProverExt
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: PrimeField32,
{
    fn extend_prover(
        &self,
        _: &Memcpy,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let pointer_max_bits = inventory.airs().pointer_max_bits();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);
        let range_bus = inventory.airs().range_checker().bus;
        let memcpy_bus = inventory
            .airs()
            .find_air::<MemcpyLoopAir>()
            .next()
            .unwrap()
            .memcpy_bus;

        let memcpy_loop_chip = Arc::new(MemcpyLoopChip::new(
            inventory.airs().system().port(),
            range_bus,
            memcpy_bus,
            Rv32MemcpyOpcode::CLASS_OFFSET,
            pointer_max_bits,
            range_checker.clone(),
        ));

        let memcpy_iter_chip = MemcpyIterChip::new(
            MemcpyIterFiller::new(
                pointer_max_bits,
                range_checker.clone(),
                memcpy_loop_chip.clone(),
            ),
            mem_helper.clone(),
        );
        // Add MemcpyLoop chip
        inventory.next_air::<MemcpyLoopAir>()?;
        inventory.add_periphery_chip(memcpy_loop_chip);

        // Add MemcpyIter chip
        inventory.next_air::<MemcpyIterAir>()?;
        inventory.add_executor_chip(memcpy_iter_chip);

        Ok(())
    }
}
