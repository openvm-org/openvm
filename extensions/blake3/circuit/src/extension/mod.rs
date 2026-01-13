use std::{result::Result, sync::Arc};

use derive_more::derive::From;
use openvm_blake3_transpiler::Rv32Blake3Opcode;
use openvm_circuit::{
    arch::{
        AirInventory, AirInventoryError, ChipInventory, ChipInventoryError, ExecutionBridge,
        ExecutorInventoryBuilder, ExecutorInventoryError, RowMajorMatrixArena, VmCircuitExtension,
        VmExecutionExtension, VmProverExtension,
    },
    system::{memory::SharedMemoryHelper, SystemPort},
};
use openvm_circuit_derive::{AnyEnum, Executor, MeteredExecutor, PreflightExecutor};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_instructions::*;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_field::PrimeField32,
    prover::cpu::{CpuBackend, CpuDevice},
};
use openvm_stark_sdk::engine::StarkEngine;
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use crate::{Blake3VmAir, Blake3VmChip, Blake3VmExecutor, Blake3VmFiller};

// =================================== VM Extension Definition =================================

/// The BLAKE3 extension marker type
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Blake3;

/// Executor enum for the BLAKE3 extension
#[derive(Clone, Copy, From, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
#[cfg_attr(
    feature = "aot",
    derive(
        openvm_circuit_derive::AotExecutor,
        openvm_circuit_derive::AotMeteredExecutor
    )
)]
pub enum Blake3Executor {
    Blake3(Blake3VmExecutor),
}

// =================================== VmExecutionExtension =================================

impl<F> VmExecutionExtension<F> for Blake3 {
    type Executor = Blake3Executor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, Blake3Executor>,
    ) -> Result<(), ExecutorInventoryError> {
        let pointer_max_bits = inventory.pointer_max_bits();
        let blake3_executor =
            Blake3VmExecutor::new(Rv32Blake3Opcode::CLASS_OFFSET, pointer_max_bits);
        inventory.add_executor(
            blake3_executor,
            Rv32Blake3Opcode::iter().map(|x| x.global_opcode()),
        )?;

        Ok(())
    }
}

// =================================== VmCircuitExtension =================================

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for Blake3 {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = inventory.system().port();

        let exec_bridge = ExecutionBridge::new(execution_bus, program_bus);
        let pointer_max_bits = inventory.pointer_max_bits();

        // Get or create the shared bitwise lookup AIR
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

        // Create and register the BLAKE3 AIR
        let blake3 = Blake3VmAir::new(
            exec_bridge,
            memory_bridge,
            bitwise_lu,
            pointer_max_bits,
            Rv32Blake3Opcode::CLASS_OFFSET,
        );
        inventory.add_air(blake3);

        Ok(())
    }
}

// =================================== VmProverExtension =================================

pub struct Blake3CpuProverExt;

impl<E, SC, RA> VmProverExtension<E, RA, Blake3> for Blake3CpuProverExt
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: PrimeField32,
{
    fn extend_prover(
        &self,
        _: &Blake3,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);
        let pointer_max_bits = inventory.airs().pointer_max_bits();

        // Get or create the shared bitwise lookup chip
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

        // Create and register the BLAKE3 chip
        inventory.next_air::<Blake3VmAir>()?;
        let blake3 = Blake3VmChip::new(
            Blake3VmFiller::new(bitwise_lu, pointer_max_bits),
            mem_helper,
        );
        inventory.add_executor_chip(blake3);

        Ok(())
    }
}
