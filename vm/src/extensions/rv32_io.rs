use std::sync::Arc;

use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, BitwiseOperationLookupChip,
};
use axvm_circuit_derive::{AnyEnum, InstructionExecutor};
use axvm_instructions::*;
use derive_more::derive::From;
use p3_field::PrimeField32;
use strum::IntoEnumIterator;

use super::rv32im::Rv32Periphery;
use crate::{
    arch::{VmExtension, VmInventory, VmInventoryBuilder, VmInventoryError},
    rv32im::{adapters::*, *},
};

/// RISC-V HintStore Extension for handling IO
#[derive(Clone, Copy, Debug, Default)]
pub struct Rv32HintStore;

/// RISC-V 32-bit HintStore Instruction Executors
#[derive(ChipUsageGetter, Chip, InstructionExecutor, From, AnyEnum)]
pub enum Rv32HintStoreExecutor<F: PrimeField32> {
    HintStore(Rv32HintStoreChip<F>),
}

impl<F: PrimeField32> VmExtension<F> for Rv32HintStore {
    type Executor = Rv32HintStoreExecutor<F>;
    type Periphery = Rv32Periphery<F>;

    fn build(
        &self,
        builder: &mut VmInventoryBuilder<F>,
    ) -> Result<VmInventory<Self::Executor, Self::Periphery>, VmInventoryError> {
        let mut inventory = VmInventory::new();
        let execution_bus = builder.system_base().execution_bus();
        let program_bus = builder.system_base().program_bus();
        let memory_controller = builder.memory_controller().clone();
        let range_checker = builder.system_base().range_checker_chip.clone();
        let bitwise_lu_chip = if let Some(chip) = builder
            .find_chip::<Arc<BitwiseOperationLookupChip<8>>>()
            .first()
        {
            Arc::clone(chip)
        } else {
            let bitwise_lu_bus = BitwiseOperationLookupBus::new(builder.new_bus_idx());
            let chip = Arc::new(BitwiseOperationLookupChip::new(bitwise_lu_bus));
            inventory.add_periphery_chip(chip.clone());
            chip
        };

        let mut hintstore_chip = Rv32HintStoreChip::new(
            Rv32HintStoreAdapterChip::new(
                execution_bus,
                program_bus,
                memory_controller.clone(),
                range_checker.clone(),
            ),
            Rv32HintStoreCoreChip::new(
                bitwise_lu_chip.clone(),
                Rv32HintStoreOpcode::default_offset(),
            ),
            memory_controller.clone(),
        );
        hintstore_chip.core.set_streams(builder.streams().clone());

        inventory.add_executor(
            hintstore_chip,
            Rv32HintStoreOpcode::iter().map(|x| x.with_default_offset()),
        )?;

        Ok(inventory)
    }
}
