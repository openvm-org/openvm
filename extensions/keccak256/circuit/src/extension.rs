use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_circuit_primitives::bitwise_op_lookup::BitwiseOperationLookupBus;
use axvm_circuit::{
    arch::{
        SystemConfig, SystemExecutor, SystemPeriphery, VmChipComplex, VmExtension, VmGenericConfig,
        VmInventory, VmInventoryBuilder, VmInventoryError,
    },
    system::phantom::PhantomChip,
};
use axvm_circuit_derive::{AnyEnum, InstructionExecutor};
use axvm_instructions::*;
use axvm_rv32im_circuit::{Rv32HintStore, Rv32I, Rv32M};
use derive_more::derive::From;
use strum::IntoEnumIterator;

use crate::*;

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Keccak256Config {
    pub system: SystemConfig,
    pub rv32i: Rv32I,
    pub rv32m: Rv32M,
    pub io: Rv32HintStore,
    pub keccak: Keccak256Rv32,
}

impl Default for Keccak256Config {
    fn default() -> Self {
        Self {
            system: SystemConfig::default().with_continuations(),
            rv32i: Rv32I::default(),
            rv32m: Rv32M::default(),
            io: Rv32HintStore::default(),
            keccak: Keccak256Rv32,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Keccak256Rv32;

#[derive(ChipUsageGetter, Chip, InstructionExecutor, From, AnyEnum)]
pub enum Keccak256Rv32Executor<F: PrimeField32> {
    Keccak256(KeccakVmChip<F>),
}

#[derive(From, ChipUsageGetter, Chip, AnyEnum)]
pub enum Keccak256Rv32Periphery<F: PrimeField32> {
    BitwiseOperationLookup(Arc<BitwiseOperationLookupChip<8>>),
    // We put this only to get the <F> generic to work
    Phantom(PhantomChip<F>),
}

impl<F: PrimeField32> VmExtension<F> for Keccak256Rv32 {
    type Executor = Keccak256Rv32Executor<F>;
    type Periphery = Keccak256Rv32Periphery<F>;

    fn build(
        &self,
        builder: &mut VmInventoryBuilder<F>,
    ) -> Result<VmInventory<Self::Executor, Self::Periphery>, VmInventoryError> {
        let mut inventory = VmInventory::new();
        let execution_bus = builder.system_base().execution_bus();
        let program_bus = builder.system_base().program_bus();
        let memory_controller = builder.memory_controller().clone();
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

        let keccak_chip = KeccakVmChip::new(
            execution_bus,
            program_bus,
            memory_controller,
            bitwise_lu_chip,
            Rv32KeccakOpcode::default_offset(),
        );
        inventory.add_executor(
            keccak_chip,
            Rv32KeccakOpcode::iter().map(|opcode| opcode.with_default_offset()),
        )?;

        Ok(inventory)
    }
}
