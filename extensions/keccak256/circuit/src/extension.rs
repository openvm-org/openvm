use derive_more::derive::From;
use openvm_circuit::{
    arch::{
        InitFileGenerator, SystemConfig, SystemPort, VmExtension, VmInventory, VmInventoryBuilder,
        VmInventoryError,
    },
    system::phantom::PhantomChip,
};
use openvm_circuit_derive::{AnyEnum, InstructionExecutor, VmConfig};
use openvm_circuit_primitives::bitwise_op_lookup::BitwiseOperationLookupBus;
use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
use openvm_instructions::*;
use openvm_rv32im_circuit::{
    Rv32I, Rv32IExecutor, Rv32IPeriphery, Rv32Io, Rv32IoExecutor, Rv32IoPeriphery, Rv32M,
    Rv32MExecutor, Rv32MPeriphery,
};
use openvm_stark_backend::p3_field::PrimeField32;
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use crate::*;

#[derive(Clone, Debug, VmConfig, derive_new::new, Serialize, Deserialize)]
pub struct Keccak256Rv32Config {
    #[system]
    pub system: SystemConfig,
    #[extension]
    pub rv32i: Rv32I,
    #[extension]
    pub rv32m: Rv32M,
    #[extension]
    pub io: Rv32Io,
    #[extension]
    pub keccak: Keccak256,
}

impl Default for Keccak256Rv32Config {
    fn default() -> Self {
        Self {
            system: SystemConfig::default().with_continuations(),
            rv32i: Rv32I,
            rv32m: Rv32M::default(),
            io: Rv32Io,
            keccak: Keccak256,
        }
    }
}

// Default implementation uses no init file
impl InitFileGenerator for Keccak256Rv32Config {}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Keccak256;

#[derive(ChipUsageGetter, Chip, InstructionExecutor, From, AnyEnum)]
pub enum Keccak256Executor<F: PrimeField32> {
    Keccak256(KeccakVmChip<F>),
}

#[derive(From, ChipUsageGetter, Chip, AnyEnum)]
pub enum Keccak256Periphery<F: PrimeField32> {
    BitwiseOperationLookup(SharedBitwiseOperationLookupChip<8>),
    Phantom(PhantomChip<F>),
}

impl<F: PrimeField32> VmExtension<F> for Keccak256 {
    type Executor = Keccak256Executor<F>;
    type Periphery = Keccak256Periphery<F>;

    fn build(
        &self,
        builder: &mut VmInventoryBuilder<F>,
    ) -> Result<VmInventory<Self::Executor, Self::Periphery>, VmInventoryError> {
        let mut inventory = VmInventory::new();
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = builder.system_port();
        let bitwise_lu_chip = if let Some(&chip) = builder
            .find_chip::<SharedBitwiseOperationLookupChip<8>>()
            .first()
        {
            chip.clone()
        } else {
            let bitwise_lu_bus = BitwiseOperationLookupBus::new(builder.new_bus_idx());
            let chip = SharedBitwiseOperationLookupChip::new(bitwise_lu_bus);
            inventory.add_periphery_chip(chip.clone());
            chip
        };
        let offline_memory = builder.system_base().offline_memory();
        let address_bits = builder.system_config().memory_config.pointer_max_bits;

        let keccak_chip = KeccakVmChip::new(
            execution_bus,
            program_bus,
            memory_bridge,
            address_bits,
            bitwise_lu_chip,
            Rv32KeccakOpcode::CLASS_OFFSET,
            offline_memory,
        );
        inventory.add_executor(
            keccak_chip,
            Rv32KeccakOpcode::iter().map(|x| x.global_opcode()),
        )?;

        Ok(inventory)
    }
}
