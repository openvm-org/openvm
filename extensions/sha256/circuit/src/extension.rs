use derive_more::derive::From;
use openvm_circuit::{
    arch::{SystemConfig, VmExtension, VmInventory, VmInventoryBuilder, VmInventoryError},
    system::phantom::PhantomChip,
};
use openvm_circuit_derive::{AnyEnum, InsExecutorE1, InstructionExecutor, VmConfig};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
};
use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
use openvm_instructions::*;
use openvm_rv32im_circuit::{
    Rv32I, Rv32IExecutor, Rv32IPeriphery, Rv32Io, Rv32IoExecutor, Rv32IoPeriphery, Rv32M,
    Rv32MExecutor, Rv32MPeriphery,
};
use openvm_sha256_transpiler::Rv32Sha256Opcode;
use openvm_stark_backend::p3_field::PrimeField32;
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use crate::*;

// TODO: this should be decided after e2 execution
const MAX_INS_CAPACITY: usize = 1 << 22;

#[derive(Clone, Debug, VmConfig, derive_new::new, Serialize, Deserialize)]
pub struct Sha256Rv32Config {
    #[system]
    pub system: SystemConfig,
    #[extension]
    pub rv32i: Rv32I,
    #[extension]
    pub rv32m: Rv32M,
    #[extension]
    pub io: Rv32Io,
    #[extension]
    pub sha256: Sha256,
}

impl Default for Sha256Rv32Config {
    fn default() -> Self {
        Self {
            system: SystemConfig::default().with_continuations(),
            rv32i: Rv32I,
            rv32m: Rv32M::default(),
            io: Rv32Io,
            sha256: Sha256,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Sha256;

#[derive(ChipUsageGetter, Chip, InstructionExecutor, From, AnyEnum, InsExecutorE1)]
pub enum Sha256Executor<F: PrimeField32> {
    Sha256(Sha256VmChip<F>),
}

#[derive(From, ChipUsageGetter, Chip, AnyEnum)]
pub enum Sha256Periphery<F: PrimeField32> {
    BitwiseOperationLookup(SharedBitwiseOperationLookupChip<8>),
    Phantom(PhantomChip<F>),
}

impl<F: PrimeField32> VmExtension<F> for Sha256 {
    type Executor = Sha256Executor<F>;
    type Periphery = Sha256Periphery<F>;

    fn build(
        &self,
        builder: &mut VmInventoryBuilder<F>,
    ) -> Result<VmInventory<Self::Executor, Self::Periphery>, VmInventoryError> {
        let mut inventory = VmInventory::new();
        let pointer_max_bits = builder.system_config().memory_config.pointer_max_bits;

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

        let sha256_chip = Sha256VmChip::new(
            Sha256VmAir::new(
                builder.system_port(),
                bitwise_lu_chip.bus(),
                pointer_max_bits,
                builder.new_bus_idx(),
            ),
            Sha256VmStep::new(
                bitwise_lu_chip.clone(),
                Rv32Sha256Opcode::CLASS_OFFSET,
                pointer_max_bits,
            ),
            MAX_INS_CAPACITY,
            builder.system_base().memory_controller.helper(),
        );
        inventory.add_executor(
            sha256_chip,
            Rv32Sha256Opcode::iter().map(|x| x.global_opcode()),
        )?;

        Ok(inventory)
    }
}
