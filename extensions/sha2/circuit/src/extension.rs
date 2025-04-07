use derive_more::derive::From;
use openvm_circuit::{
    arch::{
        InitFileGenerator, SystemConfig, VmExtension, VmInventory, VmInventoryBuilder,
        VmInventoryError,
    },
    system::phantom::PhantomChip,
};
use openvm_circuit_derive::{AnyEnum, InstructionExecutor, VmConfig};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
};
use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
use openvm_instructions::*;
use openvm_rv32im_circuit::{
    Rv32I, Rv32IExecutor, Rv32IPeriphery, Rv32Io, Rv32IoExecutor, Rv32IoPeriphery, Rv32M,
    Rv32MExecutor, Rv32MPeriphery,
};
use openvm_sha2_air::{Sha256Config, Sha384Config, Sha512Config};
use openvm_sha2_transpiler::Rv32Sha2Opcode;
use openvm_stark_backend::p3_field::PrimeField32;
use serde::{Deserialize, Serialize};

use crate::*;

#[derive(Clone, Debug, VmConfig, derive_new::new, Serialize, Deserialize)]
pub struct Sha2Rv32Config {
    #[system]
    pub system: SystemConfig,
    #[extension]
    pub rv32i: Rv32I,
    #[extension]
    pub rv32m: Rv32M,
    #[extension]
    pub io: Rv32Io,
    #[extension]
    pub sha2: Sha2,
}

impl Default for Sha2Rv32Config {
    fn default() -> Self {
        Self {
            system: SystemConfig::default().with_continuations(),
            rv32i: Rv32I,
            rv32m: Rv32M::default(),
            io: Rv32Io,
            sha2: Sha2,
        }
    }
}

// Default implementation uses no init file
impl InitFileGenerator for Sha2Rv32Config {}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Sha2;

#[derive(ChipUsageGetter, Chip, InstructionExecutor, From, AnyEnum)]
pub enum Sha2Executor<F: PrimeField32> {
    Sha256(Sha2VmChip<F, Sha256Config>),
    Sha512(Sha2VmChip<F, Sha512Config>),
    Sha384(Sha2VmChip<F, Sha384Config>),
}

#[derive(From, ChipUsageGetter, Chip, AnyEnum)]
pub enum Sha2Periphery<F: PrimeField32> {
    BitwiseOperationLookup(SharedBitwiseOperationLookupChip<8>),
    Phantom(PhantomChip<F>),
}

impl<F: PrimeField32> VmExtension<F> for Sha2 {
    type Executor = Sha2Executor<F>;
    type Periphery = Sha2Periphery<F>;

    fn build(
        &self,
        builder: &mut VmInventoryBuilder<F>,
    ) -> Result<VmInventory<Self::Executor, Self::Periphery>, VmInventoryError> {
        let mut inventory = VmInventory::new();
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

        let sha256_chip = Sha2VmChip::<F, Sha256Config>::new(
            builder.system_port(),
            builder.system_config().memory_config.pointer_max_bits,
            bitwise_lu_chip.clone(),
            builder.new_bus_idx(),
            builder.system_base().offline_memory(),
        );
        inventory.add_executor(sha256_chip, vec![Rv32Sha2Opcode::SHA256.global_opcode()])?;

        let sha512_chip = Sha2VmChip::<F, Sha512Config>::new(
            builder.system_port(),
            builder.system_config().memory_config.pointer_max_bits,
            bitwise_lu_chip.clone(),
            builder.new_bus_idx(),
            builder.system_base().offline_memory(),
        );
        inventory.add_executor(sha512_chip, vec![Rv32Sha2Opcode::SHA512.global_opcode()])?;

        let sha384_chip = Sha2VmChip::<F, Sha384Config>::new(
            builder.system_port(),
            builder.system_config().memory_config.pointer_max_bits,
            bitwise_lu_chip,
            builder.new_bus_idx(),
            builder.system_base().offline_memory(),
        );
        inventory.add_executor(sha384_chip, vec![Rv32Sha2Opcode::SHA384.global_opcode()])?;

        Ok(inventory)
    }
}
