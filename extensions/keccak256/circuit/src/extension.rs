use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_stark_backend::p3_field::PrimeField32;
use axvm_circuit::arch::{
    SystemConfig, VmChipComplex, VmExtension, VmGenericConfig, VmInventory, VmInventoryBuilder,
    VmInventoryError,
};
use axvm_circuit_derive::{AnyEnum, InstructionExecutor};
use axvm_instructions::*;
use axvm_rv32im_circuit::{
    Rv32HintStore, Rv32I, Rv32ImConfig, Rv32ImExecutor, Rv32ImPeriphery, Rv32M,
};
use derive_more::derive::From;
use strum::IntoEnumIterator;

use crate::*;

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Keccak256Rv32Config {
    pub system: SystemConfig,
    pub rv32i: Rv32I,
    pub rv32m: Rv32M,
    pub io: Rv32HintStore,
    pub keccak: Keccak256Rv32,
}

impl Default for Keccak256Rv32Config {
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

impl<F: PrimeField32> VmGenericConfig<F> for Keccak256Rv32Config {
    type Executor = Keccak256Rv32Executor<F>;
    type Periphery = Keccak256Rv32Periphery<F>;

    fn system(&self) -> &SystemConfig {
        &self.system
    }

    fn create_chip_complex(
        &self,
    ) -> Result<VmChipComplex<F, Self::Executor, Self::Periphery>, VmInventoryError> {
        let base = Rv32ImConfig {
            system: self.system,
            base: self.rv32i,
            mul: self.rv32m,
            io: self.io,
        };
        let complex = base.create_chip_complex()?;
        let complex = complex.extend(&self.keccak)?;
        Ok(complex)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Keccak256Rv32;

#[derive(ChipUsageGetter, Chip, InstructionExecutor, From, AnyEnum)]
pub enum Keccak256Rv32Executor<F: PrimeField32> {
    #[any_enum]
    Rv32(Rv32ImExecutor<F>),
    Keccak256(KeccakVmChip<F>),
}

#[derive(From, ChipUsageGetter, Chip, AnyEnum)]
pub enum Keccak256Rv32Periphery<F: PrimeField32> {
    #[any_enum]
    Rv32Im(Rv32ImPeriphery<F>),
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
        let chip = builder.find_chip::<Arc<BitwiseOperationLookupChip<8>>>();
        let bitwise_lu_chip = Arc::clone(chip.first().unwrap());

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
