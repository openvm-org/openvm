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
pub struct Int256Rv32Config {
    pub system: SystemConfig,
    pub rv32i: Rv32I,
    pub rv32m: Rv32M,
    pub io: Rv32HintStore,
    pub bigint: Int256Rv32,
}

impl Default for Int256Rv32Config {
    fn default() -> Self {
        Self {
            system: SystemConfig::default(),
            rv32i: Rv32I::default(),
            rv32m: Rv32M::default(),
            io: Rv32HintStore::default(),
            bigint: Int256Rv32,
        }
    }
}

impl<F: PrimeField32> VmGenericConfig<F> for Int256Rv32Config {
    type Executor = Int256Rv32Executor<F>;
    type Periphery = Int256Rv32Periphery<F>;

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
        let complex = complex.extend(&self.bigint)?;
        Ok(complex)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Int256Rv32;

#[derive(ChipUsageGetter, Chip, InstructionExecutor, From, AnyEnum)]
pub enum Keccak256Rv32Executor<F: PrimeField32> {
    #[any_enum]
    Rv32(Rv32ImExecutor<F>),
    BaseAlu256(Rv32BaseAlu256Chip<F>),
    LessThan256(Rv32LessThan256Chip<F>),
    BranchEqual256(Rv32BranchEqual256Chip<F>),
    BranchLessThan256(Rv32BranchLessThan256Chip<F>),
    Multiplication256(Rv32Multiplication256Chip<F>),
    Shift256(Rv32Shift256Chip<F>),
}

#[derive(ChipUsageGetter, Chip, AnyEnum)]
pub enum Int256Rv32Periphery<F: PrimeField32> {
    #[any_enum]
    Rv32(Rv32ImPeriphery<F>),
}

impl<F: PrimeField32> VmExtension<F> for Int256Rv32 {
    type Executor = Int256Rv32Executor<F>;
    type Periphery = Int256Rv32Periphery<F>;

    fn build(
        &self,
        builder: &mut VmInventoryBuilder<F>,
    ) -> Result<VmInventory<Self::Executor, Self::Periphery>, VmInventoryError> {
        let mut inventory = VmInventory::new();
        let execution_bus = builder.system_base().execution_bus();
        let program_bus = builder.system_base().program_bus();
        let memory_controller = builder.memory_controller().clone();

        let bitwise_lu_chip = builder.find_chip::<Arc<BitwiseOperationLookupChip<8>>>();
        let bitwise_lu_chip = Arc::clone(bitwise_lu_chip.first().unwrap());

        // TODO[yi]: Check that this is the correct range checker for int256
        let range_checker_chip = builder.find_chip::<Arc<RangeTupleCheckerChip<2>>>();
        let range_checker_chip = Arc::clone(range_checker.first().unwrap());

        let base_alu_chip = Rv32BaseAlu256Chip::new(
            Rv32HeapAdapterChip::new(
                execution_bus,
                program_bus,
                memory_controller,
                bitwise_lu_chip,
            ),
            BaseAluCoreChip::new(bitwise_lu_chip, Rv32BaseAluOpcode::default_offset()),
            memory_controller,
        );
        inventory.add_executor(
            base_alu_chip,
            Rv32BaseAluOpcode::iter().map(|opcode| opcode.with_default_offset()),
        )?;

        let less_than_chip = Rv32LessThan256Chip::new(
            Rv32HeapAdapterChip::new(
                execution_bus,
                program_bus,
                memory_controller,
                bitwise_lu_chip,
            ),
            LessThanCoreChip::new(bitwise_lu_chip, Rv32LessThanOpcode::default_offset()),
            memory_controller,
        );
        inventory.add_executor(
            less_than_chip,
            Rv32LessThanOpcode::iter().map(|opcode| opcode.with_default_offset()),
        )?;

        let branch_equal_chip = Rv32BranchEqual256Chip::new(
            Rv32HeapBranchAdapterChip::new(
                execution_bus,
                program_bus,
                memory_controller,
                bitwise_lu_chip,
            ),
            BranchEqualCoreChip::new(bitwise_lu_chip, DEFAULT_PC_STEP),
            memory_controller,
        );
        inventory.add_executor(
            branch_equal_chip,
            Rv32BranchEqualOpcode::iter().map(|opcode| opcode.with_default_offset()),
        )?;

        let branch_less_than_chip = Rv32BranchLessThan256Chip::new(
            Rv32HeapBranchAdapterChip::new(
                execution_bus,
                program_bus,
                memory_controller,
                bitwise_lu_chip,
            ),
            BranchLessThanCoreChip::new(bitwise_lu_chip, Rv32LessThanOpcode::default_offset()),
            memory_controller,
        );
        inventory.add_executor(
            branch_less_than_chip,
            Rv32BranchLessThanOpcode::iter().map(|opcode| opcode.with_default_offset()),
        )?;

        let multiplication_chip = Rv32Multiplication256Chip::new(
            Rv32HeapAdapterChip::new(
                execution_bus,
                program_bus,
                memory_controller,
                bitwise_lu_chip,
            ),
            MultiplicationCoreChip::new(
                range_checker_chip,
                Rv32MultiplicationOpcode::default_offset(),
            ),
            memory_controller,
        );
        inventory.add_executor(
            multiplication_chip,
            Rv32MultiplicationOpcode::iter().map(|opcode| opcode.with_default_offset()),
        )?;

        let shift_chip = Rv32Shift256Chip::new(
            Rv32HeapAdapterChip::new(
                execution_bus,
                program_bus,
                memory_controller,
                bitwise_lu_chip,
            ),
            ShiftCoreChip::new(
                bitwise_lu_chip,
                range_checker_chip,
                Rv32ShiftOpcode::default_offset(),
            ),
            memory_controller,
        );
        inventory.add_executor(
            shift_chip,
            Rv32ShiftOpcode::iter().map(|opcode| opcode.with_default_offset()),
        )?;

        Ok(inventory)
    }
}
