use std::sync::Arc;

use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, BitwiseOperationLookupChip},
    range_tuple::{RangeTupleCheckerBus, RangeTupleCheckerChip},
};
use ax_stark_backend::p3_field::PrimeField32;
use axvm_circuit::{
    arch::{
        SystemConfig, SystemExecutor, SystemPeriphery, VmChipComplex, VmExtension, VmGenericConfig,
        VmInventory, VmInventoryBuilder, VmInventoryError,
    },
    system::phantom::PhantomChip,
};
use axvm_circuit_derive::{AnyEnum, InstructionExecutor, VmGenericConfig};
use axvm_instructions::*;
use axvm_rv32im_circuit::{
    Rv32I, Rv32IExecutor, Rv32IPeriphery, Rv32Io, Rv32IoExecutor, Rv32IoPeriphery, Rv32M,
    Rv32MExecutor, Rv32MPeriphery,
};
use derive_more::derive::From;
use program::DEFAULT_PC_STEP;

use crate::*;

#[derive(Clone, Debug, VmGenericConfig, derive_new::new)]
pub struct Int256Rv32Config {
    #[system]
    pub system: SystemConfig,
    #[extension]
    pub rv32i: Rv32I,
    #[extension]
    pub rv32m: Rv32M,
    #[extension]
    pub io: Rv32Io,
    #[extension]
    pub bigint: Int256,
}

impl Default for Int256Rv32Config {
    fn default() -> Self {
        Self {
            system: SystemConfig::default().with_continuations(),
            rv32i: Rv32I,
            rv32m: Rv32M::default(),
            io: Rv32Io,
            bigint: Int256::default(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Int256 {
    pub range_tuple_checker_sizes: [u32; 2],
}

impl Default for Int256 {
    fn default() -> Self {
        Self {
            range_tuple_checker_sizes: [1 << 8, 32 * (1 << 8)],
        }
    }
}

#[derive(ChipUsageGetter, Chip, InstructionExecutor, From, AnyEnum)]
pub enum Int256Executor<F: PrimeField32> {
    BaseAlu256(Rv32BaseAlu256Chip<F>),
    LessThan256(Rv32LessThan256Chip<F>),
    BranchEqual256(Rv32BranchEqual256Chip<F>),
    BranchLessThan256(Rv32BranchLessThan256Chip<F>),
    Multiplication256(Rv32Multiplication256Chip<F>),
    Shift256(Rv32Shift256Chip<F>),
}

#[derive(From, ChipUsageGetter, Chip, AnyEnum)]
pub enum Int256Periphery<F: PrimeField32> {
    BitwiseOperationLookup(Arc<BitwiseOperationLookupChip<8>>),
    /// Only needed for multiplication extension
    RangeTupleChecker(Arc<RangeTupleCheckerChip<2>>),
    Phantom(PhantomChip<F>),
}

impl<F: PrimeField32> VmExtension<F> for Int256 {
    type Executor = Int256Executor<F>;
    type Periphery = Int256Periphery<F>;

    fn build(
        &self,
        builder: &mut VmInventoryBuilder<F>,
    ) -> Result<VmInventory<Self::Executor, Self::Periphery>, VmInventoryError> {
        let mut inventory = VmInventory::new();
        let execution_bus = builder.system_base().execution_bus();
        let program_bus = builder.system_base().program_bus();
        let memory_controller = builder.memory_controller().clone();
        let range_checker_chip = builder.system_base().range_checker_chip.clone();

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

        let range_tuple_chip = if let Some(chip) = builder
            .find_chip::<Arc<RangeTupleCheckerChip<2>>>()
            .into_iter()
            .find(|c| {
                c.bus().sizes[0] >= self.range_tuple_checker_sizes[0]
                    && c.bus().sizes[1] >= self.range_tuple_checker_sizes[1]
            }) {
            chip.clone()
        } else {
            let range_tuple_bus =
                RangeTupleCheckerBus::new(builder.new_bus_idx(), self.range_tuple_checker_sizes);
            let chip = Arc::new(RangeTupleCheckerChip::new(range_tuple_bus));
            inventory.add_periphery_chip(chip.clone());
            chip
        };

        let base_alu_chip = Rv32BaseAlu256Chip::new(
            Rv32HeapAdapterChip::new(
                execution_bus,
                program_bus,
                memory_controller.clone(),
                bitwise_lu_chip.clone(),
            ),
            BaseAluCoreChip::new(
                bitwise_lu_chip.clone(),
                Rv32BaseAlu256Opcode::default_offset(),
            ),
            memory_controller.clone(),
        );
        inventory.add_executor(
            base_alu_chip,
            Rv32BaseAlu256Opcode::iter().map(|opcode| opcode.with_default_offset()),
        )?;

        let less_than_chip = Rv32LessThan256Chip::new(
            Rv32HeapAdapterChip::new(
                execution_bus,
                program_bus,
                memory_controller.clone(),
                bitwise_lu_chip.clone(),
            ),
            LessThanCoreChip::new(
                bitwise_lu_chip.clone(),
                Rv32LessThan256Opcode::default_offset(),
            ),
            memory_controller.clone(),
        );
        inventory.add_executor(
            less_than_chip,
            Rv32LessThan256Opcode::iter().map(|opcode| opcode.with_default_offset()),
        )?;

        let branch_equal_chip = Rv32BranchEqual256Chip::new(
            Rv32HeapBranchAdapterChip::new(
                execution_bus,
                program_bus,
                memory_controller.clone(),
                bitwise_lu_chip.clone(),
            ),
            BranchEqualCoreChip::new(Rv32BranchEqual256Opcode::default_offset(), DEFAULT_PC_STEP),
            memory_controller.clone(),
        );
        inventory.add_executor(
            branch_equal_chip,
            Rv32BranchEqual256Opcode::iter().map(|opcode| opcode.with_default_offset()),
        )?;

        let branch_less_than_chip = Rv32BranchLessThan256Chip::new(
            Rv32HeapBranchAdapterChip::new(
                execution_bus,
                program_bus,
                memory_controller.clone(),
                bitwise_lu_chip.clone(),
            ),
            BranchLessThanCoreChip::new(
                bitwise_lu_chip.clone(),
                Rv32LessThan256Opcode::default_offset(),
            ),
            memory_controller.clone(),
        );
        inventory.add_executor(
            branch_less_than_chip,
            Rv32BranchLessThan256Opcode::iter().map(|opcode| opcode.with_default_offset()),
        )?;

        let multiplication_chip = Rv32Multiplication256Chip::new(
            Rv32HeapAdapterChip::new(
                execution_bus,
                program_bus,
                memory_controller.clone(),
                bitwise_lu_chip.clone(),
            ),
            MultiplicationCoreChip::new(range_tuple_chip, Rv32Mul256Opcode::default_offset()),
            memory_controller.clone(),
        );
        inventory.add_executor(
            multiplication_chip,
            Rv32Mul256Opcode::iter().map(|opcode| opcode.with_default_offset()),
        )?;

        let shift_chip = Rv32Shift256Chip::new(
            Rv32HeapAdapterChip::new(
                execution_bus,
                program_bus,
                memory_controller.clone(),
                bitwise_lu_chip.clone(),
            ),
            ShiftCoreChip::new(
                bitwise_lu_chip.clone(),
                range_checker_chip,
                Rv32Shift256Opcode::default_offset(),
            ),
            memory_controller.clone(),
        );
        inventory.add_executor(
            shift_chip,
            Rv32Shift256Opcode::iter().map(|opcode| opcode.with_default_offset()),
        )?;

        Ok(inventory)
    }
}
