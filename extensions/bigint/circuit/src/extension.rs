use derive_more::derive::From;
use openvm_bigint_transpiler::{
    Rv32BaseAlu256Opcode, Rv32BranchEqual256Opcode, Rv32BranchLessThan256Opcode,
    Rv32LessThan256Opcode, Rv32Mul256Opcode, Rv32Shift256Opcode,
};
use openvm_circuit::{
    arch::{
        InitFileGenerator, SystemConfig, SystemPort, VmExtension, VmInventory, VmInventoryBuilder,
        VmInventoryError,
    },
    system::phantom::PhantomChip,
};
use openvm_circuit_derive::{AnyEnum, InstructionExecutor, VmConfig};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    range_tuple::{RangeTupleCheckerBus, SharedRangeTupleCheckerChip},
};
use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
use openvm_instructions::{program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_rv32im_circuit::{
    Rv32I, Rv32IExecutor, Rv32IPeriphery, Rv32Io, Rv32IoExecutor, Rv32IoPeriphery, Rv32M,
    Rv32MExecutor, Rv32MPeriphery,
};
use openvm_stark_backend::p3_field::PrimeField32;
use serde::{Deserialize, Serialize};

use crate::*;

#[derive(Clone, Debug, VmConfig, derive_new::new, Serialize, Deserialize)]
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

// Default implementation uses no init file
impl InitFileGenerator for Int256Rv32Config {}

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

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Int256 {
    #[serde(default = "default_range_tuple_checker_sizes")]
    pub range_tuple_checker_sizes: [u32; 2],
}

impl Default for Int256 {
    fn default() -> Self {
        Self {
            range_tuple_checker_sizes: default_range_tuple_checker_sizes(),
        }
    }
}

fn default_range_tuple_checker_sizes() -> [u32; 2] {
    [1 << 8, 32 * (1 << 8)]
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
    BitwiseOperationLookup(SharedBitwiseOperationLookupChip<8>),
    /// Only needed for multiplication extension
    RangeTupleChecker(SharedRangeTupleCheckerChip<2>),
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
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = builder.system_port();
        let range_checker_chip = builder.system_base().range_checker_chip.clone();
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

        let range_tuple_chip = if let Some(chip) = builder
            .find_chip::<SharedRangeTupleCheckerChip<2>>()
            .into_iter()
            .find(|c| {
                c.bus().sizes[0] >= self.range_tuple_checker_sizes[0]
                    && c.bus().sizes[1] >= self.range_tuple_checker_sizes[1]
            }) {
            chip.clone()
        } else {
            let range_tuple_bus =
                RangeTupleCheckerBus::new(builder.new_bus_idx(), self.range_tuple_checker_sizes);
            let chip = SharedRangeTupleCheckerChip::new(range_tuple_bus);
            inventory.add_periphery_chip(chip.clone());
            chip
        };

        let base_alu_chip = Rv32BaseAlu256Chip::new(
            Rv32HeapAdapterChip::new(
                execution_bus,
                program_bus,
                memory_bridge,
                address_bits,
                bitwise_lu_chip.clone(),
            ),
            BaseAluCoreChip::new(bitwise_lu_chip.clone(), Rv32BaseAlu256Opcode::CLASS_OFFSET),
            offline_memory.clone(),
        );
        inventory.add_executor(
            base_alu_chip,
            Rv32BaseAlu256Opcode::iter().map(|x| x.global_opcode()),
        )?;

        let less_than_chip = Rv32LessThan256Chip::new(
            Rv32HeapAdapterChip::new(
                execution_bus,
                program_bus,
                memory_bridge,
                address_bits,
                bitwise_lu_chip.clone(),
            ),
            LessThanCoreChip::new(bitwise_lu_chip.clone(), Rv32LessThan256Opcode::CLASS_OFFSET),
            offline_memory.clone(),
        );
        inventory.add_executor(
            less_than_chip,
            Rv32LessThan256Opcode::iter().map(|x| x.global_opcode()),
        )?;

        let branch_equal_chip = Rv32BranchEqual256Chip::new(
            Rv32HeapBranchAdapterChip::new(
                execution_bus,
                program_bus,
                memory_bridge,
                address_bits,
                bitwise_lu_chip.clone(),
            ),
            BranchEqualCoreChip::new(Rv32BranchEqual256Opcode::CLASS_OFFSET, DEFAULT_PC_STEP),
            offline_memory.clone(),
        );
        inventory.add_executor(
            branch_equal_chip,
            Rv32BranchEqual256Opcode::iter().map(|x| x.global_opcode()),
        )?;

        let branch_less_than_chip = Rv32BranchLessThan256Chip::new(
            Rv32HeapBranchAdapterChip::new(
                execution_bus,
                program_bus,
                memory_bridge,
                address_bits,
                bitwise_lu_chip.clone(),
            ),
            BranchLessThanCoreChip::new(
                bitwise_lu_chip.clone(),
                Rv32BranchLessThan256Opcode::CLASS_OFFSET,
            ),
            offline_memory.clone(),
        );
        inventory.add_executor(
            branch_less_than_chip,
            Rv32BranchLessThan256Opcode::iter().map(|x| x.global_opcode()),
        )?;

        let multiplication_chip = Rv32Multiplication256Chip::new(
            Rv32HeapAdapterChip::new(
                execution_bus,
                program_bus,
                memory_bridge,
                address_bits,
                bitwise_lu_chip.clone(),
            ),
            MultiplicationCoreChip::new(range_tuple_chip, Rv32Mul256Opcode::CLASS_OFFSET),
            offline_memory.clone(),
        );
        inventory.add_executor(
            multiplication_chip,
            Rv32Mul256Opcode::iter().map(|x| x.global_opcode()),
        )?;

        let shift_chip = Rv32Shift256Chip::new(
            Rv32HeapAdapterChip::new(
                execution_bus,
                program_bus,
                memory_bridge,
                address_bits,
                bitwise_lu_chip.clone(),
            ),
            ShiftCoreChip::new(
                bitwise_lu_chip.clone(),
                range_checker_chip,
                Rv32Shift256Opcode::CLASS_OFFSET,
            ),
            offline_memory.clone(),
        );
        inventory.add_executor(
            shift_chip,
            Rv32Shift256Opcode::iter().map(|x| x.global_opcode()),
        )?;

        Ok(inventory)
    }
}
