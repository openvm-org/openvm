use std::sync::Arc;

use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChip, range_tuple::RangeTupleCheckerChip,
};
use axvm_circuit_derive::{AnyEnum, InstructionExecutor};
use derive_more::derive::From;
use p3_field::PrimeField32;

use crate::{
    arch::{AnyEnum, VmExtension, VmInventory, VmInventoryBuilder},
    rv32im::*,
    system::phantom::PhantomChip,
};

/// RISC-V 32-bit Base (RV32I) Extension
pub struct Rv32I;
/// RISC-V 32-bit Multiplication Extension (RV32M) Extension
pub struct Rv32M;

/// RISC-V 32-bit Base (RV32I) Instruction Executors
#[derive(ChipUsageGetter, Chip, InstructionExecutor, From, AnyEnum)]
pub enum Rv32IExecutor<F: PrimeField32> {
    // Rv32 (for standard 32-bit integers):
    BaseAluRv32(Rv32BaseAluChip<F>),
    LessThanRv32(Rv32LessThanChip<F>),
    ShiftRv32(Rv32ShiftChip<F>),
    LoadStoreRv32(Rv32LoadStoreChip<F>),
    LoadSignExtendRv32(Rv32LoadSignExtendChip<F>),
    BranchEqualRv32(Rv32BranchEqualChip<F>),
    BranchLessThanRv32(Rv32BranchLessThanChip<F>),
    JalLuiRv32(Rv32JalLuiChip<F>),
    JalrRv32(Rv32JalrChip<F>),
    AuipcRv32(Rv32AuipcChip<F>),
}

/// RISC-V 32-bit Multiplication Extension (RV32M) Instruction Executors
#[derive(ChipUsageGetter, Chip, InstructionExecutor, From, AnyEnum)]
pub enum Rv32MExecutor<F: PrimeField32> {
    MultiplicationRv32(Rv32MultiplicationChip<F>),
    MultiplicationHighRv32(Rv32MulHChip<F>),
    DivRemRv32(Rv32DivRemChip<F>),
}

#[derive(From, ChipUsageGetter, Chip)]
pub enum Rv32Periphery<F: PrimeField32> {
    RangeTupleChecker(Arc<RangeTupleCheckerChip<2>>),
    BitwiseOperationLookup(Arc<BitwiseOperationLookupChip<8>>),
    // We put this only to get the <F> generic to work
    Phantom(PhantomChip<F>),
}

impl<F: PrimeField32> VmExtension<F> for Rv32I {
    type Executor = Rv32IExecutor<F>;
    type Periphery = Rv32Periphery<F>;

    fn build(
        &self,
        builder: &mut VmInventoryBuilder<F>,
    ) -> VmInventory<Rv32IExecutor<F>, Rv32Periphery<F>> {
        let mut inventory = VmInventory::new();
        let bitwise_lu_chip = if let Some(chip) = builder
            .find_chip::<Arc<BitwiseOperationLookupChip<8>>>()
            .first()
        {
            chip.clone()
        } else {
            let bitwise_lu_bus = builder.new_bus();
            let chip = BitwiseOperationLookupChip::new(bitwise_lu_bus);
            inventory.add_periphery_chip(chip.clone());
            chip
        };

        inventory
    }
}

// impl<F: PrimeField32> VmExtension<F> for Rv32M {
//     type Executor = Rv32MExecutor<F>;
//     type Periphery = Rv32Periphery<F>;
// }
