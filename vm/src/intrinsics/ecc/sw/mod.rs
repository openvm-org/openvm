mod add_ne;
mod double;

pub use add_ne::*;
pub use double::*;

#[cfg(test)]
mod tests;

use ax_circuit_derive::{Chip, ChipUsageGetter};
use axvm_circuit_derive::InstructionExecutor;
use num_bigint_dig::BigUint;
use p3_field::PrimeField32;

use crate::{
    arch::{instructions::EccOpcode, VmChipWrapper},
    intrinsics::field_expression::FieldExpressionCoreChip,
    rv32im::adapters::Rv32VecHeapAdapterChip,
    system::memory::MemoryControllerRef,
};

#[derive(Chip, ChipUsageGetter, InstructionExecutor)]
pub struct EcAddNeChip<F: PrimeField32, const NUM_LIMBS: usize>(
    VmChipWrapper<
        F,
        Rv32VecHeapAdapterChip<F, 2, 2, 2, NUM_LIMBS, NUM_LIMBS>,
        FieldExpressionCoreChip,
    >,
);

impl<F: PrimeField32, const NUM_LIMBS: usize> EcAddNeChip<F, NUM_LIMBS> {
    pub fn new(
        adapter: Rv32VecHeapAdapterChip<F, 2, 2, 2, NUM_LIMBS, NUM_LIMBS>,
        memory_controller: MemoryControllerRef<F>,
        modulus: BigUint,
        limb_bits: usize,
        offset: usize,
    ) -> Self {
        let expr = ec_add_ne_expr(
            modulus,
            NUM_LIMBS,
            limb_bits,
            memory_controller.borrow().range_checker.bus(),
        );
        let core = FieldExpressionCoreChip::new(
            expr,
            offset,
            vec![EccOpcode::EC_ADD_NE as usize],
            memory_controller.borrow().range_checker.clone(),
            "EcAddNe",
        );
        Self(VmChipWrapper::new(adapter, core, memory_controller))
    }
}

#[derive(Chip, ChipUsageGetter, InstructionExecutor)]
pub struct EcDoubleChip<F: PrimeField32, const NUM_LIMBS: usize>(
    VmChipWrapper<
        F,
        Rv32VecHeapAdapterChip<F, 1, 2, 2, NUM_LIMBS, NUM_LIMBS>,
        FieldExpressionCoreChip,
    >,
);

impl<F: PrimeField32, const NUM_LIMBS: usize> EcDoubleChip<F, NUM_LIMBS> {
    pub fn new(
        adapter: Rv32VecHeapAdapterChip<F, 1, 2, 2, NUM_LIMBS, NUM_LIMBS>,
        memory_controller: MemoryControllerRef<F>,
        modulus: BigUint,
        limb_bits: usize,
        offset: usize,
    ) -> Self {
        let expr = ec_double_expr(
            modulus,
            NUM_LIMBS,
            limb_bits,
            memory_controller.borrow().range_checker.bus(),
        );
        let core = FieldExpressionCoreChip::new(
            expr,
            offset,
            vec![EccOpcode::EC_DOUBLE as usize],
            memory_controller.borrow().range_checker.clone(),
            "EcDouble",
        );
        Self(VmChipWrapper::new(adapter, core, memory_controller))
    }
}
