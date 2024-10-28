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

// LANE_SIZE: how many cells do we read at a time, must be a power of 2.
// NUM_LANES: how many lanes do we need to represent one field element.
// TWO_NUM_LANES: how many lanes do we need to represent one of the input/output, which is a EcPoint of 2 field elements.
// For example, for bls12_381, LANE_SIZE = 16, NUM_LANES = 3, and TWO_NUM_LANES = 6.
// For secp256k1, LANE_SIZE = 32, NUM_LANES = 1, and TWO_NUM_LANES = 2.
#[derive(Chip, ChipUsageGetter, InstructionExecutor)]
pub struct EcAddNeChip<F: PrimeField32, const TWO_NUM_LANES: usize, const LANE_SIZE: usize>(
    VmChipWrapper<
        F,
        Rv32VecHeapAdapterChip<F, 2, TWO_NUM_LANES, TWO_NUM_LANES, LANE_SIZE, LANE_SIZE>,
        FieldExpressionCoreChip,
    >,
);

impl<F: PrimeField32, const TWO_NUM_LANES: usize, const LANE_SIZE: usize>
    EcAddNeChip<F, TWO_NUM_LANES, LANE_SIZE>
{
    pub fn new(
        adapter: Rv32VecHeapAdapterChip<F, 2, TWO_NUM_LANES, TWO_NUM_LANES, LANE_SIZE, LANE_SIZE>,
        memory_controller: MemoryControllerRef<F>,
        modulus: BigUint,
        num_limbs: usize,
        limb_bits: usize,
        offset: usize,
    ) -> Self {
        let expr = ec_add_ne_expr(
            modulus,
            num_limbs,
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
pub struct EcDoubleChip<F: PrimeField32, const TWO_NUM_LANES: usize, const LANE_SIZE: usize>(
    VmChipWrapper<
        F,
        Rv32VecHeapAdapterChip<F, 1, TWO_NUM_LANES, TWO_NUM_LANES, LANE_SIZE, LANE_SIZE>,
        FieldExpressionCoreChip,
    >,
);

impl<F: PrimeField32, const TWO_NUM_LANES: usize, const LANE_SIZE: usize>
    EcDoubleChip<F, TWO_NUM_LANES, LANE_SIZE>
{
    pub fn new(
        adapter: Rv32VecHeapAdapterChip<F, 1, TWO_NUM_LANES, TWO_NUM_LANES, LANE_SIZE, LANE_SIZE>,
        memory_controller: MemoryControllerRef<F>,
        modulus: BigUint,
        num_limbs: usize,
        limb_bits: usize,
        offset: usize,
    ) -> Self {
        let expr = ec_double_expr(
            modulus,
            num_limbs,
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
