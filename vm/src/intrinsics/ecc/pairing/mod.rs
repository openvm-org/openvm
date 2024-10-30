mod miller_double_step;

use ax_circuit_derive::{Chip, ChipUsageGetter};
use axvm_circuit_derive::InstructionExecutor;
use miller_double_step::*;
use num_bigint_dig::BigUint;
use p3_field::PrimeField32;

use crate::{
    arch::{instructions::PairingOpcode, VmChipWrapper},
    intrinsics::field_expression::FieldExpressionCoreChip,
    rv32im::adapters::Rv32VecHeapAdapterChip,
    system::memory::MemoryControllerRef,
};

// Input: EcPoint<Fp2>: 4 field elements
// Output: (EcPoint<Fp2>, Fp2, Fp2) -> 8 field elements
#[derive(Chip, ChipUsageGetter, InstructionExecutor)]
pub struct MillerDoubleStepChip<
    F: PrimeField32,
    const INPUT_BLOCKS: usize,
    const OUTPUT_BLOCKS: usize,
    const BLOCK_SIZE: usize,
>(
    VmChipWrapper<
        F,
        Rv32VecHeapAdapterChip<F, 1, INPUT_BLOCKS, OUTPUT_BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        FieldExpressionCoreChip,
    >,
);

impl<
        F: PrimeField32,
        const INPUT_BLOCKS: usize,
        const OUTPUT_BLOCKS: usize,
        const BLOCK_SIZE: usize,
    > MillerDoubleStepChip<F, INPUT_BLOCKS, OUTPUT_BLOCKS, BLOCK_SIZE>
{
    pub fn new(
        adapter: Rv32VecHeapAdapterChip<F, 1, INPUT_BLOCKS, OUTPUT_BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        memory_controller: MemoryControllerRef<F>,
        modulus: BigUint,
        num_limbs: usize,
        limb_bits: usize,
        offset: usize,
    ) -> Self {
        let expr = miller_double_step_expr(
            modulus,
            num_limbs,
            limb_bits,
            memory_controller.borrow().range_checker.bus(),
        );
        let core = FieldExpressionCoreChip::new(
            expr,
            offset,
            vec![PairingOpcode::MILLER_DOUBLE_STEP as usize],
            memory_controller.borrow().range_checker.clone(),
            "MillerDoubleStep",
        );
        Self(VmChipWrapper::new(adapter, core, memory_controller))
    }
}
