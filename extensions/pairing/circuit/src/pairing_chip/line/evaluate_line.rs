use std::{cell::RefCell, rc::Rc};

use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_circuit_primitives::var_range::VariableRangeCheckerBus;
use ax_mod_circuit_builder::{ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreChip};
use ax_stark_backend::p3_field::PrimeField32;
use axvm_algebra_circuit::Fp2;
use axvm_circuit::{arch::VmChipWrapper, system::memory::MemoryControllerRef};
use axvm_circuit_derive::InstructionExecutor;
use axvm_pairing_transpiler::PairingOpcode;
use axvm_rv32_adapters::Rv32VecHeapTwoReadsAdapterChip;

// Input: UnevaluatedLine<Fp2>, (Fp, Fp)
// Output: EvaluatedLine<Fp2>
#[derive(Chip, ChipUsageGetter, InstructionExecutor)]
pub struct EvaluateLineChip<
    F: PrimeField32,
    const INPUT_BLOCKS1: usize,
    const INPUT_BLOCKS2: usize,
    const OUTPUT_BLOCKS: usize,
    const BLOCK_SIZE: usize,
>(
    pub  VmChipWrapper<
        F,
        Rv32VecHeapTwoReadsAdapterChip<
            F,
            INPUT_BLOCKS1,
            INPUT_BLOCKS2,
            OUTPUT_BLOCKS,
            BLOCK_SIZE,
            BLOCK_SIZE,
        >,
        FieldExpressionCoreChip,
    >,
);

impl<
        F: PrimeField32,
        const INPUT_BLOCKS1: usize,
        const INPUT_BLOCKS2: usize,
        const OUTPUT_BLOCKS: usize,
        const BLOCK_SIZE: usize,
    > EvaluateLineChip<F, INPUT_BLOCKS1, INPUT_BLOCKS2, OUTPUT_BLOCKS, BLOCK_SIZE>
{
    pub fn new(
        adapter: Rv32VecHeapTwoReadsAdapterChip<
            F,
            INPUT_BLOCKS1,
            INPUT_BLOCKS2,
            OUTPUT_BLOCKS,
            BLOCK_SIZE,
            BLOCK_SIZE,
        >,
        memory_controller: MemoryControllerRef<F>,
        config: ExprBuilderConfig,
        offset: usize,
    ) -> Self {
        let expr = evaluate_line_expr(config, memory_controller.borrow().range_checker.bus());
        let core = FieldExpressionCoreChip::new(
            expr,
            offset,
            vec![PairingOpcode::EVALUATE_LINE as usize],
            vec![],
            memory_controller.borrow().range_checker.clone(),
            "EvaluateLine",
        );
        Self(VmChipWrapper::new(adapter, core, memory_controller))
    }
}

pub fn evaluate_line_expr(
    config: ExprBuilderConfig,
    range_bus: VariableRangeCheckerBus,
) -> FieldExpr {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let mut uneval_b = Fp2::new(builder.clone());
    let mut uneval_c = Fp2::new(builder.clone());

    let mut x_over_y = ExprBuilder::new_input(builder.clone());
    let mut y_inv = ExprBuilder::new_input(builder.clone());

    let mut b = uneval_b.scalar_mul(&mut x_over_y);
    let mut c = uneval_c.scalar_mul(&mut y_inv);
    b.save_output();
    c.save_output();

    let builder = builder.borrow().clone();
    FieldExpr::new(builder, range_bus, false)
}
