use std::{
    cell::RefCell,
    rc::Rc,
    sync::{Arc, Mutex},
};

use openvm_algebra_circuit::Fp2;
use openvm_circuit::{arch::VmChipWrapper, system::memory::OfflineMemory};
use openvm_circuit_derive::InstructionExecutor;
use openvm_circuit_primitives::var_range::{
    SharedVariableRangeCheckerChip, VariableRangeCheckerBus,
};
use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
use openvm_mod_circuit_builder::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreChip,
};
use openvm_pairing_transpiler::PairingOpcode;
use openvm_rv32_adapters::Rv32VecHeapTwoReadsAdapterChip;
use openvm_stark_backend::p3_field::PrimeField32;

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
        config: ExprBuilderConfig,
        offset: usize,
        range_checker: SharedVariableRangeCheckerChip,
        offline_memory: Arc<Mutex<OfflineMemory<F>>>,
    ) -> Self {
        let expr = evaluate_line_expr(config, range_checker.bus());
        let core = FieldExpressionCoreChip::new(
            expr,
            offset,
            vec![PairingOpcode::EVALUATE_LINE as usize],
            vec![],
            range_checker,
            "EvaluateLine",
            false,
        );
        Self(VmChipWrapper::new(adapter, core, offline_memory))
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
