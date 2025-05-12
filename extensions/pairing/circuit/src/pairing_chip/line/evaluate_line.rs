use std::{cell::RefCell, rc::Rc};

use openvm_algebra_circuit::Fp2;
use openvm_circuit::{
    arch::ExecutionBridge,
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_derive::{InsExecutorE1, InstructionExecutor};
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
};
use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_mod_circuit_builder::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreAir,
};
use openvm_pairing_transpiler::PairingOpcode;
use openvm_rv32_adapters::{Rv32VecHeapTwoReadsAdapterAir, Rv32VecHeapTwoReadsAdapterStep};
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{PairingTwoReadsAir, PairingTwoReadsChip, PairingTwoReadsStep};

// Input: UnevaluatedLine<Fp2>, (Fp, Fp)
// Output: EvaluatedLine<Fp2>
#[derive(Chip, ChipUsageGetter, InstructionExecutor, InsExecutorE1)]
pub struct EvaluateLineChip<
    F: PrimeField32,
    const INPUT_BLOCKS1: usize,
    const INPUT_BLOCKS2: usize,
    const OUTPUT_BLOCKS: usize,
    const BLOCK_SIZE: usize,
>(pub PairingTwoReadsChip<F, INPUT_BLOCKS1, INPUT_BLOCKS2, OUTPUT_BLOCKS, BLOCK_SIZE>);

impl<
        F: PrimeField32,
        const INPUT_BLOCKS1: usize,
        const INPUT_BLOCKS2: usize,
        const OUTPUT_BLOCKS: usize,
        const BLOCK_SIZE: usize,
    > EvaluateLineChip<F, INPUT_BLOCKS1, INPUT_BLOCKS2, OUTPUT_BLOCKS, BLOCK_SIZE>
{
    pub fn new(
        execution_bridge: ExecutionBridge,
        memory_bridge: MemoryBridge,
        mem_helper: SharedMemoryHelper<F>,
        pointer_max_bits: usize,
        config: ExprBuilderConfig,
        offset: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
        range_checker: SharedVariableRangeCheckerChip,
        height: usize,
    ) -> Self {
        let expr = evaluate_line_expr(config, range_checker.bus());
        let local_opcode_idx = vec![PairingOpcode::EVALUATE_LINE as usize];

        let air = PairingTwoReadsAir::new(
            Rv32VecHeapTwoReadsAdapterAir::new(
                execution_bridge,
                memory_bridge,
                bitwise_lookup_chip.bus(),
                pointer_max_bits,
            ),
            FieldExpressionCoreAir::new(expr.clone(), offset, local_opcode_idx.clone(), vec![]),
        );

        let step = PairingTwoReadsStep::new(
            Rv32VecHeapTwoReadsAdapterStep::new(pointer_max_bits, bitwise_lookup_chip),
            expr,
            offset,
            local_opcode_idx,
            vec![],
            range_checker,
            "EvaluateLine",
            false,
        );
        Self(PairingTwoReadsChip::new(air, step, height, mem_helper))
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
