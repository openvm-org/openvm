use std::{cell::RefCell, rc::Rc};

use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_circuit_primitives::var_range::VariableRangeCheckerBus;
use ax_mod_circuit_builder::{ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreChip};
use ax_stark_backend::p3_field::PrimeField32;
use axvm_algebra_circuit::Fp2;
use axvm_circuit::{arch::VmChipWrapper, system::memory::MemoryControllerRef};
use axvm_circuit_derive::InstructionExecutor;
use axvm_instructions::PairingOpcode;
use axvm_rv32_adapters::Rv32VecHeapTwoReadsAdapterChip;

use crate::Fp12;

// Input: Fp12 (12 field elements), [Fp2; 5] (5 x 2 field elements)
// Output: Fp12 (12 field elements)
#[derive(Chip, ChipUsageGetter, InstructionExecutor)]
pub struct EcLineMulBy01234Chip<
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
    > EcLineMulBy01234Chip<F, INPUT_BLOCKS1, INPUT_BLOCKS2, OUTPUT_BLOCKS, BLOCK_SIZE>
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
        xi: [isize; 2],
        offset: usize,
    ) -> Self {
        assert!(
            xi[0].unsigned_abs() < 1 << config.limb_bits,
            "expect xi to be small"
        ); // not a hard rule, but we expect xi to be small
        assert!(
            xi[1].unsigned_abs() < 1 << config.limb_bits,
            "expect xi to be small"
        );
        let expr = mul_by_01234_expr(config, memory_controller.borrow().range_checker.bus(), xi);
        let core = FieldExpressionCoreChip::new(
            expr,
            offset,
            vec![PairingOpcode::MUL_BY_01234 as usize],
            vec![],
            memory_controller.borrow().range_checker.clone(),
            "MulBy01234",
        );
        Self(VmChipWrapper::new(adapter, core, memory_controller))
    }
}

pub fn mul_by_01234_expr(
    config: ExprBuilderConfig,
    range_bus: VariableRangeCheckerBus,
    xi: [isize; 2],
) -> FieldExpr {
    config.check_valid();
    let builder = ExprBuilder::new(config.clone(), range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let mut f = Fp12::new(builder.clone());
    let mut x0 = Fp2::new(builder.clone());
    let mut x1 = Fp2::new(builder.clone());
    let mut x2 = Fp2::new(builder.clone());
    let mut x3 = Fp2::new(builder.clone());
    let mut x4 = Fp2::new(builder.clone());

    let mut r = f.mul_by_01234(&mut x0, &mut x1, &mut x2, &mut x3, &mut x4, xi);
    r.save_output();

    let builder = builder.borrow().clone();
    FieldExpr::new(builder, range_bus, false)
}
