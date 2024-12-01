use std::{cell::RefCell, rc::Rc};

use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_circuit_primitives::var_range::VariableRangeCheckerBus;
use ax_mod_circuit_builder::{ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreChip};
use ax_stark_backend::p3_field::PrimeField32;
use axvm_algebra_circuit::Fp2;
use axvm_circuit::{
    arch::{instructions::PairingOpcode, VmChipWrapper},
    system::memory::MemoryControllerRef,
};
use axvm_circuit_derive::InstructionExecutor;
use axvm_rv32_adapters::Rv32VecHeapAdapterChip;

// Input: line0.b, line0.c, line1.b, line1.c <Fp2>: 2 x 4 field elements
// Output: 5 Fp2 coefficients -> 10 field elements
#[derive(Chip, ChipUsageGetter, InstructionExecutor)]
pub struct EcLineMul023By023Chip<
    F: PrimeField32,
    const INPUT_BLOCKS: usize,
    const OUTPUT_BLOCKS: usize,
    const BLOCK_SIZE: usize,
>(
    pub  VmChipWrapper<
        F,
        Rv32VecHeapAdapterChip<F, 2, INPUT_BLOCKS, OUTPUT_BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        FieldExpressionCoreChip,
    >,
);

impl<
        F: PrimeField32,
        const INPUT_BLOCKS: usize,
        const OUTPUT_BLOCKS: usize,
        const BLOCK_SIZE: usize,
    > EcLineMul023By023Chip<F, INPUT_BLOCKS, OUTPUT_BLOCKS, BLOCK_SIZE>
{
    pub fn new(
        adapter: Rv32VecHeapAdapterChip<F, 2, INPUT_BLOCKS, OUTPUT_BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
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
        let expr = mul_023_by_023_expr(config, memory_controller.borrow().range_checker.bus(), xi);
        let core = FieldExpressionCoreChip::new(
            expr,
            offset,
            vec![PairingOpcode::MUL_023_BY_023 as usize],
            vec![],
            memory_controller.borrow().range_checker.clone(),
            "Mul023By023",
        );
        Self(VmChipWrapper::new(adapter, core, memory_controller))
    }
}

pub fn mul_023_by_023_expr(
    config: ExprBuilderConfig,
    range_bus: VariableRangeCheckerBus,
    xi: [isize; 2],
) -> FieldExpr {
    config.check_valid();
    let builder = ExprBuilder::new(config.clone(), range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let mut b0 = Fp2::new(builder.clone()); // x2
    let mut c0 = Fp2::new(builder.clone()); // x3
    let mut b1 = Fp2::new(builder.clone()); // y2
    let mut c1 = Fp2::new(builder.clone()); // y3

    // where w⁶ = xi
    // l0 * l1 = c0c1 + (c0b1 + c1b0)w² + (c0 + c1)w³ + (b0b1)w⁴ + (b0 +b1)w⁵ + w⁶
    //         = (c0c1 + xi) + (c0b1 + c1b0)w² + (c0 + c1)w³ + (b0b1)w⁴ + (b0 + b1)w⁵
    let l0 = c0.mul(&mut c1).int_add(xi);
    let l2 = c0.mul(&mut b1).add(&mut c1.mul(&mut b0));
    let l3 = c0.add(&mut c1);
    let l4 = b0.mul(&mut b1);
    let l5 = b0.add(&mut b1);

    [l0, l2, l3, l4, l5].map(|mut l| l.save_output());

    let builder = builder.borrow().clone();
    FieldExpr::new(builder, range_bus, false)
}
