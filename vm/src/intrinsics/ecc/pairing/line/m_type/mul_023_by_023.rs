use std::{cell::RefCell, rc::Rc};

use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_circuit_primitives::{
    bigint::check_carry_mod_to_zero::CheckCarryModToZeroSubAir, var_range::VariableRangeCheckerBus,
};
use ax_ecc_primitives::{
    field_expression::{ExprBuilder, ExprBuilderConfig, FieldExpr},
    field_extension::Fp2,
};
use axvm_circuit_derive::InstructionExecutor;
use axvm_ecc_constants::BLS12381;
use axvm_instructions::PairingOpcode;
use p3_field::PrimeField32;

use crate::{
    arch::VmChipWrapper, intrinsics::field_expression::FieldExpressionCoreChip,
    rv32im::adapters::Rv32VecHeapAdapterChip, system::memory::MemoryControllerRef,
};

// Input: line0.b, line0.c, line1.b, line1.c <Fp2>: 2 x 4 field elements
// Output: 5 Fp2 coefficients -> 10 field elements
#[derive(Chip, ChipUsageGetter, InstructionExecutor)]
pub struct EcLineMul023By023Chip<
    F: PrimeField32,
    const INPUT_LANES: usize,
    const OUTPUT_LANES: usize,
    const LANE_SIZE: usize,
>(
    VmChipWrapper<
        F,
        Rv32VecHeapAdapterChip<F, 2, INPUT_LANES, OUTPUT_LANES, LANE_SIZE, LANE_SIZE>,
        FieldExpressionCoreChip,
    >,
);

impl<
        F: PrimeField32,
        const INPUT_LANES: usize,
        const OUTPUT_LANES: usize,
        const LANE_SIZE: usize,
    > EcLineMul023By023Chip<F, INPUT_LANES, OUTPUT_LANES, LANE_SIZE>
{
    pub fn new(
        adapter: Rv32VecHeapAdapterChip<F, 2, INPUT_LANES, OUTPUT_LANES, LANE_SIZE, LANE_SIZE>,
        memory_controller: MemoryControllerRef<F>,
        config: ExprBuilderConfig,
        offset: usize,
    ) -> Self {
        let expr = mul_023_by_023_expr(
            config,
            memory_controller.borrow().range_checker.bus(),
            BLS12381.XI,
        );
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
    let subair = CheckCarryModToZeroSubAir::new(
        config.modulus,
        config.limb_bits,
        range_bus.index,
        range_bus.range_max_bits,
    );

    let mut b0 = Fp2::new(builder.clone()); // x2
    let mut c0 = Fp2::new(builder.clone()); // x3
    let mut b1 = Fp2::new(builder.clone()); // y2
    let mut c1 = Fp2::new(builder.clone()); // y3

    // where w⁶ = xi
    // l0 * l1 = c0c1 + (c0b1 + c1b0)w² + (c0 + c1)w³ + (b0b1)w⁴ + (b0 +b1)w⁵ + w⁶
    //         = (c0c1 + xi) + (c0b1 + c1b0)w² + (c0 + c1)w³ + (b0b1)w⁴ + (b0 + b1)w⁵
    let mut l0 = c0.mul(&mut c1).int_add(xi);
    let mut l2 = c0.mul(&mut b1).add(&mut c1.mul(&mut b0));
    let mut l3 = c0.add(&mut c1);
    let mut l4 = b0.mul(&mut b1);
    let mut l5 = b0.add(&mut b1);

    l0.save_output();
    l2.save_output();
    l3.save_output();
    l4.save_output();
    l5.save_output();

    let builder = builder.borrow().clone();
    FieldExpr {
        builder,
        check_carry_mod_to_zero: subair,
        range_bus,
    }
}
