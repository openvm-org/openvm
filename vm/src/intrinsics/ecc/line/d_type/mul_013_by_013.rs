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
use axvm_ecc_constants::BN254;
use axvm_instructions::EcLineDTypeOpcode;
use p3_field::PrimeField32;

use crate::{
    arch::VmChipWrapper, intrinsics::field_expression::FieldExpressionCoreChip,
    rv32im::adapters::Rv32VecHeapAdapterChip, system::memory::MemoryControllerRef,
};

// Input: line0.b, line0.c, line1.b, line1.c <Fp2>: 2 x 4 field elements
// Output: 5 Fp2 coefficients -> 10 field elements
#[derive(Chip, ChipUsageGetter, InstructionExecutor)]
pub struct EcLineMul013By013Chip<
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
    > EcLineMul013By013Chip<F, INPUT_LANES, OUTPUT_LANES, LANE_SIZE>
{
    pub fn new(
        adapter: Rv32VecHeapAdapterChip<F, 2, INPUT_LANES, OUTPUT_LANES, LANE_SIZE, LANE_SIZE>,
        memory_controller: MemoryControllerRef<F>,
        config: ExprBuilderConfig,
        offset: usize,
    ) -> Self {
        let expr = mul_013_by_013_expr(
            config,
            memory_controller.borrow().range_checker.bus(),
            BN254.XI,
        );
        let core = FieldExpressionCoreChip::new(
            expr,
            offset,
            vec![EcLineDTypeOpcode::MUL_013_BY_013 as usize],
            memory_controller.borrow().range_checker.clone(),
            "Mul013By013",
        );
        Self(VmChipWrapper::new(adapter, core, memory_controller))
    }
}

pub fn mul_013_by_013_expr(
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

    let mut b0 = Fp2::new(builder.clone()); // x1
    let mut c0 = Fp2::new(builder.clone()); // x3
    let mut b1 = Fp2::new(builder.clone()); // y1
    let mut c1 = Fp2::new(builder.clone()); // y3

    // where w⁶ = xi
    // l0 * l1 = 1 + (b0 + b1)w + (b0b1)w² + (c0 + c1)w³ + (b0c1 + b1c0)w⁴ + (c0c1)w⁶
    //         = (1 + c0c1 * xi) + (b0 + b1)w + (b0b1)w² + (c0 + c1)w³ + (b0c1 + b1c0)w⁴
    let mut l0 = c0.mul(&mut c1).int_mul(xi).int_add([1, 0]);
    let mut l1 = b0.add(&mut b1);
    let mut l2 = b0.mul(&mut b1);
    let mut l3 = c0.add(&mut c1);
    let mut l4 = b0.mul(&mut c1).add(&mut b1.mul(&mut c0));

    l0.save_output();
    l1.save_output();
    l2.save_output();
    l3.save_output();
    l4.save_output();

    let builder = builder.borrow().clone();
    FieldExpr {
        builder,
        check_carry_mod_to_zero: subair,
        range_bus,
    }
}
