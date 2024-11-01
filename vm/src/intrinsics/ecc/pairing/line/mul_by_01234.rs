use std::{cell::RefCell, rc::Rc};

use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_circuit_primitives::{
    bigint::check_carry_mod_to_zero::CheckCarryModToZeroSubAir, var_range::VariableRangeCheckerBus,
};
use ax_ecc_primitives::{
    field_expression::{ExprBuilder, ExprBuilderConfig, FieldExpr},
    field_extension::{Fp12, Fp2},
};
use axvm_circuit_derive::InstructionExecutor;
use axvm_ecc_constants::BN254;
use axvm_instructions::PairingOpcode;
use p3_field::PrimeField32;

use crate::{
    arch::VmChipWrapper, intrinsics::field_expression::FieldExpressionCoreChip,
    rv32im::adapters::Rv32VecHeapAdapterChip, system::memory::MemoryControllerRef,
};

// TODO[yj]: Update to use 10 FE for 2nd input once the adapter change is merged for unbalanced inputs
// Input: 2 Fp12: 2 x 12 field elements
// Output: Fp12 -> 12 field elements
#[derive(Chip, ChipUsageGetter, InstructionExecutor)]
pub struct EcLineMulBy01234Chip<
    F: PrimeField32,
    const INPUT_BLOCKS: usize,
    const OUTPUT_BLOCKS: usize,
    const BLOCK_SIZE: usize,
>(
    VmChipWrapper<
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
    > EcLineMulBy01234Chip<F, INPUT_BLOCKS, OUTPUT_BLOCKS, BLOCK_SIZE>
{
    pub fn new(
        adapter: Rv32VecHeapAdapterChip<F, 2, INPUT_BLOCKS, OUTPUT_BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        memory_controller: MemoryControllerRef<F>,
        config: ExprBuilderConfig,
        offset: usize,
    ) -> Self {
        let expr = mul_by_01234_expr(
            config,
            memory_controller.borrow().range_checker.bus(),
            BN254.XI,
        );
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
    let subair = CheckCarryModToZeroSubAir::new(
        config.modulus,
        config.limb_bits,
        range_bus.index,
        range_bus.range_max_bits,
    );

    let mut f = Fp12::new(builder.clone());
    let mut x0 = Fp2::new(builder.clone());
    let mut x1 = Fp2::new(builder.clone());
    let mut x2 = Fp2::new(builder.clone());
    let mut x3 = Fp2::new(builder.clone());
    let mut x4 = Fp2::new(builder.clone());
    // x5 is unused; required for input sizes to balance to 12 on the adapter
    let _x5 = Fp2::new(builder.clone());

    let mut r = f.mul_by_01234(&mut x0, &mut x1, &mut x2, &mut x3, &mut x4, xi);
    r.save_output();

    let builder = builder.borrow().clone();
    FieldExpr {
        builder,
        check_carry_mod_to_zero: subair,
        range_bus,
    }
}
