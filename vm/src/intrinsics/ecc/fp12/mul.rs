use std::{cell::RefCell, rc::Rc};

use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_circuit_primitives::var_range::VariableRangeCheckerBus;
use ax_ecc_primitives::{
    field_expression::{ExprBuilder, ExprBuilderConfig, FieldExpr},
    field_extension::Fp12,
};
use axvm_circuit_derive::InstructionExecutor;
use p3_field::PrimeField32;

use crate::{
    arch::{instructions::Fp12Opcode, VmChipWrapper},
    intrinsics::field_expression::FieldExpressionCoreChip,
    rv32im::adapters::Rv32VecHeapAdapterChip,
    system::memory::MemoryControllerRef,
};

// Input: Fp12 * 2
// Output: Fp12
#[derive(Chip, ChipUsageGetter, InstructionExecutor)]
pub struct Fp12MulChip<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    pub  VmChipWrapper<
        F,
        Rv32VecHeapAdapterChip<F, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        FieldExpressionCoreChip,
    >,
);

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>
    Fp12MulChip<F, BLOCKS, BLOCK_SIZE>
{
    pub fn new(
        adapter: Rv32VecHeapAdapterChip<F, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        memory_controller: MemoryControllerRef<F>,
        config: ExprBuilderConfig,
        offset: usize,
        xi: [isize; 2],
    ) -> Self {
        let expr = fp12_mul_expr(config, memory_controller.borrow().range_checker.bus(), xi);
        let core = FieldExpressionCoreChip::new(
            expr,
            offset,
            vec![Fp12Opcode::MUL as usize],
            vec![],
            memory_controller.borrow().range_checker.clone(),
            "Fp12Mul",
        );
        Self(VmChipWrapper::new(adapter, core, memory_controller))
    }
}

pub fn fp12_mul_expr(
    config: ExprBuilderConfig,
    range_bus: VariableRangeCheckerBus,
    xi: [isize; 2],
) -> FieldExpr {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let mut x = Fp12::new(builder.clone());
    let mut y = Fp12::new(builder.clone());
    let mut res = x.mul(&mut y, xi);
    res.save_output();

    let builder = builder.borrow().clone();
    FieldExpr::new(builder, range_bus)
}
