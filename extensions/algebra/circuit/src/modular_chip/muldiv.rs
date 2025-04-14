use std::{
    cell::RefCell,
    rc::Rc,
    sync::{Arc, Mutex},
};

use openvm_algebra_transpiler::Rv32ModularArithmeticOpcode;
use openvm_circuit::{arch::VmChipWrapper, system::memory::OfflineMemory};
use openvm_circuit_derive::InstructionExecutor;
use openvm_circuit_primitives::var_range::{
    SharedVariableRangeCheckerChip, VariableRangeCheckerBus,
};
use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
use openvm_mod_circuit_builder::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreChip, FieldVariable, SymbolicExpr,
};
use openvm_rv32_adapters::Rv32VecHeapAdapterChip;
use openvm_stark_backend::p3_field::PrimeField32;

pub fn muldiv_expr(
    config: ExprBuilderConfig,
    range_bus: VariableRangeCheckerBus,
) -> (FieldExpr, usize, usize) {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));
    let x = ExprBuilder::new_input(builder.clone());
    let y = ExprBuilder::new_input(builder.clone());
    let (z_idx, z) = builder.borrow_mut().new_var();
    let mut z = FieldVariable::from_var(builder.clone(), z);
    let is_mul_flag = builder.borrow_mut().new_flag();
    let is_div_flag = builder.borrow_mut().new_flag();
    // constraint is x * y = z, or z * y = x
    let lvar = FieldVariable::select(is_mul_flag, &x, &z);
    let rvar = FieldVariable::select(is_mul_flag, &z, &x);
    // When it's SETUP op, x = p == 0, y = 0, both flags are false, and it still works: z * 0 - x =
    // 0, whatever z is.
    let constraint = lvar * y.clone() - rvar;
    builder.borrow_mut().set_constraint(z_idx, constraint.expr);
    let compute = SymbolicExpr::Select(
        is_mul_flag,
        Box::new(x.expr.clone() * y.expr.clone()),
        Box::new(SymbolicExpr::Select(
            is_div_flag,
            Box::new(x.expr.clone() / y.expr.clone()),
            Box::new(x.expr.clone()),
        )),
    );
    builder.borrow_mut().set_compute(z_idx, compute);
    z.save_output();

    let builder = builder.borrow().clone();

    (
        FieldExpr::new(builder, range_bus, true),
        is_mul_flag,
        is_div_flag,
    )
}

#[derive(Chip, ChipUsageGetter, InstructionExecutor)]
pub struct ModularMulDivChip<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    pub  VmChipWrapper<
        F,
        Rv32VecHeapAdapterChip<F, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        FieldExpressionCoreChip,
    >,
);

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>
    ModularMulDivChip<F, BLOCKS, BLOCK_SIZE>
{
    pub fn new(
        adapter: Rv32VecHeapAdapterChip<F, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        config: ExprBuilderConfig,
        offset: usize,
        range_checker: SharedVariableRangeCheckerChip,
        offline_memory: Arc<Mutex<OfflineMemory<F>>>,
    ) -> Self {
        let (expr, is_mul_flag, is_div_flag) = muldiv_expr(config, range_checker.bus());
        let core = FieldExpressionCoreChip::new(
            expr,
            offset,
            vec![
                Rv32ModularArithmeticOpcode::MUL as usize,
                Rv32ModularArithmeticOpcode::DIV as usize,
                Rv32ModularArithmeticOpcode::SETUP_MULDIV as usize,
            ],
            vec![is_mul_flag, is_div_flag],
            range_checker,
            "ModularMulDiv",
            false,
        );
        Self(VmChipWrapper::new(adapter, core, offline_memory))
    }
}
