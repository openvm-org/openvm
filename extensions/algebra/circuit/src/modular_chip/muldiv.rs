use std::{cell::RefCell, rc::Rc};

use openvm_algebra_transpiler::Rv64ModularArithmeticOpcode;
use openvm_circuit::{
    arch::ExecutionBridge,
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_primitives::var_range::{
    SharedVariableRangeCheckerChip, VariableRangeCheckerBus,
};
use openvm_mod_circuit_builder::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreAir, FieldExpressionExecutor,
    FieldExpressionFiller, FieldExpressionProgram, FieldVariable, SymbolicExpr,
};
use openvm_riscv_adapters::{
    Rv64VecHeapAdapterAir, Rv64VecHeapAdapterExecutor, Rv64VecHeapAdapterFiller,
};

use super::{ModularAir, ModularChip, ModularExecutor};
use crate::FieldExprVecHeapExecutor;

pub fn muldiv_expr(
    config: ExprBuilderConfig,
    range_bus: VariableRangeCheckerBus,
) -> (FieldExpr, usize, usize) {
    let (program, is_mul_flag, is_div_flag) = muldiv_program(config, range_bus.range_max_bits);
    (FieldExpr::new(program, range_bus), is_mul_flag, is_div_flag)
}

fn muldiv_program(
    config: ExprBuilderConfig,
    range_max_bits: usize,
) -> (FieldExpressionProgram, usize, usize) {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_max_bits);
    let builder = Rc::new(RefCell::new(builder));
    let x = ExprBuilder::new_input(builder.clone());
    let y = ExprBuilder::new_input(builder.clone());
    let (z_idx, z) = (*builder).borrow_mut().new_var();
    let mut z = FieldVariable::from_var(builder.clone(), z);
    let is_mul_flag = (*builder).borrow_mut().new_flag();
    let is_div_flag = (*builder).borrow_mut().new_flag();
    // constraint is x * y = z, or z * y = x
    let lvar = FieldVariable::select(is_mul_flag, &x, &z);
    let rvar = FieldVariable::select(is_mul_flag, &z, &x);
    // When it's SETUP op, x = p == 0, y = 0, both flags are false, and it still works: z * 0 - x =
    // 0, whatever z is.
    let constraint = lvar * y.clone() - rvar;
    (*builder)
        .borrow_mut()
        .set_constraint(z_idx, constraint.expr);
    let compute = SymbolicExpr::Select(
        is_mul_flag,
        Box::new(x.expr.clone() * y.expr.clone()),
        Box::new(SymbolicExpr::Select(
            is_div_flag,
            Box::new(x.expr.clone() / y.expr.clone()),
            Box::new(x.expr.clone()),
        )),
    );
    (*builder).borrow_mut().set_compute(z_idx, compute);
    z.save_output();

    let builder = (*builder).borrow().clone();

    (
        FieldExpressionProgram::new(builder, true),
        is_mul_flag,
        is_div_flag,
    )
}

fn gen_base_program(
    config: ExprBuilderConfig,
    range_max_bits: usize,
) -> (FieldExpressionProgram, Vec<usize>, Vec<usize>) {
    let (program, is_mul_flag, is_div_flag) = muldiv_program(config, range_max_bits);

    let local_opcode_idx = vec![
        Rv64ModularArithmeticOpcode::MUL as usize,
        Rv64ModularArithmeticOpcode::DIV as usize,
        Rv64ModularArithmeticOpcode::SETUP_MULDIV as usize,
    ];
    let opcode_flag_idx = vec![is_mul_flag, is_div_flag];

    (program, local_opcode_idx, opcode_flag_idx)
}

pub fn get_modular_muldiv_air<const BLOCKS: usize>(
    exec_bridge: ExecutionBridge,
    mem_bridge: MemoryBridge,
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
    offset: usize,
) -> ModularAir<BLOCKS> {
    let (program, local_opcode_idx, opcode_flag_idx) =
        gen_base_program(config, range_checker_bus.range_max_bits);
    let expr = FieldExpr::new(program, range_checker_bus);
    ModularAir::new(
        Rv64VecHeapAdapterAir::new(exec_bridge, mem_bridge, range_checker_bus, pointer_max_bits),
        FieldExpressionCoreAir::new(expr, offset, local_opcode_idx, opcode_flag_idx),
    )
}

pub fn get_modular_muldiv_executor<const BLOCKS: usize>(
    config: ExprBuilderConfig,
    range_max_bits: usize,
    pointer_max_bits: usize,
    offset: usize,
) -> ModularExecutor<BLOCKS> {
    let (program, local_opcode_idx, opcode_flag_idx) = gen_base_program(config, range_max_bits);

    FieldExprVecHeapExecutor::new(FieldExpressionExecutor::new(
        Rv64VecHeapAdapterExecutor::new(pointer_max_bits),
        program,
        offset,
        local_opcode_idx,
        opcode_flag_idx,
        "ModularMulDiv",
    ))
}

pub fn get_modular_muldiv_chip<F, const BLOCKS: usize>(
    config: ExprBuilderConfig,
    mem_helper: SharedMemoryHelper<F>,
    range_checker: SharedVariableRangeCheckerChip,
    pointer_max_bits: usize,
) -> ModularChip<F, BLOCKS> {
    let range_bus = range_checker.bus();
    let (program, local_opcode_idx, opcode_flag_idx) =
        gen_base_program(config, range_bus.range_max_bits);
    let expr = FieldExpr::new(program, range_bus);
    ModularChip::new(
        FieldExpressionFiller::new(
            Rv64VecHeapAdapterFiller::new(pointer_max_bits, range_checker.clone()),
            expr,
            local_opcode_idx,
            opcode_flag_idx,
            range_checker,
            false,
        ),
        mem_helper,
    )
}
