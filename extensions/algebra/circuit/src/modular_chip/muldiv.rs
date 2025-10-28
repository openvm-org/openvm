use std::{cell::RefCell, rc::Rc};

use openvm_algebra_transpiler::Rv32ModularArithmeticOpcode;
use openvm_circuit::{
    arch::ExecutionBridge,
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
};
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_mod_circuit_builder::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreAir, FieldExpressionExecutor,
    FieldExpressionFiller, FieldVariable, SymbolicExpr,
};
use openvm_rv32_adapters::{
    Rv32VecHeapAdapterAir, Rv32VecHeapAdapterExecutor, Rv32VecHeapAdapterFiller,
};

use super::{ModularAir, ModularChip, ModularExecutor};
use crate::FieldExprVecHeapExecutor;

pub fn muldiv_expr(
    config: ExprBuilderConfig,
    range_bus: VariableRangeCheckerBus,
) -> (FieldExpr, usize, usize) {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_bus.range_max_bits);
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
        FieldExpr::new(builder, range_bus, true),
        is_mul_flag,
        is_div_flag,
    )
}

fn gen_base_expr(
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
) -> (FieldExpr, Vec<usize>, Vec<usize>) {
    let (expr, is_mul_flag, is_div_flag) = muldiv_expr(config, range_checker_bus);

    let local_opcode_idx = vec![
        Rv32ModularArithmeticOpcode::MUL as usize,
        Rv32ModularArithmeticOpcode::DIV as usize,
        Rv32ModularArithmeticOpcode::SETUP_MULDIV as usize,
    ];
    let opcode_flag_idx = vec![is_mul_flag, is_div_flag];

    (expr, local_opcode_idx, opcode_flag_idx)
}

pub fn get_modular_muldiv_air<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    exec_bridge: ExecutionBridge,
    mem_bridge: MemoryBridge,
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
    pointer_max_bits: usize,
    offset: usize,
) -> ModularAir<BLOCKS, BLOCK_SIZE> {
    let (expr, local_opcode_idx, opcode_flag_idx) = gen_base_expr(config, range_checker_bus);
    ModularAir::new(
        Rv32VecHeapAdapterAir::new(
            exec_bridge,
            mem_bridge,
            bitwise_lookup_bus,
            pointer_max_bits,
        ),
        FieldExpressionCoreAir::new(expr, offset, local_opcode_idx, opcode_flag_idx),
    )
}

pub fn get_modular_muldiv_step<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
    offset: usize,
) -> ModularExecutor<BLOCKS, BLOCK_SIZE> {
    let (expr, local_opcode_idx, opcode_flag_idx) = gen_base_expr(config, range_checker_bus);

    FieldExprVecHeapExecutor(FieldExpressionExecutor::new(
        Rv32VecHeapAdapterExecutor::new(pointer_max_bits),
        expr,
        offset,
        local_opcode_idx,
        opcode_flag_idx,
        "ModularMulDiv",
    ))
}

pub fn get_modular_muldiv_chip<F, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    config: ExprBuilderConfig,
    mem_helper: SharedMemoryHelper<F>,
    range_checker: SharedVariableRangeCheckerChip,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    pointer_max_bits: usize,
) -> ModularChip<F, BLOCKS, BLOCK_SIZE> {
    let (expr, local_opcode_idx, opcode_flag_idx) = gen_base_expr(config, range_checker.bus());
    ModularChip::new(
        FieldExpressionFiller::new(
            Rv32VecHeapAdapterFiller::new(pointer_max_bits, bitwise_lookup_chip),
            expr,
            local_opcode_idx,
            opcode_flag_idx,
            range_checker,
            false,
        ),
        mem_helper,
    )
}
