use std::{cell::RefCell, rc::Rc};

use openvm_algebra_transpiler::Fp2Opcode;
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
    FieldExpressionFiller, SymbolicExpr,
};
use openvm_rv32_adapters::{
    Rv32VecHeapAdapterAir, Rv32VecHeapAdapterExecutor, Rv32VecHeapAdapterFiller,
};

use super::{Fp2Air, Fp2Chip, Fp2Executor};
use crate::{FieldExprVecHeapExecutor, Fp2};

pub fn fp2_muldiv_expr(
    config: ExprBuilderConfig,
    range_bus: VariableRangeCheckerBus,
) -> (FieldExpr, usize, usize) {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let x = Fp2::new(builder.clone());
    let mut y = Fp2::new(builder.clone());
    let is_mul_flag = builder.borrow_mut().new_flag();
    let is_div_flag = builder.borrow_mut().new_flag();
    let (z_idx, mut z) = Fp2::new_var(builder.clone());

    let mut lvar = Fp2::select(is_mul_flag, &x, &z);

    let mut rvar = Fp2::select(is_mul_flag, &z, &x);
    let fp2_constraint = lvar.mul(&mut y).sub(&mut rvar);
    // When it's SETUP op, the constraints is z * y - x = 0, it still works as:
    // x.c0 = x.c1 = p == 0, y.c0 = y.c1 = 0, so whatever z is, z * 0 - 0 = 0

    z.save_output();
    builder
        .borrow_mut()
        .set_constraint(z_idx.0, fp2_constraint.c0.expr);
    builder
        .borrow_mut()
        .set_constraint(z_idx.1, fp2_constraint.c1.expr);

    // Compute expression has to be done manually at the SymbolicExpr level.
    // Otherwise it saves the quotient and introduces new variables.
    let compute_z0_div = (&x.c0.expr * &y.c0.expr + &x.c1.expr * &y.c1.expr)
        / (&y.c0.expr * &y.c0.expr + &y.c1.expr * &y.c1.expr);
    let compute_z0_mul = &x.c0.expr * &y.c0.expr - &x.c1.expr * &y.c1.expr;
    let compute_z0 = SymbolicExpr::Select(
        is_mul_flag,
        Box::new(compute_z0_mul),
        Box::new(SymbolicExpr::Select(
            is_div_flag,
            Box::new(compute_z0_div),
            Box::new(x.c0.expr.clone()),
        )),
    );
    let compute_z1_div = (&x.c1.expr * &y.c0.expr - &x.c0.expr * &y.c1.expr)
        / (&y.c0.expr * &y.c0.expr + &y.c1.expr * &y.c1.expr);
    let compute_z1_mul = &x.c1.expr * &y.c0.expr + &x.c0.expr * &y.c1.expr;
    let compute_z1 = SymbolicExpr::Select(
        is_mul_flag,
        Box::new(compute_z1_mul),
        Box::new(SymbolicExpr::Select(
            is_div_flag,
            Box::new(compute_z1_div),
            Box::new(x.c1.expr),
        )),
    );
    builder.borrow_mut().set_compute(z_idx.0, compute_z0);
    builder.borrow_mut().set_compute(z_idx.1, compute_z1);

    let builder = builder.borrow().clone();
    (
        FieldExpr::new(builder, range_bus, true),
        is_mul_flag,
        is_div_flag,
    )
}

// Input: Fp2 * 2
// Output: Fp2

fn gen_base_expr(
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
) -> (FieldExpr, Vec<usize>, Vec<usize>) {
    let (expr, is_mul_flag, is_div_flag) = fp2_muldiv_expr(config, range_checker_bus);

    let local_opcode_idx = vec![
        Fp2Opcode::MUL as usize,
        Fp2Opcode::DIV as usize,
        Fp2Opcode::SETUP_MULDIV as usize,
    ];
    let opcode_flag_idx = vec![is_mul_flag, is_div_flag];

    (expr, local_opcode_idx, opcode_flag_idx)
}

pub fn get_fp2_muldiv_air<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    exec_bridge: ExecutionBridge,
    mem_bridge: MemoryBridge,
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
    pointer_max_bits: usize,
    offset: usize,
) -> Fp2Air<BLOCKS, BLOCK_SIZE> {
    let (expr, local_opcode_idx, opcode_flag_idx) = gen_base_expr(config, range_checker_bus);
    Fp2Air::new(
        Rv32VecHeapAdapterAir::new(
            exec_bridge,
            mem_bridge,
            bitwise_lookup_bus,
            pointer_max_bits,
        ),
        FieldExpressionCoreAir::new(expr, offset, local_opcode_idx, opcode_flag_idx),
    )
}

pub fn get_fp2_muldiv_step<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
    offset: usize,
) -> Fp2Executor<BLOCKS, BLOCK_SIZE> {
    let (expr, local_opcode_idx, opcode_flag_idx) = gen_base_expr(config, range_checker_bus);

    FieldExprVecHeapExecutor(FieldExpressionExecutor::new(
        Rv32VecHeapAdapterExecutor::new(pointer_max_bits),
        expr,
        offset,
        local_opcode_idx,
        opcode_flag_idx,
        "Fp2MulDiv",
    ))
}

pub fn get_fp2_muldiv_chip<F, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    config: ExprBuilderConfig,
    mem_helper: SharedMemoryHelper<F>,
    range_checker: SharedVariableRangeCheckerChip,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    pointer_max_bits: usize,
) -> Fp2Chip<F, BLOCKS, BLOCK_SIZE> {
    let (expr, local_opcode_idx, opcode_flag_idx) = gen_base_expr(config, range_checker.bus());
    Fp2Chip::new(
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
