use std::{cell::RefCell, rc::Rc};

use openvm_algebra_transpiler::Fp2Opcode;
use openvm_circuit::{
    arch::ExecutionBridge,
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_primitives::var_range::{
    SharedVariableRangeCheckerChip, VariableRangeCheckerBus,
};
use openvm_mod_circuit_builder::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreAir, FieldExpressionExecutor,
    FieldExpressionFiller, FieldExpressionProgram,
};
use openvm_riscv_adapters::{
    Rv64VecHeapAdapterAir, Rv64VecHeapAdapterExecutor, Rv64VecHeapAdapterFiller,
};

use super::{Fp2Air, Fp2Chip, Fp2Executor};
use crate::{FieldExprVecHeapExecutor, Fp2};

pub fn fp2_addsub_expr(
    config: ExprBuilderConfig,
    range_bus: VariableRangeCheckerBus,
) -> (FieldExpr, usize, usize) {
    let (program, is_add_flag, is_sub_flag) = fp2_addsub_program(config, range_bus.range_max_bits);
    (FieldExpr::new(program, range_bus), is_add_flag, is_sub_flag)
}

fn fp2_addsub_program(
    config: ExprBuilderConfig,
    range_max_bits: usize,
) -> (FieldExpressionProgram, usize, usize) {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let mut x = Fp2::new(builder.clone());
    let mut y = Fp2::new(builder.clone());
    let add = x.add(&mut y);
    let sub = x.sub(&mut y);

    let is_add_flag = builder.borrow_mut().new_flag();
    let is_sub_flag = builder.borrow_mut().new_flag();
    let diff = Fp2::select(is_sub_flag, &sub, &x);
    let mut z = Fp2::select(is_add_flag, &add, &diff);
    z.save_output();

    let builder = builder.borrow().clone();
    (
        FieldExpressionProgram::new(builder, true),
        is_add_flag,
        is_sub_flag,
    )
}

// Input: Fp2 * 2
// Output: Fp2
fn gen_base_program(
    config: ExprBuilderConfig,
    range_max_bits: usize,
) -> (FieldExpressionProgram, Vec<usize>, Vec<usize>) {
    let (program, is_add_flag, is_sub_flag) = fp2_addsub_program(config, range_max_bits);

    let local_opcode_idx = vec![
        Fp2Opcode::ADD as usize,
        Fp2Opcode::SUB as usize,
        Fp2Opcode::SETUP_ADDSUB as usize,
    ];
    let opcode_flag_idx = vec![is_add_flag, is_sub_flag];

    (program, local_opcode_idx, opcode_flag_idx)
}

pub fn get_fp2_addsub_air<const BLOCKS: usize>(
    exec_bridge: ExecutionBridge,
    mem_bridge: MemoryBridge,
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
    offset: usize,
) -> Fp2Air<BLOCKS> {
    let (program, local_opcode_idx, opcode_flag_idx) =
        gen_base_program(config, range_checker_bus.range_max_bits);
    let expr = FieldExpr::new(program, range_checker_bus);
    Fp2Air::new(
        Rv64VecHeapAdapterAir::new(exec_bridge, mem_bridge, range_checker_bus, pointer_max_bits),
        FieldExpressionCoreAir::new(expr, offset, local_opcode_idx, opcode_flag_idx),
    )
}

pub fn get_fp2_addsub_executor<const BLOCKS: usize>(
    config: ExprBuilderConfig,
    range_max_bits: usize,
    pointer_max_bits: usize,
    offset: usize,
) -> Fp2Executor<BLOCKS> {
    let (program, local_opcode_idx, opcode_flag_idx) = gen_base_program(config, range_max_bits);

    FieldExprVecHeapExecutor::new(FieldExpressionExecutor::new(
        Rv64VecHeapAdapterExecutor::new(pointer_max_bits),
        program,
        offset,
        local_opcode_idx,
        opcode_flag_idx,
        "Fp2AddSub",
    ))
}

pub fn get_fp2_addsub_chip<F, const BLOCKS: usize>(
    config: ExprBuilderConfig,
    mem_helper: SharedMemoryHelper<F>,
    range_checker: SharedVariableRangeCheckerChip,
    pointer_max_bits: usize,
) -> Fp2Chip<F, BLOCKS> {
    let range_bus = range_checker.bus();
    let (program, local_opcode_idx, opcode_flag_idx) =
        gen_base_program(config, range_bus.range_max_bits);
    let expr = FieldExpr::new(program, range_bus);
    Fp2Chip::new(
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
