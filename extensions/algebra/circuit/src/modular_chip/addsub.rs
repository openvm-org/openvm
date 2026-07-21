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
    FieldExpressionFiller, FieldExpressionProgram, FieldVariable,
};
use openvm_riscv_adapters::{
    Rv64VecHeapAdapterAir, Rv64VecHeapAdapterExecutor, Rv64VecHeapAdapterFiller,
};

use super::{ModularAir, ModularChip, ModularExecutor};
use crate::FieldExprVecHeapExecutor;

pub fn addsub_expr(
    config: ExprBuilderConfig,
    range_bus: VariableRangeCheckerBus,
) -> (FieldExpr, usize, usize) {
    let (program, is_add_flag, is_sub_flag) = addsub_program(config, range_bus.range_max_bits);
    (FieldExpr::new(program, range_bus), is_add_flag, is_sub_flag)
}

fn addsub_program(
    config: ExprBuilderConfig,
    range_max_bits: usize,
) -> (FieldExpressionProgram, usize, usize) {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let x1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let x3 = x1.clone() + x2.clone();
    let x4 = x1.clone() - x2.clone();
    let is_add_flag = (*builder).borrow_mut().new_flag();
    let is_sub_flag = (*builder).borrow_mut().new_flag();
    let x5 = FieldVariable::select(is_sub_flag, &x4, &x1);
    let mut x6 = FieldVariable::select(is_add_flag, &x3, &x5);
    x6.save_output();
    let builder = (*builder).borrow().clone();

    (
        FieldExpressionProgram::new(builder, true),
        is_add_flag,
        is_sub_flag,
    )
}

fn gen_base_program(
    config: ExprBuilderConfig,
    range_max_bits: usize,
) -> (FieldExpressionProgram, Vec<usize>, Vec<usize>) {
    let (program, is_add_flag, is_sub_flag) = addsub_program(config, range_max_bits);

    let local_opcode_idx = vec![
        Rv64ModularArithmeticOpcode::ADD as usize,
        Rv64ModularArithmeticOpcode::SUB as usize,
        Rv64ModularArithmeticOpcode::SETUP_ADDSUB as usize,
    ];
    let opcode_flag_idx = vec![is_add_flag, is_sub_flag];

    (program, local_opcode_idx, opcode_flag_idx)
}

pub fn get_modular_addsub_air<const BLOCKS: usize>(
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

pub fn get_modular_addsub_executor<const BLOCKS: usize>(
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
        "ModularAddSub",
    ))
}

pub fn get_modular_addsub_chip<F, const BLOCKS: usize>(
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
