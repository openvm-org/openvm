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
    FieldExpressionFiller, FieldVariable,
};
use openvm_riscv_adapters::{
    Rv64VecHeapAdapterAir, Rv64VecHeapAdapterExecutor, Rv64VecHeapAdapterFiller,
    Rv64VecHeapU16AdapterAir, Rv64VecHeapU16AdapterExecutor, Rv64VecHeapU16AdapterFiller,
};

use super::{
    ModularAir, ModularChip, ModularExecutor, ModularU16Air, ModularU16Chip, ModularU16Executor,
};
use crate::FieldExprVecHeapExecutor;

pub fn addsub_expr(
    config: ExprBuilderConfig,
    range_bus: VariableRangeCheckerBus,
) -> (FieldExpr, usize, usize) {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_bus.range_max_bits);
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
        FieldExpr::new(builder, range_bus, true),
        is_add_flag,
        is_sub_flag,
    )
}

pub(crate) fn is_modular_addsub_u16_supported(
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
) -> bool {
    let (expr, _, _) = addsub_expr(config, range_checker_bus);
    expr.builder.constraints.iter().all(|constraint| {
        constraint.constraint_carry_bits_with_pq(
            &expr.builder.prime,
            expr.builder.limb_bits,
            expr.builder.num_limbs,
            expr.builder.proper_max(),
        ) <= expr.builder.max_carry_bits
    })
}

fn gen_base_expr(
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
) -> (FieldExpr, Vec<usize>, Vec<usize>) {
    let (expr, is_add_flag, is_sub_flag) = addsub_expr(config, range_checker_bus);

    let local_opcode_idx = vec![
        Rv64ModularArithmeticOpcode::ADD as usize,
        Rv64ModularArithmeticOpcode::SUB as usize,
        Rv64ModularArithmeticOpcode::SETUP_ADDSUB as usize,
    ];
    let opcode_flag_idx = vec![is_add_flag, is_sub_flag];

    (expr, local_opcode_idx, opcode_flag_idx)
}

pub fn get_modular_addsub_air<const BLOCKS: usize>(
    exec_bridge: ExecutionBridge,
    mem_bridge: MemoryBridge,
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
    offset: usize,
) -> ModularAir<BLOCKS> {
    let (expr, local_opcode_idx, opcode_flag_idx) = gen_base_expr(config, range_checker_bus);
    ModularAir::new(
        Rv64VecHeapAdapterAir::new(exec_bridge, mem_bridge, range_checker_bus, pointer_max_bits),
        FieldExpressionCoreAir::new(expr, offset, local_opcode_idx, opcode_flag_idx),
    )
}

pub fn get_modular_addsub_executor<const BLOCKS: usize>(
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
    offset: usize,
) -> ModularExecutor<BLOCKS> {
    let (expr, local_opcode_idx, opcode_flag_idx) = gen_base_expr(config, range_checker_bus);

    FieldExprVecHeapExecutor::new(FieldExpressionExecutor::new(
        Rv64VecHeapAdapterExecutor::new(pointer_max_bits),
        expr,
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
    let (expr, local_opcode_idx, opcode_flag_idx) = gen_base_expr(config, range_checker.bus());
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

pub(crate) fn get_modular_addsub_u16_air<const BLOCKS: usize>(
    exec_bridge: ExecutionBridge,
    mem_bridge: MemoryBridge,
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
    offset: usize,
) -> ModularU16Air<BLOCKS> {
    let (expr, local_opcode_idx, opcode_flag_idx) = gen_base_expr(config, range_checker_bus);
    ModularU16Air::new(
        Rv64VecHeapU16AdapterAir::new(exec_bridge, mem_bridge, range_checker_bus, pointer_max_bits),
        FieldExpressionCoreAir::new(expr, offset, local_opcode_idx, opcode_flag_idx),
    )
}

pub(crate) fn get_modular_addsub_u16_executor<const BLOCKS: usize>(
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
    offset: usize,
) -> ModularU16Executor<BLOCKS> {
    let (expr, local_opcode_idx, opcode_flag_idx) = gen_base_expr(config, range_checker_bus);

    FieldExprVecHeapExecutor::new(FieldExpressionExecutor::new(
        Rv64VecHeapU16AdapterExecutor::new(pointer_max_bits),
        expr,
        offset,
        local_opcode_idx,
        opcode_flag_idx,
        "Rv64ModularAddSubU16",
    ))
}

pub(crate) fn get_modular_addsub_u16_chip<F, const BLOCKS: usize>(
    config: ExprBuilderConfig,
    mem_helper: SharedMemoryHelper<F>,
    range_checker: SharedVariableRangeCheckerChip,
    pointer_max_bits: usize,
) -> ModularU16Chip<F, BLOCKS> {
    let (expr, local_opcode_idx, opcode_flag_idx) = gen_base_expr(config, range_checker.bus());
    ModularU16Chip::new(
        FieldExpressionFiller::new(
            Rv64VecHeapU16AdapterFiller::new(pointer_max_bits, range_checker.clone()),
            expr,
            local_opcode_idx,
            opcode_flag_idx,
            range_checker,
            false,
        ),
        mem_helper,
    )
}
