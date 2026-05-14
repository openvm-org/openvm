use std::{cell::RefCell, rc::Rc};

use openvm_algebra_transpiler::Rv64ModularArithmeticOpcode;
use openvm_circuit::{
    arch::ExecutionBridge,
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
};
use openvm_instructions::riscv::RV64_CELL_BITS;
use openvm_mod_circuit_builder::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreAir, FieldExpressionExecutor,
    FieldExpressionFiller, FieldVariable,
};
use openvm_riscv_adapters::{
    Rv64VecHeapAdapterAir, Rv64VecHeapAdapterExecutor, Rv64VecHeapAdapterFiller,
};

use super::{ModularAir, ModularChip, ModularExecutor};
use crate::FieldExprVecHeapExecutor;

// TODO(rv64-u16): `mod-builder` `FieldExpression` chips (modular addsub/muldiv, fp2 addsub/muldiv,
// ecc weierstrass, pairing) cannot migrate to `limb_bits = 16` on BabyBear. The
// `check_carry_to_zero` AIR enforces `carry_abs_bits + limb_bits < F::bits() - 1` (= 30 for
// BabyBear); at `limb_bits = 16` with a full-width 256-bit prime, the modular reduction
// `(x ± y) - q·p` produces `carry_abs_bits = 18` because the cross-product `q·p` per output
// limb has max `≈ 2^32` (= `(2^16 - 1)²`, dominated by `min(q_limbs, num_limbs) *
// canonical_limb_max_abs²`), giving `max_overflow_bits ≈ 33` and `carry_bits = 17`. 18 + 16 = 34 ≫
// 30 — fails. This is a hard F-level guard, not a range-checker `decomp` setting. Only a larger
// prime field (e.g. Goldilocks) or `limb_bits ≤ 13` (which forfeits the cell savings) unblocks it.

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

pub fn get_modular_addsub_air<const BLOCKS: usize, const BLOCK_SIZE: usize>(
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
        Rv64VecHeapAdapterAir::new(
            exec_bridge,
            mem_bridge,
            bitwise_lookup_bus,
            pointer_max_bits,
        ),
        FieldExpressionCoreAir::new(expr, offset, local_opcode_idx, opcode_flag_idx),
    )
}

pub fn get_modular_addsub_step<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
    offset: usize,
) -> ModularExecutor<BLOCKS, BLOCK_SIZE> {
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

pub fn get_modular_addsub_chip<F, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    config: ExprBuilderConfig,
    mem_helper: SharedMemoryHelper<F>,
    range_checker: SharedVariableRangeCheckerChip,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_CELL_BITS>,
    pointer_max_bits: usize,
) -> ModularChip<F, BLOCKS, BLOCK_SIZE> {
    let (expr, local_opcode_idx, opcode_flag_idx) = gen_base_expr(config, range_checker.bus());
    ModularChip::new(
        FieldExpressionFiller::new(
            Rv64VecHeapAdapterFiller::new(pointer_max_bits, bitwise_lookup_chip),
            expr,
            local_opcode_idx,
            opcode_flag_idx,
            range_checker,
            false,
        ),
        mem_helper,
    )
}
