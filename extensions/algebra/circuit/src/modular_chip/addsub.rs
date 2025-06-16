use std::{cell::RefCell, rc::Rc};

use openvm_algebra_transpiler::Rv32ModularArithmeticOpcode;
use openvm_circuit::{
    arch::ExecutionBridge,
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_derive::{InsExecutorE1, InstructionExecutor};
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    Chip, ChipUsageGetter,
};
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_mod_circuit_builder::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreAir, FieldVariable,
};
use openvm_rv32_adapters::{Rv32VecHeapAdapterAir, Rv32VecHeapAdapterStep};
use openvm_stark_backend::p3_field::PrimeField32;

use super::{ModularAir, ModularChip, ModularStep};

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
    let is_add_flag = builder.borrow_mut().new_flag();
    let is_sub_flag = builder.borrow_mut().new_flag();
    let x5 = FieldVariable::select(is_sub_flag, &x4, &x1);
    let mut x6 = FieldVariable::select(is_add_flag, &x3, &x5);
    x6.save_output();
    let builder = builder.borrow().clone();

    (
        FieldExpr::new(builder, range_bus, true),
        is_add_flag,
        is_sub_flag,
    )
}

#[derive(Chip, ChipUsageGetter, InstructionExecutor, InsExecutorE1)]
pub struct ModularAddSubChip<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    pub ModularChip<F, BLOCKS, BLOCK_SIZE>,
);

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>
    ModularAddSubChip<F, BLOCKS, BLOCK_SIZE>
{
    pub fn new(
        execution_bridge: ExecutionBridge,
        memory_bridge: MemoryBridge,
        mem_helper: SharedMemoryHelper<F>,
        pointer_max_bits: usize,
        config: ExprBuilderConfig,
        offset: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
        range_checker: SharedVariableRangeCheckerChip,
    ) -> Self {
        let (expr, is_add_flag, is_sub_flag) = addsub_expr(config, range_checker.bus());

        let local_opcode_idx = vec![
            Rv32ModularArithmeticOpcode::ADD as usize,
            Rv32ModularArithmeticOpcode::SUB as usize,
            Rv32ModularArithmeticOpcode::SETUP_ADDSUB as usize,
        ];
        let opcode_flag_idx = vec![is_add_flag, is_sub_flag];
        let air = ModularAir::new(
            Rv32VecHeapAdapterAir::new(
                execution_bridge,
                memory_bridge,
                bitwise_lookup_chip.bus(),
                pointer_max_bits,
            ),
            FieldExpressionCoreAir::new(
                expr.clone(),
                offset,
                local_opcode_idx.clone(),
                opcode_flag_idx.clone(),
            ),
        );

        let step = ModularStep::new(
            Rv32VecHeapAdapterStep::new(pointer_max_bits, bitwise_lookup_chip),
            expr,
            offset,
            local_opcode_idx,
            opcode_flag_idx,
            range_checker,
            "ModularAddSub",
            false,
        );
        Self(ModularChip::new(air, step, mem_helper))
    }
}
