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
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreAir, FieldVariable, SymbolicExpr,
};
use openvm_rv32_adapters::{Rv32VecHeapAdapterAir, Rv32VecHeapAdapterStep};
use openvm_stark_backend::p3_field::PrimeField32;

use super::{ModularAir, ModularChip, ModularStep};

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

#[derive(Chip, ChipUsageGetter, InstructionExecutor, InsExecutorE1)]
pub struct ModularMulDivChip<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    pub ModularChip<F, BLOCKS, BLOCK_SIZE>,
);

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>
    ModularMulDivChip<F, BLOCKS, BLOCK_SIZE>
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
        let (expr, is_mul_flag, is_div_flag) = muldiv_expr(config, range_checker.bus());

        let local_opcode_idx = vec![
            Rv32ModularArithmeticOpcode::MUL as usize,
            Rv32ModularArithmeticOpcode::DIV as usize,
            Rv32ModularArithmeticOpcode::SETUP_MULDIV as usize,
        ];
        let opcode_flag_idx = vec![is_mul_flag, is_div_flag];
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
            "ModularMulDiv",
            false,
        );
        Self(ModularChip::new(air, step, mem_helper))
    }
}
