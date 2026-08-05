use std::{
    cell::RefCell,
    ops::{Deref, DerefMut},
    rc::Rc,
};

use num_bigint::BigUint;
use num_traits::One;
use openvm_circuit::{
    arch::*,
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_primitives::var_range::{
    SharedVariableRangeCheckerChip, VariableRangeCheckerBus,
};
use openvm_ecc_transpiler::WeierstrassOpcode;
use openvm_mod_circuit_builder::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreAir, FieldExpressionExecutor,
    FieldExpressionFiller, FieldExpressionProgram, FieldVariable,
};
use openvm_riscv_adapters::{VecHeapAdapterAir, VecHeapAdapterFiller};

use super::{WeierstrassAir, WeierstrassChip};

mod execution;

fn build_ec_double_ne_expr(
    config: ExprBuilderConfig,
    range_max_bits: usize,
    a_biguint: &BigUint,
) -> ExprBuilder {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let mut x1 = ExprBuilder::new_input(builder.clone());
    let mut y1 = ExprBuilder::new_input(builder.clone());
    let a = ExprBuilder::new_const(builder.clone(), a_biguint.clone());
    let is_double_flag = (*builder).borrow_mut().new_flag();
    // We need to prevent divide by zero when not double flag
    // (equivalently, when it is the setup opcode)
    let lambda_denom = FieldVariable::select(
        is_double_flag,
        &y1.int_mul(2),
        &ExprBuilder::new_const(builder.clone(), BigUint::one()),
    );
    let mut lambda = (x1.square().int_mul(3) + a) / lambda_denom;
    let mut x3 = lambda.square() - x1.int_mul(2);
    x3.save_output();
    let mut y3 = lambda * (x1 - x3.clone()) - y1;
    y3.save_output();

    let builder = builder.borrow().clone();
    builder
}

pub fn ec_double_ne_expr(
    config: ExprBuilderConfig,
    range_bus: VariableRangeCheckerBus,
    a_biguint: BigUint,
) -> FieldExpr {
    FieldExpr::new(
        ec_double_ne_program(config, range_bus.range_max_bits, a_biguint),
        range_bus,
    )
}

pub fn ec_double_ne_program(
    config: ExprBuilderConfig,
    range_max_bits: usize,
    a_biguint: BigUint,
) -> FieldExpressionProgram {
    FieldExpressionProgram::new_with_setup_values(
        build_ec_double_ne_expr(config, range_max_bits, &a_biguint),
        true,
        vec![a_biguint],
    )
}

/// `BLOCKS` is the number of memory blocks needed to represent one input or output point.
// Preflight executes this transition with fast native arithmetic.
#[derive(Clone)]
pub struct EcDoubleExecutor<const BLOCKS: usize> {
    pub(crate) inner: FieldExpressionExecutor,
}

impl<const BLOCKS: usize> EcDoubleExecutor<BLOCKS> {
    pub fn new(inner: FieldExpressionExecutor) -> Self {
        Self { inner }
    }
}

impl<const BLOCKS: usize> Deref for EcDoubleExecutor<BLOCKS> {
    type Target = FieldExpressionExecutor;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<const BLOCKS: usize> DerefMut for EcDoubleExecutor<BLOCKS> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

fn gen_base_program(
    config: ExprBuilderConfig,
    range_max_bits: usize,
    a_biguint: BigUint,
) -> (FieldExpressionProgram, Vec<usize>) {
    let program = ec_double_ne_program(config, range_max_bits, a_biguint);
    let local_opcode_idx = vec![
        WeierstrassOpcode::EC_DOUBLE as usize,
        WeierstrassOpcode::SETUP_EC_DOUBLE as usize,
    ];
    (program, local_opcode_idx)
}

#[allow(clippy::too_many_arguments)]
pub fn get_ec_double_air<const BLOCKS: usize>(
    exec_bridge: ExecutionBridge,
    mem_bridge: MemoryBridge,
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
    offset: usize,
    a_biguint: BigUint,
) -> WeierstrassAir<1, BLOCKS> {
    let (program, local_opcode_idx) =
        gen_base_program(config, range_checker_bus.range_max_bits, a_biguint);
    let expr = FieldExpr::new(program, range_checker_bus);
    WeierstrassAir::new(
        VecHeapAdapterAir::new(exec_bridge, mem_bridge, range_checker_bus, pointer_max_bits),
        FieldExpressionCoreAir::new(expr, offset, local_opcode_idx, vec![]),
    )
}

pub fn get_ec_double_executor<const BLOCKS: usize>(
    config: ExprBuilderConfig,
    range_max_bits: usize,
    offset: usize,
    a_biguint: BigUint,
) -> EcDoubleExecutor<BLOCKS> {
    let (program, local_opcode_idx) = gen_base_program(config, range_max_bits, a_biguint);
    EcDoubleExecutor::new(FieldExpressionExecutor::new(
        program,
        offset,
        local_opcode_idx,
        vec![],
        "EcDouble",
    ))
}

pub fn get_ec_double_chip<F, const BLOCKS: usize>(
    config: ExprBuilderConfig,
    mem_helper: SharedMemoryHelper<F>,
    range_checker: SharedVariableRangeCheckerChip,
    pointer_max_bits: usize,
    a_biguint: BigUint,
) -> WeierstrassChip<F, 1, BLOCKS> {
    let range_bus = range_checker.bus();
    let (program, local_opcode_idx) = gen_base_program(config, range_bus.range_max_bits, a_biguint);
    let expr = FieldExpr::new(program, range_bus);
    WeierstrassChip::new(
        FieldExpressionFiller::new(
            VecHeapAdapterFiller::new(pointer_max_bits),
            expr,
            local_opcode_idx,
            vec![],
            range_checker,
            true,
        ),
        mem_helper,
    )
}
