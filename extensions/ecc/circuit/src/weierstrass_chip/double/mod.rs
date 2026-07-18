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
use openvm_ecc_transpiler::Rv64WeierstrassOpcode;
use openvm_mod_circuit_builder::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreAir, FieldExpressionExecutor,
    FieldExpressionFiller, FieldVariable,
};
use openvm_riscv_adapters::{
    Rv64VecHeapAdapterAir, Rv64VecHeapAdapterExecutor, Rv64VecHeapAdapterFiller,
};

use super::{
    curves::{get_curve_type, CurveType},
    WeierstrassAir, WeierstrassChip,
};

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
    FieldExpr::new_with_setup_values(
        build_ec_double_ne_expr(config, range_bus.range_max_bits, &a_biguint),
        range_bus,
        true,
        vec![a_biguint],
    )
}

pub fn ec_double_ne_builder(
    config: ExprBuilderConfig,
    range_max_bits: usize,
    a_biguint: BigUint,
) -> ExprBuilder {
    let mut builder = build_ec_double_ne_expr(config, range_max_bits, &a_biguint);
    builder.finalize(true);
    builder
}

/// `BLOCKS` is the number of memory blocks needed to represent one input or output point.
// Note: PreflightExecutor is implemented manually in preflight.rs with fast native arithmetic
#[derive(Clone)]
pub struct EcDoubleExecutor<const BLOCKS: usize> {
    pub(crate) inner: FieldExpressionExecutor<Rv64VecHeapAdapterExecutor<1, BLOCKS, BLOCKS>>,
    pub(crate) cached_curve_type: Option<CurveType>,
}

impl<const BLOCKS: usize> EcDoubleExecutor<BLOCKS> {
    pub fn new(
        inner: FieldExpressionExecutor<Rv64VecHeapAdapterExecutor<1, BLOCKS, BLOCKS>>,
    ) -> Self {
        let cached_curve_type = inner
            .expr
            .setup_values
            .first()
            .and_then(|a| get_curve_type(&inner.expr.prime, a));
        Self {
            inner,
            cached_curve_type,
        }
    }
}

impl<const BLOCKS: usize> Deref for EcDoubleExecutor<BLOCKS> {
    type Target = FieldExpressionExecutor<Rv64VecHeapAdapterExecutor<1, BLOCKS, BLOCKS>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<const BLOCKS: usize> DerefMut for EcDoubleExecutor<BLOCKS> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

fn gen_base_expr(
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    a_biguint: BigUint,
) -> (FieldExpr, Vec<usize>) {
    let expr = ec_double_ne_expr(config, range_checker_bus, a_biguint);
    let local_opcode_idx = vec![
        Rv64WeierstrassOpcode::EC_DOUBLE as usize,
        Rv64WeierstrassOpcode::SETUP_EC_DOUBLE as usize,
    ];
    (expr, local_opcode_idx)
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
    let (expr, local_opcode_idx) = gen_base_expr(config, range_checker_bus, a_biguint);
    WeierstrassAir::new(
        Rv64VecHeapAdapterAir::new(exec_bridge, mem_bridge, range_checker_bus, pointer_max_bits),
        FieldExpressionCoreAir::new(expr.clone(), offset, local_opcode_idx.clone(), vec![]),
    )
}

pub fn get_ec_double_executor<const BLOCKS: usize>(
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
    offset: usize,
    a_biguint: BigUint,
) -> EcDoubleExecutor<BLOCKS> {
    let (expr, local_opcode_idx) = gen_base_expr(config, range_checker_bus, a_biguint);
    EcDoubleExecutor::new(FieldExpressionExecutor::new(
        Rv64VecHeapAdapterExecutor::new(pointer_max_bits),
        expr,
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
    let (expr, local_opcode_idx) = gen_base_expr(config, range_checker.bus(), a_biguint);
    WeierstrassChip::new(
        FieldExpressionFiller::new(
            Rv64VecHeapAdapterFiller::new(pointer_max_bits, range_checker.clone()),
            expr,
            local_opcode_idx,
            vec![],
            range_checker,
            true,
        ),
        mem_helper,
    )
}
