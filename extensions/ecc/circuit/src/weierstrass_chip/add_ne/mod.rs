use std::{
    cell::RefCell,
    ops::{Deref, DerefMut},
    rc::Rc,
};

use openvm_algebra_circuit::fields::{get_field_type, FieldType};
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
    FieldExpressionFiller, FieldExpressionProgram,
};
use openvm_riscv_adapters::{
    Rv64VecHeapAdapterAir, Rv64VecHeapAdapterExecutor, Rv64VecHeapAdapterFiller,
};

use super::{WeierstrassAir, WeierstrassChip};

mod execution;

// Assumes that (x1, y1), (x2, y2) both lie on the curve and are not the identity point.
// Further assumes that x1, x2 are not equal in the coordinate field.
fn build_ec_add_ne_expr(config: ExprBuilderConfig, range_max_bits: usize) -> ExprBuilder {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let x1 = ExprBuilder::new_input(builder.clone());
    let y1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let y2 = ExprBuilder::new_input(builder.clone());
    let mut lambda = (y2 - y1.clone()) / (x2.clone() - x1.clone());
    let mut x3 = lambda.square() - x1.clone() - x2;
    x3.save_output();
    let mut y3 = lambda * (x1 - x3.clone()) - y1;
    y3.save_output();

    let builder = builder.borrow().clone();
    builder
}

pub fn ec_add_ne_expr(
    config: ExprBuilderConfig, // The coordinate field.
    range_bus: VariableRangeCheckerBus,
) -> FieldExpr {
    FieldExpr::new(
        ec_add_ne_program(config, range_bus.range_max_bits),
        range_bus,
    )
}

pub fn ec_add_ne_program(
    config: ExprBuilderConfig,
    range_max_bits: usize,
) -> FieldExpressionProgram {
    FieldExpressionProgram::new(build_ec_add_ne_expr(config, range_max_bits), true)
}

/// `BLOCKS` is the number of memory blocks needed to represent one input or output point.
// Note: PreflightExecutor is implemented manually in preflight.rs with fast native arithmetic
#[derive(Clone)]
pub struct EcAddNeExecutor<const BLOCKS: usize> {
    pub(crate) inner: FieldExpressionExecutor<Rv64VecHeapAdapterExecutor<2, BLOCKS, BLOCKS>>,
    pub(crate) cached_field_type: Option<FieldType>,
}

impl<const BLOCKS: usize> EcAddNeExecutor<BLOCKS> {
    pub fn new(
        inner: FieldExpressionExecutor<Rv64VecHeapAdapterExecutor<2, BLOCKS, BLOCKS>>,
    ) -> Self {
        let cached_field_type = get_field_type(inner.program().prime());
        Self {
            inner,
            cached_field_type,
        }
    }
}

impl<const BLOCKS: usize> Deref for EcAddNeExecutor<BLOCKS> {
    type Target = FieldExpressionExecutor<Rv64VecHeapAdapterExecutor<2, BLOCKS, BLOCKS>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<const BLOCKS: usize> DerefMut for EcAddNeExecutor<BLOCKS> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

fn gen_base_program(
    config: ExprBuilderConfig,
    range_max_bits: usize,
) -> (FieldExpressionProgram, Vec<usize>) {
    let program = ec_add_ne_program(config, range_max_bits);
    let local_opcode_idx = vec![
        Rv64WeierstrassOpcode::EC_ADD_NE as usize,
        Rv64WeierstrassOpcode::SETUP_EC_ADD_NE as usize,
    ];
    (program, local_opcode_idx)
}

pub fn get_ec_addne_air<const BLOCKS: usize>(
    exec_bridge: ExecutionBridge,
    mem_bridge: MemoryBridge,
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
    offset: usize,
) -> WeierstrassAir<2, BLOCKS> {
    let (program, local_opcode_idx) = gen_base_program(config, range_checker_bus.range_max_bits);
    let expr = FieldExpr::new(program, range_checker_bus);
    WeierstrassAir::new(
        Rv64VecHeapAdapterAir::new(exec_bridge, mem_bridge, range_checker_bus, pointer_max_bits),
        FieldExpressionCoreAir::new(expr, offset, local_opcode_idx, vec![]),
    )
}

pub fn get_ec_addne_executor<const BLOCKS: usize>(
    config: ExprBuilderConfig,
    range_max_bits: usize,
    pointer_max_bits: usize,
    offset: usize,
) -> EcAddNeExecutor<BLOCKS> {
    let (program, local_opcode_idx) = gen_base_program(config, range_max_bits);
    EcAddNeExecutor::new(FieldExpressionExecutor::new(
        Rv64VecHeapAdapterExecutor::new(pointer_max_bits),
        program,
        offset,
        local_opcode_idx,
        vec![],
        "EcAddNe",
    ))
}

pub fn get_ec_addne_chip<F, const BLOCKS: usize>(
    config: ExprBuilderConfig,
    mem_helper: SharedMemoryHelper<F>,
    range_checker: SharedVariableRangeCheckerChip,
    pointer_max_bits: usize,
) -> WeierstrassChip<F, 2, BLOCKS> {
    let range_bus = range_checker.bus();
    let (program, local_opcode_idx) = gen_base_program(config, range_bus.range_max_bits);
    let expr = FieldExpr::new(program, range_bus);
    WeierstrassChip::new(
        FieldExpressionFiller::new(
            Rv64VecHeapAdapterFiller::new(pointer_max_bits, range_checker.clone()),
            expr,
            local_opcode_idx,
            vec![],
            range_checker,
            false,
        ),
        mem_helper,
    )
}
