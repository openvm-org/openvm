use std::{cell::RefCell, rc::Rc};

use derive_more::derive::{Deref, DerefMut};
use num_bigint::BigUint;
use num_traits::One;
use openvm_circuit::{
    arch::ExecutionBridge,
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_derive::PreflightExecutor;
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
};
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_mod_circuit_builder::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreAir, FieldExpressionExecutor,
    FieldExpressionFiller,
};
use openvm_rv32_adapters::{
    Rv32VecHeapAdapterAir, Rv32VecHeapAdapterExecutor, Rv32VecHeapAdapterFiller,
};
use openvm_edwards_transpiler::Rv32EdwardsOpcode;

use super::{utils::jacobi, EdwardsAir, EdwardsChip};

#[cfg(feature = "cuda")]
mod cuda;
mod execution;

#[cfg(feature = "cuda")]
pub use cuda::*;

pub fn te_add_expr(
    config: ExprBuilderConfig, // The coordinate field.
    range_bus: VariableRangeCheckerBus,
    a_biguint: BigUint,
    d_biguint: BigUint,
) -> FieldExpr {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let x1 = ExprBuilder::new_input(builder.clone());
    let y1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let y2 = ExprBuilder::new_input(builder.clone());
    let a = ExprBuilder::new_const(builder.clone(), a_biguint.clone());
    let d = ExprBuilder::new_const(builder.clone(), d_biguint.clone());
    let one = ExprBuilder::new_const(builder.clone(), BigUint::one());

    let x1y2 = x1.clone() * y2.clone();
    let x2y1 = x2.clone() * y1.clone();
    let y1y2 = y1 * y2;
    let x1x2 = x1 * x2;
    let dx1x2y1y2 = d * x1x2.clone() * y1y2.clone();

    let mut x3 = (x1y2 + x2y1) / (one.clone() + dx1x2y1y2.clone());
    let mut y3 = (y1y2 - a * x1x2) / (one - dx1x2y1y2);

    x3.save_output();
    y3.save_output();

    let builder = (*builder).borrow().clone();

    FieldExpr::new_with_setup_values(builder, range_bus, true, vec![a_biguint, d_biguint])
}

#[derive(Clone, PreflightExecutor, Deref, DerefMut)]
pub struct TeAddExecutor<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    pub(crate)  FieldExpressionExecutor<
        Rv32VecHeapAdapterExecutor<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
    >,
);

fn gen_base_expr(
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    a_biguint: BigUint,
    d_biguint: BigUint,
) -> (FieldExpr, Vec<usize>) {
    let expr = te_add_expr(config, range_checker_bus, a_biguint, d_biguint);

    let local_opcode_idx = vec![
        Rv32EdwardsOpcode::TE_ADD as usize,
        Rv32EdwardsOpcode::SETUP_TE_ADD as usize,
    ];

    (expr, local_opcode_idx)
}

#[allow(clippy::too_many_arguments)]
pub fn get_te_add_air<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    exec_bridge: ExecutionBridge,
    mem_bridge: MemoryBridge,
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
    pointer_max_bits: usize,
    offset: usize,
    a_biguint: BigUint,
    d_biguint: BigUint,
) -> EdwardsAir<2, BLOCKS, BLOCK_SIZE> {
    // Ensure that the addition operation is complete
    assert!(jacobi(&a_biguint.clone().into(), &config.modulus.clone().into()) == 1);
    assert!(jacobi(&d_biguint.clone().into(), &config.modulus.clone().into()) == -1);

    let (expr, local_opcode_idx) = gen_base_expr(config, range_checker_bus, a_biguint, d_biguint);
    EdwardsAir::new(
        Rv32VecHeapAdapterAir::new(
            exec_bridge,
            mem_bridge,
            bitwise_lookup_bus,
            pointer_max_bits,
        ),
        FieldExpressionCoreAir::new(expr.clone(), offset, local_opcode_idx.clone(), vec![]),
    )
}

pub fn get_te_add_step<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
    offset: usize,
    a_biguint: BigUint,
    d_biguint: BigUint,
) -> TeAddExecutor<BLOCKS, BLOCK_SIZE> {
    // Ensure that the addition operation is complete
    assert!(jacobi(&a_biguint.clone().into(), &config.modulus.clone().into()) == 1);
    assert!(jacobi(&d_biguint.clone().into(), &config.modulus.clone().into()) == -1);

    let (expr, local_opcode_idx) = gen_base_expr(config, range_checker_bus, a_biguint, d_biguint);
    TeAddExecutor(FieldExpressionExecutor::new(
        Rv32VecHeapAdapterExecutor::new(pointer_max_bits),
        expr,
        offset,
        local_opcode_idx,
        vec![],
        "TeAdd",
    ))
}

pub fn get_te_add_chip<F, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    config: ExprBuilderConfig,
    mem_helper: SharedMemoryHelper<F>,
    range_checker: SharedVariableRangeCheckerChip,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    pointer_max_bits: usize,
    a_biguint: BigUint,
    d_biguint: BigUint,
) -> EdwardsChip<F, 2, BLOCKS, BLOCK_SIZE> {
    // Ensure that the addition operation is complete
    assert!(jacobi(&a_biguint.clone().into(), &config.modulus.clone().into()) == 1);
    assert!(jacobi(&d_biguint.clone().into(), &config.modulus.clone().into()) == -1);

    let (expr, local_opcode_idx) = gen_base_expr(config, range_checker.bus(), a_biguint, d_biguint);
    EdwardsChip::new(
        FieldExpressionFiller::new(
            Rv32VecHeapAdapterFiller::new(pointer_max_bits, bitwise_lookup_chip),
            expr,
            local_opcode_idx,
            vec![],
            range_checker,
            true,
        ),
        mem_helper,
    )
}
