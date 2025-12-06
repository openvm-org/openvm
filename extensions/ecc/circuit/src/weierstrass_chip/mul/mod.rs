use std::{cell::RefCell, rc::Rc};

use derive_more::derive::{Deref, DerefMut};
use num_bigint::BigUint;
use openvm_circuit::{
    arch::*,
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_derive::PreflightExecutor;
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
};
use openvm_ecc_transpiler::Rv32WeierstrassOpcode;
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_mod_circuit_builder::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreAir, FieldExpressionExecutor,
    FieldExpressionFiller,
};
use openvm_rv32_adapters::{Rv32EcMulAdapterAir, Rv32EcMulAdapterExecutor, Rv32EcMulAdapterFiller};

mod execution;

/// dummy implementation for now
pub fn ec_mul_expr(
    config: ExprBuilderConfig, // The coordinate field.
    range_bus: VariableRangeCheckerBus,
    a_biguint: BigUint,
) -> FieldExpr {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    // Create inputs
    let _scalar = ExprBuilder::new_input(builder.clone());
    let x1 = ExprBuilder::new_input(builder.clone());
    let y1 = ExprBuilder::new_input(builder.clone());

    // Create dummy outputs: result x and result y
    // Note: The actual computation is done natively, but we need these for the AIR structure
    let mut x_out = x1.clone(); // Dummy - actual computation happens in native code
    x_out.save_output();
    let mut y_out = y1.clone(); // Dummy - actual computation happens in native code
    y_out.save_output();

    let builder = (*builder).borrow().clone();
    FieldExpr::new_with_setup_values(builder, range_bus, true, vec![a_biguint])
}

#[derive(Clone, PreflightExecutor, Deref, DerefMut)]
pub struct EcMulExecutor<
    const BLOCKS_PER_SCALAR: usize,
    const BLOCKS_PER_POINT: usize,
    const SCALAR_SIZE: usize,
    const POINT_SIZE: usize,
>(
    FieldExpressionExecutor<
        Rv32EcMulAdapterExecutor<BLOCKS_PER_SCALAR, BLOCKS_PER_POINT, SCALAR_SIZE, POINT_SIZE>,
    >,
);

pub type WeierstrassEcMulAir<
    const BLOCKS_PER_SCALAR: usize,
    const BLOCKS_PER_POINT: usize,
    const SCALAR_SIZE: usize,
    const POINT_SIZE: usize,
> = VmAirWrapper<
    Rv32EcMulAdapterAir<BLOCKS_PER_SCALAR, BLOCKS_PER_POINT, SCALAR_SIZE, POINT_SIZE>,
    FieldExpressionCoreAir,
>;

pub type WeierstrassEcMulChip<
    F,
    const BLOCKS_PER_SCALAR: usize,
    const BLOCKS_PER_POINT: usize,
    const SCALAR_SIZE: usize,
    const POINT_SIZE: usize,
> = VmChipWrapper<
    F,
    FieldExpressionFiller<
        Rv32EcMulAdapterFiller<BLOCKS_PER_SCALAR, BLOCKS_PER_POINT, SCALAR_SIZE, POINT_SIZE>,
    >,
>;

fn gen_base_expr(
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    a_biguint: BigUint,
) -> (FieldExpr, Vec<usize>) {
    let expr = ec_mul_expr(config, range_checker_bus, a_biguint);

    let local_opcode_idx = vec![
        Rv32WeierstrassOpcode::EC_MUL as usize,
        Rv32WeierstrassOpcode::SETUP_EC_MUL as usize,
    ];

    (expr, local_opcode_idx)
}

pub fn get_ec_mul_air<
    const BLOCKS_PER_SCALAR: usize,
    const BLOCKS_PER_POINT: usize,
    const SCALAR_SIZE: usize,
    const POINT_SIZE: usize,
>(
    exec_bridge: ExecutionBridge,
    mem_bridge: MemoryBridge,
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
    pointer_max_bits: usize,
    offset: usize,
    a_biguint: BigUint,
) -> WeierstrassEcMulAir<BLOCKS_PER_SCALAR, BLOCKS_PER_POINT, SCALAR_SIZE, POINT_SIZE> {
    let (expr, local_opcode_idx) = gen_base_expr(config, range_checker_bus, a_biguint);
    WeierstrassEcMulAir::new(
        Rv32EcMulAdapterAir::new(
            exec_bridge,
            mem_bridge,
            bitwise_lookup_bus,
            pointer_max_bits,
        ),
        FieldExpressionCoreAir::new(expr.clone(), offset, local_opcode_idx.clone(), vec![]),
    )
}

pub fn get_ec_mul_step<
    const BLOCKS_PER_SCALAR: usize,
    const BLOCKS_PER_POINT: usize,
    const SCALAR_SIZE: usize,
    const POINT_SIZE: usize,
>(
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
    offset: usize,
    a_biguint: BigUint,
) -> EcMulExecutor<BLOCKS_PER_SCALAR, BLOCKS_PER_POINT, SCALAR_SIZE, POINT_SIZE> {
    let (expr, local_opcode_idx) = gen_base_expr(config, range_checker_bus, a_biguint);
    EcMulExecutor(FieldExpressionExecutor::new(
        Rv32EcMulAdapterExecutor::new(pointer_max_bits),
        expr,
        offset,
        local_opcode_idx,
        vec![],
        "EcMul",
    ))
}

pub fn get_ec_mul_chip<
    F,
    const BLOCKS_PER_SCALAR: usize,
    const BLOCKS_PER_POINT: usize,
    const SCALAR_SIZE: usize,
    const POINT_SIZE: usize,
>(
    config: ExprBuilderConfig,
    mem_helper: SharedMemoryHelper<F>,
    range_checker: SharedVariableRangeCheckerChip,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    pointer_max_bits: usize,
    a_biguint: BigUint,
) -> WeierstrassEcMulChip<F, BLOCKS_PER_SCALAR, BLOCKS_PER_POINT, SCALAR_SIZE, POINT_SIZE> {
    let (expr, local_opcode_idx) = gen_base_expr(config, range_checker.bus(), a_biguint);
    WeierstrassEcMulChip::new(
        FieldExpressionFiller::new(
            Rv32EcMulAdapterFiller::new(pointer_max_bits, bitwise_lookup_chip),
            expr,
            local_opcode_idx,
            vec![],
            range_checker,
            false,
        ),
        mem_helper,
    )
}
