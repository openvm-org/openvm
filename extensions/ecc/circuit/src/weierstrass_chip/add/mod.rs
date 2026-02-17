use std::{cell::RefCell, rc::Rc};

use num_bigint::BigUint;
use num_traits::Zero;
use openvm_circuit::{
    arch::*,
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
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
use openvm_ecc_transpiler::Rv32WeierstrassOpcode;

use super::{WeierstrassAir, WeierstrassChip};

#[cfg(feature = "cuda")]
mod cuda;
mod execution;

#[cfg(feature = "cuda")]
pub use cuda::*;

pub fn ec_add_proj_expr(
    config: ExprBuilderConfig, // The coordinate field.
    range_bus: VariableRangeCheckerBus,
    a: BigUint,
    b: BigUint,
) -> FieldExpr {
    let b3 = (&b * 3u32) % &config.modulus;
    if a.is_zero() {
        ec_add_proj_a0_expr(config, range_bus, a, b, b3)
    } else {
        ec_add_proj_general_expr(config, range_bus, a.clone(), b, b3)
    }
}

fn ec_add_proj_a0_expr(
    config: ExprBuilderConfig,
    range_bus: VariableRangeCheckerBus,
    a: BigUint,
    b: BigUint,
    b3: BigUint,
) -> FieldExpr {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let x1 = ExprBuilder::new_input(builder.clone());
    let y1 = ExprBuilder::new_input(builder.clone());
    let z1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let y2 = ExprBuilder::new_input(builder.clone());
    let z2 = ExprBuilder::new_input(builder.clone());

    let b3_const = ExprBuilder::new_const(builder.clone(), b3);

    let t0 = x1.clone() * x2.clone();
    let t1 = y1.clone() * y2.clone();
    let t2 = z1.clone() * z2.clone();
    let t3 = (x1.clone() + y1.clone()) * (x2.clone() + y2.clone()) - t0.clone() - t1.clone();
    let t4 = (y1.clone() + z1.clone()) * (y2.clone() + z2.clone()) - t1.clone() - t2.clone();
    let y3 = (x1 + z1) * (x2 + z2) - t0.clone() - t2.clone();

    let x3 = t0.clone().int_mul(3);
    let t2 = t2 * b3_const.clone();
    let z3 = t1.clone() + t2.clone();
    let t1 = t1 - t2;
    let y3 = y3 * b3_const;

    let mut x3_out = t3.clone() * t1.clone() - t4.clone() * y3.clone();
    x3_out.save_output();

    let mut y3_out = t1 * z3.clone() + y3.clone() * x3.clone();
    y3_out.save_output();

    let mut z3_out = z3 * t4 + x3 * t3;
    z3_out.save_output();

    let builder = (*builder).borrow().clone();
    FieldExpr::new_with_setup_values(builder, range_bus, true, vec![a, b])
}

fn ec_add_proj_general_expr(
    config: ExprBuilderConfig,
    range_bus: VariableRangeCheckerBus,
    a_val: BigUint,
    b: BigUint,
    b3: BigUint,
) -> FieldExpr {
    config.check_valid();

    let builder = ExprBuilder::new(config, range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let x1 = ExprBuilder::new_input(builder.clone());
    let y1 = ExprBuilder::new_input(builder.clone());
    let z1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let y2 = ExprBuilder::new_input(builder.clone());
    let z2 = ExprBuilder::new_input(builder.clone());

    let a = ExprBuilder::new_const(builder.clone(), a_val.clone());
    let b3_const = ExprBuilder::new_const(builder.clone(), b3);

    let t0 = x1.clone() * x2.clone();
    let t1 = y1.clone() * y2.clone();
    let t2 = z1.clone() * z2.clone();
    let t3 = (x1.clone() + y1.clone()) * (x2.clone() + y2.clone()) - t0.clone() - t1.clone();
    let t4 = (x1.clone() + z1.clone()) * (x2.clone() + z2.clone()) - t0.clone() - t2.clone();
    let t5 = (y1 + z1) * (y2 + z2) - t1.clone() - t2.clone();

    let z3 = t2.clone() * b3_const.clone() + t4.clone() * a.clone() + t1.clone();
    let x3 = t1.clone().int_mul(2) - z3.clone();
    let t1 = t0.clone().int_mul(3) + t2.clone() * a.clone();
    let t2 = t0 - t2 * a.clone();
    let t4 = t4 * b3_const + t2 * a;

    let mut x3_out = t3.clone() * x3.clone() - t5.clone() * t4.clone();
    x3_out.save_output();

    let mut y3_out = x3.clone() * z3.clone() + t1.clone() * t4.clone();
    y3_out.save_output();

    let mut z3_out = t5 * z3 + t3 * t1;
    z3_out.save_output();

    let builder = (*builder).borrow().clone();
    FieldExpr::new_with_setup_values(builder, range_bus, true, vec![a_val, b])
}

pub use execution::EcAddExecutor;

fn gen_base_expr(
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    a: BigUint,
    b: BigUint,
) -> (FieldExpr, Vec<usize>) {
    let expr = ec_add_proj_expr(config, range_checker_bus, a, b);

    let local_opcode_idx = vec![
        Rv32WeierstrassOpcode::SW_EC_ADD_PROJ as usize,
        Rv32WeierstrassOpcode::SETUP_SW_EC_ADD_PROJ as usize,
    ];

    (expr, local_opcode_idx)
}

#[allow(clippy::too_many_arguments)]
pub fn get_ec_add_air<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    exec_bridge: ExecutionBridge,
    mem_bridge: MemoryBridge,
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
    pointer_max_bits: usize,
    offset: usize,
    a: BigUint,
    b: BigUint,
) -> WeierstrassAir<2, BLOCKS, BLOCK_SIZE> {
    let (expr, local_opcode_idx) = gen_base_expr(config, range_checker_bus, a, b);
    WeierstrassAir::new(
        Rv32VecHeapAdapterAir::new(
            exec_bridge,
            mem_bridge,
            bitwise_lookup_bus,
            pointer_max_bits,
        ),
        FieldExpressionCoreAir::new(expr.clone(), offset, local_opcode_idx.clone(), vec![]),
    )
}

pub fn get_ec_add_step<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
    offset: usize,
    a: BigUint,
    b: BigUint,
) -> EcAddExecutor<BLOCKS, BLOCK_SIZE> {
    let (expr, local_opcode_idx) = gen_base_expr(config, range_checker_bus, a, b);
    EcAddExecutor(FieldExpressionExecutor::new(
        Rv32VecHeapAdapterExecutor::new(pointer_max_bits),
        expr,
        offset,
        local_opcode_idx,
        vec![],
        "EcAdd",
    ))
}

pub fn get_ec_add_chip<F, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    config: ExprBuilderConfig,
    mem_helper: SharedMemoryHelper<F>,
    range_checker: SharedVariableRangeCheckerChip,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    pointer_max_bits: usize,
    a: BigUint,
    b: BigUint,
) -> WeierstrassChip<F, 2, BLOCKS, BLOCK_SIZE> {
    let (expr, local_opcode_idx) = gen_base_expr(config, range_checker.bus(), a, b);
    WeierstrassChip::new(
        FieldExpressionFiller::new(
            Rv32VecHeapAdapterFiller::new(pointer_max_bits, bitwise_lookup_chip),
            expr,
            local_opcode_idx,
            vec![],
            range_checker,
            false,
        ),
        mem_helper,
    )
}
