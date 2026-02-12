use std::{cell::RefCell, rc::Rc};

use derive_more::derive::{Deref, DerefMut};
use num_bigint::BigUint;
use num_traits::Zero;
use openvm_circuit::{
    arch::*,
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
use openvm_ecc_transpiler::Rv32WeierstrassOpcode;

use super::{WeierstrassAir, WeierstrassChip};

#[cfg(feature = "cuda")]
mod cuda;
mod execution;

#[cfg(feature = "cuda")]
pub use cuda::*;

pub fn ec_double_proj_expr(
    config: ExprBuilderConfig, // The coordinate field.
    range_bus: VariableRangeCheckerBus,
    a: BigUint,
    b: BigUint,
) -> FieldExpr {
    let b3 = (&b * 3u32) % &config.modulus;
    if a.is_zero() {
        ec_double_proj_a0_expr(config, range_bus, a, b, b3)
    } else {
        ec_double_proj_general_expr(config, range_bus, a.clone(), b, b3)
    }
}

fn ec_double_proj_a0_expr(
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

    let b3_const = ExprBuilder::new_const(builder.clone(), b3);

    let t0 = y1.clone() * y1.clone();
    let z3 = t0.clone().int_mul(8);
    let t1 = y1.clone() * z1.clone();
    let t2 = z1.clone() * z1 * b3_const;

    let x3 = t2.clone() * z3.clone();
    let y3 = t0.clone() + t2.clone();
    let t0 = t0 - t2.clone().int_mul(3);

    let mut x3_out = (t0.clone() * (x1 * y1)).int_mul(2);
    x3_out.save_output();

    let mut y3_out = x3 + t0.clone() * y3;
    y3_out.save_output();

    let mut z3_out = t1 * z3;
    z3_out.save_output();

    let builder = (*builder).borrow().clone();
    FieldExpr::new_with_setup_values(builder, range_bus, true, vec![a, b])
}

fn ec_double_proj_general_expr(
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

    let a = ExprBuilder::new_const(builder.clone(), a_val.clone());
    let b3_const = ExprBuilder::new_const(builder.clone(), b3);

    let t0 = x1.clone() * x1.clone();
    let t1 = y1.clone() * y1.clone();
    let t2 = z1.clone() * z1.clone();
    let t3 = (x1.clone() * y1.clone()).int_mul(2);
    let z3 = (x1 * z1.clone()).int_mul(2);

    let y3 = z3.clone() * a.clone() + t2.clone() * b3_const.clone();
    let x3 = t3.clone() * (t1.clone() - y3.clone());
    let y3_sq = y3.clone() * y3;
    let t2 = t2 * a.clone();
    let t3 = (t0.clone() - t2.clone()) * a + z3 * b3_const;

    let t2_yz = (y1 * z1).int_mul(2);
    let mut x3_out = x3 - t2_yz.clone() * t3.clone();
    x3_out.save_output();

    let mut y3_out = t1.clone() * t1.clone() - y3_sq + (t0.clone().int_mul(3) + t2) * t3.clone();
    y3_out.save_output();

    let mut z3_out = (t2_yz * t1).int_mul(4);
    z3_out.save_output();

    let builder = (*builder).borrow().clone();
    FieldExpr::new_with_setup_values(builder, range_bus, true, vec![a_val, b])
}

/// BLOCK_SIZE: how many cells do we read at a time, must be a power of 2.
/// BLOCKS: how many blocks do we need to represent one input or output
/// For example, for bls12_381, BLOCK_SIZE = 16, each element has 3 blocks and with three elements
/// per input ProjectivePoint, BLOCKS = 9. For secp256k1, BLOCK_SIZE = 32, BLOCKS = 3.
#[derive(Clone, PreflightExecutor, Deref, DerefMut)]
pub struct EcDoubleExecutor<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    FieldExpressionExecutor<Rv32VecHeapAdapterExecutor<1, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>>,
);

fn gen_base_expr(
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    a: BigUint,
    b: BigUint,
) -> (FieldExpr, Vec<usize>) {
    let expr = ec_double_proj_expr(config, range_checker_bus, a, b);

    let local_opcode_idx = vec![
        Rv32WeierstrassOpcode::SW_EC_DOUBLE_PROJ as usize,
        Rv32WeierstrassOpcode::SETUP_SW_EC_DOUBLE_PROJ as usize,
    ];

    (expr, local_opcode_idx)
}

#[allow(clippy::too_many_arguments)]
pub fn get_ec_double_air<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    exec_bridge: ExecutionBridge,
    mem_bridge: MemoryBridge,
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
    pointer_max_bits: usize,
    offset: usize,
    a: BigUint,
    b: BigUint,
) -> WeierstrassAir<1, BLOCKS, BLOCK_SIZE> {
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

pub fn get_ec_double_step<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
    offset: usize,
    a: BigUint,
    b: BigUint,
) -> EcDoubleExecutor<BLOCKS, BLOCK_SIZE> {
    let (expr, local_opcode_idx) = gen_base_expr(config, range_checker_bus, a, b);
    EcDoubleExecutor(FieldExpressionExecutor::new(
        Rv32VecHeapAdapterExecutor::new(pointer_max_bits),
        expr,
        offset,
        local_opcode_idx,
        vec![],
        "EcDouble",
    ))
}

pub fn get_ec_double_chip<F, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    config: ExprBuilderConfig,
    mem_helper: SharedMemoryHelper<F>,
    range_checker: SharedVariableRangeCheckerChip,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    pointer_max_bits: usize,
    a: BigUint,
    b: BigUint,
) -> WeierstrassChip<F, 1, BLOCKS, BLOCK_SIZE> {
    let (expr, local_opcode_idx) = gen_base_expr(config, range_checker.bus(), a, b);
    WeierstrassChip::new(
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
