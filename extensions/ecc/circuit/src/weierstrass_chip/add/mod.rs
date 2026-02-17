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

    let mut x1 = ExprBuilder::new_input(builder.clone());
    let mut y1 = ExprBuilder::new_input(builder.clone());
    let mut z1 = ExprBuilder::new_input(builder.clone());
    let mut x2 = ExprBuilder::new_input(builder.clone());
    let mut y2 = ExprBuilder::new_input(builder.clone());
    let mut z2 = ExprBuilder::new_input(builder.clone());

    let b3_const = ExprBuilder::new_const(builder.clone(), b3);

    // Algorithm 7 from ePrint 2015/1060 (complete addition for a=0).
    //
    // Column count (secp256k1, 256-bit): 1761 total (34.0% reduction from naive 2668).
    //
    // Optimization strategy: each save() creates 1 intermediate variable, which adds
    // N extra AIR columns (variable limbs + quotient limbs + carry limbs). Fewer saves
    // = fewer columns = faster proving. Saves are called explicitly throughout to avoid
    // suboptimal automatic save_if_overflow calls. Three techniques are used:
    //   1. Combined mul-sub: (A+B)·(C+D) - A·C - B·D in one save instead of mul then sub
    //   2. Inlined intermediates: values folded into output expressions without saving
    //   3. Degree-2 outputs: two muls combined into a single save_output()

    // Spec steps 1-3
    let mut t0 = (&mut x1).mul(&mut x2);
    t0.save();
    let mut t1 = (&mut y1).mul(&mut y2);
    t1.save();
    let mut t2 = (&mut z1).mul(&mut z2);
    t2.save();

    // Spec steps 4-8: t3 = (X1+Y1)·(X2+Y2) - t0 - t1 = X1·Y2 + X2·Y1
    // [combined mul-sub]
    let mut t3_lhs = x1.clone() + y1.clone();
    let mut t3_rhs = x2.clone() + y2.clone();
    let mut t3 = (&mut t3_lhs).mul(&mut t3_rhs) - t0.clone() - t1.clone();
    t3.save();

    // Spec steps 9-13: t4 = (Y1+Z1)·(Y2+Z2) - t1 - t2 = Y1·Z2 + Y2·Z1
    // [combined mul-sub]
    let mut t4_lhs = y1.clone() + z1.clone();
    let mut t4_rhs = y2.clone() + z2.clone();
    let mut t4 = (&mut t4_lhs).mul(&mut t4_rhs) - t1.clone() - t2.clone();
    t4.save();

    // Spec steps 14-18: y3 = (X1+Z1)·(X2+Z2) - t0 - t2 = X1·Z2 + X2·Z1
    // [combined mul-sub]
    let mut y3_lhs = x1 + z1;
    let mut y3_rhs = x2 + z2;
    let mut y3 = (&mut y3_lhs).mul(&mut y3_rhs) - t0.clone() - t2.clone();
    y3.save();

    // Spec steps 19-20: t0 = 3·t0
    t0 = t0.int_mul(3);
    t0.save();

    // Spec step 21: t2 = 3b·t2 (using b3 = 3b constant directly)
    t2 = t2 * b3_const.clone();
    t2.save();

    // Spec steps 22-23: z3 = t1+t2, t1 = t1-t2
    // NOT saved — inlined into the output expressions below, saving 2 variables.
    let mut z3 = t1.clone() + t2.clone();
    t1 = t1 - t2;

    // Spec step 24: y3 = 3b·y3
    y3 = y3 * b3_const;
    y3.save();

    // Spec steps 25-33: each output combines 2 spec muls into a single degree-2
    // save_output(), saving 3 variables vs the spec's approach.

    // Spec steps 25-27: X3 = t3·t1 - t4·y3
    let t3_mul_t1 = (&mut t3).mul(&mut t1);
    let t4_mul_y3 = (&mut t4).mul(&mut y3);
    let mut x3_out = t3_mul_t1 - t4_mul_y3;
    x3_out.save_output();

    // Spec steps 28-30: Y3 = t1·z3 + y3·t0
    let t1_mul_z3 = (&mut t1).mul(&mut z3);
    let y3_mul_t0 = (&mut y3).mul(&mut t0);
    let mut y3_out = t1_mul_z3 + y3_mul_t0;
    y3_out.save_output();

    // Spec steps 31-33: Z3 = z3·t4 + t0·t3
    let z3_mul_t4 = (&mut z3).mul(&mut t4);
    let t0_mul_t3 = (&mut t0).mul(&mut t3);
    let mut z3_out = z3_mul_t4 + t0_mul_t3;
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

    let mut x1 = ExprBuilder::new_input(builder.clone());
    let mut y1 = ExprBuilder::new_input(builder.clone());
    let mut z1 = ExprBuilder::new_input(builder.clone());
    let mut x2 = ExprBuilder::new_input(builder.clone());
    let mut y2 = ExprBuilder::new_input(builder.clone());
    let mut z2 = ExprBuilder::new_input(builder.clone());

    let a = ExprBuilder::new_const(builder.clone(), a_val.clone());
    let b3_const = ExprBuilder::new_const(builder.clone(), b3);

    // Algorithm 1 from ePrint 2015/1060 (complete addition for general a).
    //
    // Column count (secp256r1, 256-bit): 2081 total (38.3% reduction from naive 3371).
    //
    // Optimization strategy: each save() creates 1 intermediate variable, which adds
    // N extra AIR columns (variable limbs + quotient limbs + carry limbs). Fewer saves
    // = fewer columns = faster proving. Saves are called explicitly throughout to avoid
    // suboptimal automatic save_if_overflow calls. Three techniques are used:
    //   1. Combined mul-sub: (A+B)·(C+D) - A·C - B·D in one save instead of mul then sub
    //   2. Inlined intermediates: values folded into later expressions without saving
    //   3. Degree-2 outputs: two muls combined into a single save_output()

    // Spec steps 1-3
    let mut t0 = (&mut x1).mul(&mut x2);
    t0.save();
    let mut t1 = (&mut y1).mul(&mut y2);
    t1.save();
    let mut t2 = (&mut z1).mul(&mut z2);
    t2.save();

    // Spec steps 4-8: t3 = (X1+Y1)·(X2+Y2) - t0 - t1 = X1·Y2 + X2·Y1
    // [combined mul-sub]
    let mut t3_lhs = x1.clone() + y1.clone();
    let mut t3_rhs = x2.clone() + y2.clone();
    let mut t3 = (&mut t3_lhs).mul(&mut t3_rhs) - t0.clone() - t1.clone();
    t3.save();

    // Spec steps 9-13: t4 = (X1+Z1)·(X2+Z2) - t0 - t2 = X1·Z2 + X2·Z1
    // [combined mul-sub]
    let mut t4_lhs = x1.clone() + z1.clone();
    let mut t4_rhs = x2.clone() + z2.clone();
    let mut t4 = (&mut t4_lhs).mul(&mut t4_rhs) - t0.clone() - t2.clone();
    t4.save();

    // Spec steps 14-18: t5 = (Y1+Z1)·(Y2+Z2) - t1 - t2 = Y1·Z2 + Y2·Z1
    // [combined mul-sub]
    let mut t5_lhs = y1 + z1;
    let mut t5_rhs = y2 + z2;
    let mut t5 = (&mut t5_lhs).mul(&mut t5_rhs) - t1.clone() - t2.clone();
    t5.save();

    // Spec steps 19-21 give Z3 = 3b·t2 + a·t4. We absorb step 23 (Z3 = t1+Z3)
    // by adding t1 here: z3 = 3b·t2 + a·t4 + t1.
    let mut z3 = t2.clone() * b3_const.clone() + t4.clone() * a.clone() + t1.clone();
    z3.save();

    // Spec step 22: X3 = t1-Z3. Since our z3 = Z3_spec + t1:
    // x3 = 2·t1 - z3 = 2·t1 - (Z3_spec + t1) = t1 - Z3_spec = spec X3
    let mut x3 = t1.clone().int_mul(2) - z3.clone();
    x3.save();

    // Spec steps 25-29: t1 = 3·t0 + a·t2
    t1 = t0.clone().int_mul(3) + t2.clone() * a.clone();
    t1.save();

    // Spec step 30: t2 = t0 - a·t2_orig
    t2 = t0 - t2 * a.clone();
    t2.save();

    // Spec steps 28, 31-32: t4 = 3b·t4 + a·t2
    t4 = t4 * b3_const + t2 * a;
    t4.save();

    // Spec steps 33-40: each output combines 2 spec muls into a single degree-2
    // save_output(), saving 3 variables vs the spec's approach.

    // Spec steps 35-37: X3 = t3·x3 - t5·t4
    let t3_mul_x3 = (&mut t3).mul(&mut x3);
    let t5_mul_t4 = (&mut t5).mul(&mut t4);
    let mut x3_out = t3_mul_x3 - t5_mul_t4;
    x3_out.save_output();

    // Spec steps 24, 33-34: Y3 = x3·z3 + t1·t4
    let x3_mul_z3 = (&mut x3).mul(&mut z3);
    let t1_mul_t4 = (&mut t1).mul(&mut t4);
    let mut y3_out = x3_mul_z3 + t1_mul_t4;
    y3_out.save_output();

    // Spec steps 38-40: Z3 = t5·z3 + t3·t1
    let t5_mul_z3 = (&mut t5).mul(&mut z3);
    let t3_mul_t1 = (&mut t3).mul(&mut t1);
    let mut z3_out = t5_mul_z3 + t3_mul_t1;
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
