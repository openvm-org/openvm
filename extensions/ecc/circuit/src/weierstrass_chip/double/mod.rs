use std::{
    cell::RefCell,
    ops::{Deref, DerefMut},
    rc::Rc,
};

use num_bigint::BigUint;
use num_traits::Zero;
use openvm_algebra_circuit::fields::FieldType;
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

pub fn ec_double_proj_expr(
    config: ExprBuilderConfig, // The coordinate field.
    range_bus: VariableRangeCheckerBus,
    a: BigUint,
    b: BigUint,
) -> FieldExpr {
    FieldExpr::new(
        ec_double_proj_program(config, range_bus.range_max_bits, a, b),
        range_bus,
    )
}

pub fn ec_double_proj_program(
    config: ExprBuilderConfig,
    range_max_bits: usize,
    a: BigUint,
    b: BigUint,
) -> FieldExpressionProgram {
    let b3 = (&b * 3u32) % &config.modulus;
    let builder = if a.is_zero() {
        build_ec_double_proj_a0_expr(config, range_max_bits, b3)
    } else {
        build_ec_double_proj_general_expr(config, range_max_bits, a.clone(), b3)
    };
    FieldExpressionProgram::new_with_setup_values(builder, true, vec![a, b])
}

fn build_ec_double_proj_a0_expr(
    config: ExprBuilderConfig,
    range_max_bits: usize,
    b3: BigUint,
) -> ExprBuilder {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let mut x1 = ExprBuilder::new_input(builder.clone());
    let mut y1 = ExprBuilder::new_input(builder.clone());
    let mut z1 = ExprBuilder::new_input(builder.clone());

    let b3_const = ExprBuilder::new_const(builder.clone(), b3);

    // Algorithm 9 from ePrint 2015/1060 (complete doubling for a=0).
    //
    // Column count (secp256k1, 256-bit): 1359 total (22.3% reduction from naive 1748).
    //
    // Optimization strategy: each save() creates 1 intermediate variable, which adds
    // N extra AIR columns (variable limbs + quotient limbs + carry limbs). Fewer saves
    // = fewer columns = faster proving. Saves are called explicitly throughout to avoid
    // suboptimal automatic save_if_overflow calls. Techniques used:
    //   1. Inlined intermediates: values folded into output expressions without saving
    //   2. Algebraic rearrangement: rewriting expressions to avoid extra saves

    // Spec step 1
    let mut t0 = y1.square();
    t0.save();

    // Spec steps 2-4: z3 = 8·t0
    let mut z3 = t0.clone().int_mul(8);
    z3.save();

    // Spec step 5
    let mut t1 = (&mut y1).mul(&mut z1);
    t1.save();

    // Spec step 6. Must save before const mul to prevent limb overflow.
    let mut z1_sq = z1.square();
    z1_sq.save();

    // Spec step 7: t2 = 3b·Z1² (using b3 = 3b constant directly)
    let mut t2 = z1_sq * b3_const;
    t2.save();

    // Spec steps 11-13: t1 = t2+t2, t2 = t1+t2 (triples t2), t0 = t0-t2.
    t0 = t0 - t2.clone().int_mul(3);
    t0.save();

    // Spec steps 16-18: t1 = X1·Y1, X3 = t0·t1, X3 = X3+X3
    // X3 = 2·t0·(X1·Y1)
    let mut t1_xy = (&mut x1).mul(&mut y1);
    t1_xy.save();
    let mut x3_out = (&mut t0).mul(&mut t1_xy).int_mul(2);
    x3_out.save_output();

    // Spec steps 8-9, 14-15: Y3 = X3_step8 + t0·Y3_step9
    //   where X3_step8 = t2·Z3 and Y3_step9 = t0_orig + t2.
    //
    // Algebraic rearrangement: let t0_new = our t0 = t0_orig - 3·t2.
    // Then Y3_step9 = t0_orig + t2 = (t0_new + 3·t2) + t2 = t0_new + 4·t2, so:
    //   Y3 = t2·z3 + t0_new·(t0_new + 4·t2)
    //      = t0_new² + 4·t0_new·t2 + t2·z3
    //      = t0² + t2·(4·t0 + z3)
    // This avoids saving (t0 + 4·t2) as a separate variable.
    let mut z3_plus_4t0 = z3.clone() + t0.clone().int_mul(4);
    let mut y3_out = t0.square() + (&mut t2).mul(&mut z3_plus_4t0);
    y3_out.save_output();

    // Spec step 10
    let mut z3_out = (&mut t1).mul(&mut z3);
    z3_out.save_output();

    let builder = (*builder).borrow().clone();
    builder
}

fn build_ec_double_proj_general_expr(
    config: ExprBuilderConfig,
    range_max_bits: usize,
    a_val: BigUint,
    b3: BigUint,
) -> ExprBuilder {
    config.check_valid();

    let builder = ExprBuilder::new(config, range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let mut x1 = ExprBuilder::new_input(builder.clone());
    let mut y1 = ExprBuilder::new_input(builder.clone());
    let mut z1 = ExprBuilder::new_input(builder.clone());

    let a = ExprBuilder::new_const(builder.clone(), a_val.clone());
    let b3_const = ExprBuilder::new_const(builder.clone(), b3);

    // Algorithm 3 from ePrint 2015/1060 (complete doubling for general a).
    //
    // Column count (secp256r1, 256-bit): 2034 total (30.7% reduction from naive 2934).
    //
    // Optimization strategy: each save() creates 1 intermediate variable, which adds
    // N extra AIR columns (variable limbs + quotient limbs + carry limbs). Fewer saves
    // = fewer columns = faster proving. Saves are called explicitly throughout to avoid
    // suboptimal automatic save_if_overflow calls. Techniques used:
    //   1. Inlined intermediates: values folded into output expressions without saving
    //   2. Algebraic rearrangement: e.g. (a-b)·(a+b) → a²-b² to avoid saving factors
    //   3. Degree-2 outputs: two muls combined into a single save_output()

    // Spec steps 1-3
    let mut t0 = x1.square();
    t0.save();
    let mut t1 = y1.square();
    t1.save();
    let mut t2 = z1.square();
    t2.save();

    // Spec steps 4-5: t3 = 2·X1·Y1
    let mut t3 = (&mut x1).mul(&mut y1).int_mul(2);
    t3.save();

    // Spec steps 6-7: z3 = 2·X1·Z1
    let mut z3 = (&mut x1).mul(&mut z1).int_mul(2);
    z3.save();

    // Spec steps 8-10: y3 = a·z3 + 3b·t2
    let mut y3 = z3.clone() * a.clone() + t2.clone() * b3_const.clone();
    y3.save();

    // Spec steps 11, 14: x3 = t3·(t1 - y3)
    // (t1-y3) is NOT saved separately — inlined into the mul with t3.
    let mut x3_diff = t1.clone() - y3.clone();
    let mut x3 = (&mut t3).mul(&mut x3_diff);
    x3.save();

    // Spec steps 12-13: Y3 = (t1+y3)·(t1-y3)
    // Difference of squares: (t1+y3)·(t1-y3) = t1² - y3²
    // Avoids saving (t1+y3) and (t1-y3) as separate variables.
    y3 = t1.clone().square() - y3.square();
    y3.save();

    // Spec step 16
    t2 = t2 * a.clone();
    t2.save();

    // Spec steps 15, 17-19: t3 = a·(t0 - t2) + 3b·z3
    t3 = (t0.clone() - t2.clone()) * a + z3 * b3_const;
    t3.save();

    // Spec steps 25-26: t2_yz = 2·Y1·Z1
    let mut t2_yz = (&mut y1).mul(&mut z1).int_mul(2);
    t2_yz.save();

    // Spec steps 27-28: X3 = x3 - t2_yz·t3
    let mut x3_out = x3 - (&mut t2_yz).mul(&mut t3);
    x3_out.save_output();

    // Spec steps 20-24: Y3 = y3 + (3·t0 + t2)·t3
    // 3·t0 + t2 = 3·X1² + a·Z1²
    let mut t0_sum = t0.int_mul(3) + t2;
    let mut y3_out = y3 + (&mut t0_sum).mul(&mut t3);
    y3_out.save_output();

    // Spec steps 29-31: Z3 = 4·t2_yz·t1
    let mut z3_out = (&mut t2_yz).mul(&mut t1).int_mul(4);
    z3_out.save_output();

    let builder = (*builder).borrow().clone();
    builder
}

/// `BLOCKS` is the number of memory blocks needed to represent one input or output point.
// Note: PreflightExecutor is implemented manually in preflight.rs with fast native arithmetic
#[derive(Clone)]
pub struct EcDoubleExecutor<const BLOCKS: usize> {
    pub(crate) inner: FieldExpressionExecutor<Rv64VecHeapAdapterExecutor<1, BLOCKS, BLOCKS>>,
    pub(crate) cached_field_type: Option<FieldType>,
}

impl<const BLOCKS: usize> EcDoubleExecutor<BLOCKS> {
    pub fn new(
        inner: FieldExpressionExecutor<Rv64VecHeapAdapterExecutor<1, BLOCKS, BLOCKS>>,
    ) -> Self {
        // Fast-path dispatch requires (modulus, a, b) to match a standard curve, since the
        // native implementations hardcode the curve constants.
        let cached_field_type = match inner.program().setup_values() {
            [a, b] => super::curves::get_verified_field_type(inner.program().prime(), a, b),
            _ => None,
        };
        Self {
            inner,
            cached_field_type,
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

fn gen_base_program(
    config: ExprBuilderConfig,
    range_max_bits: usize,
    a: BigUint,
    b: BigUint,
) -> (FieldExpressionProgram, Vec<usize>) {
    let program = ec_double_proj_program(config, range_max_bits, a, b);
    let local_opcode_idx = vec![
        Rv64WeierstrassOpcode::SW_EC_DOUBLE_PROJ as usize,
        Rv64WeierstrassOpcode::SETUP_SW_EC_DOUBLE_PROJ as usize,
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
    a: BigUint,
    b: BigUint,
) -> WeierstrassAir<1, BLOCKS> {
    let (program, local_opcode_idx) =
        gen_base_program(config, range_checker_bus.range_max_bits, a, b);
    let expr = FieldExpr::new(program, range_checker_bus);
    WeierstrassAir::new(
        Rv64VecHeapAdapterAir::new(exec_bridge, mem_bridge, range_checker_bus, pointer_max_bits),
        FieldExpressionCoreAir::new(expr, offset, local_opcode_idx, vec![]),
    )
}

pub fn get_ec_double_executor<const BLOCKS: usize>(
    config: ExprBuilderConfig,
    range_max_bits: usize,
    pointer_max_bits: usize,
    offset: usize,
    a: BigUint,
    b: BigUint,
) -> EcDoubleExecutor<BLOCKS> {
    let (program, local_opcode_idx) = gen_base_program(config, range_max_bits, a, b);
    EcDoubleExecutor::new(FieldExpressionExecutor::new(
        Rv64VecHeapAdapterExecutor::new(pointer_max_bits),
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
    a: BigUint,
    b: BigUint,
) -> WeierstrassChip<F, 1, BLOCKS> {
    let range_bus = range_checker.bus();
    let (program, local_opcode_idx) = gen_base_program(config, range_bus.range_max_bits, a, b);
    let expr = FieldExpr::new(program, range_bus);
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
