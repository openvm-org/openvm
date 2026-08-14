//! Field expression for one row of the `EC_MUL` ladder.
//!
//! A row applies a fixed number of steps of `R = 2R + sigma*P` in affine coordinates,
//! chaining the accumulator between steps inside the expression. Signs come from the one-hot
//! flags: step `j` uses `sum_f flag_f * sigma_j(f)`, which is degree 1.
//!
//! Steps run as `(2R) + sigma*P`: the alternative `(R + sigma*P) + R` needs `R != +-sigma*P`,
//! which fails on the first step where `R = P`. The doubling's `y` is inlined rather than saved.

use std::{cell::RefCell, rc::Rc};

use num_bigint::BigUint;
use openvm_circuit_primitives::var_range::VariableRangeCheckerBus;
use openvm_mod_circuit_builder::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionProgram, FieldVariable,
};

use super::{EC_MUL_SIGN_PATTERNS, EC_MUL_STEPS_PER_ROW};

/// Input indices into the expression's `inputs`.
///
/// The ordering is load-bearing: `P` must precede the accumulator, so a setup row pins
/// `(prime, a)` to `P` and leaves the accumulator free for the setup inputs to keep the
/// denominators nonzero.
pub const IN_PX: usize = 0;
pub const IN_PY: usize = 1;
pub const IN_ACC_X: usize = 2;
pub const IN_ACC_Y: usize = 3;

/// The accumulator a setup row carries. It only has to keep the row's denominators nonzero;
/// `(2, 1)` works on all four supported curves, checked by `ec_mul_setup_row_is_computable`.
pub const SETUP_ACC: (u64, u64) = (2, 1);

/// The sign step `step` takes under `pattern`, as `+1` or `-1`. Step 0 reads the most significant
/// bit.
pub const fn sign_of(pattern: usize, step: usize) -> i64 {
    let bit = (pattern >> (EC_MUL_STEPS_PER_ROW - 1 - step)) & 1;
    2 * (bit as i64) - 1
}

/// Builds the ladder-step expression for one curve.
///
/// `a_biguint` is the curve's `a` coefficient, folded into the expression as a constant.
pub fn ec_mul_step_expr(
    config: ExprBuilderConfig,
    range_bus: VariableRangeCheckerBus,
    a_biguint: BigUint,
) -> FieldExpr {
    FieldExpr::new(
        ec_mul_step_program(config, range_bus.range_max_bits, a_biguint),
        range_bus,
    )
}

/// Builds a program whose public setup coefficient and formula coefficient differ, so the CUDA
/// fast-path eligibility check can prove it keys off the exact expression.
#[cfg(all(test, feature = "cuda"))]
pub(super) fn mutated_ec_mul_step_expr(
    config: ExprBuilderConfig,
    range_bus: VariableRangeCheckerBus,
    setup_a: BigUint,
    formula_a: BigUint,
) -> FieldExpr {
    FieldExpr::new(
        FieldExpressionProgram::new_with_setup_values(
            build_ec_mul_step_expr(config, range_bus.range_max_bits, &formula_a),
            true,
            vec![setup_a],
        ),
        range_bus,
    )
}

pub fn ec_mul_step_program(
    config: ExprBuilderConfig,
    range_max_bits: usize,
    a_biguint: BigUint,
) -> FieldExpressionProgram {
    FieldExpressionProgram::new_with_setup_values(
        build_ec_mul_step_expr(config, range_max_bits, &a_biguint),
        true,
        vec![a_biguint],
    )
}

/// Inputs for a setup row: the modulus, the expression's setup values, then the fixed setup
/// accumulator.
/// Execution and trace generation must build the row identically, or the memory argument will not
/// balance.
pub fn setup_row_inputs(program: &FieldExpressionProgram) -> Vec<BigUint> {
    let mut inputs = Vec::with_capacity(program.num_inputs());
    inputs.push(program.prime().clone());
    inputs.extend(program.setup_values().iter().cloned());
    inputs.push(BigUint::from(SETUP_ACC.0));
    inputs.push(BigUint::from(SETUP_ACC.1));
    inputs.resize(program.num_inputs(), BigUint::ZERO);
    inputs
}

fn build_ec_mul_step_expr(
    config: ExprBuilderConfig,
    range_max_bits: usize,
    a_biguint: &BigUint,
) -> ExprBuilder {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let px = ExprBuilder::new_input(builder.clone());
    let py = ExprBuilder::new_input(builder.clone());
    let mut acc_x = ExprBuilder::new_input(builder.clone());
    let mut acc_y = ExprBuilder::new_input(builder.clone());

    let flags: Vec<usize> = (0..EC_MUL_SIGN_PATTERNS)
        .map(|_| (*builder).borrow_mut().new_flag())
        .collect();

    let a = ExprBuilder::new_const(builder.clone(), a_biguint.clone());
    let zero = ExprBuilder::new_const(builder.clone(), BigUint::ZERO);

    for step in 0..EC_MUL_STEPS_PER_ROW {
        let signed_py = signed_point_y(&py, &zero, &flags, step);

        // ---- D = 2R -------------------------------------------------------------------
        // `2*Ry` is never zero: see the totality argument in the `mul` module.
        let mut lambda_d = (acc_x.square().int_mul(3) + a.clone()) / acc_y.int_mul(2);
        let mut dx = lambda_d.square() - acc_x.int_mul(2);
        dx.save();
        // Inlined to keep both uses at degree 2.
        let dy = lambda_d.clone() * (acc_x.clone() - dx.clone()) - acc_y.clone();

        // ---- R' = D + sigma*P ---------------------------------------------------------
        // `Px - Dx` is never zero: see the totality argument in the `mul` module.
        let mut lambda_a = (signed_py.clone() - dy.clone()) / (px.clone() - dx.clone());
        let mut next_x = lambda_a.square() - dx.clone() - px.clone();
        next_x.save();
        let mut next_y = lambda_a.clone() * (dx.clone() - next_x.clone()) - dy;
        next_y.save();

        acc_x = next_x;
        acc_y = next_y;
    }

    acc_x.save_output();
    acc_y.save_output();

    let builder = (*builder).borrow().clone();
    builder
}

/// `sigma_step * Py`, written as `2*(b_step * Py) - Py` for the scalar bit `b_step`. Exactly one
/// flag is set on a compute row, giving `+-Py`; a setup row sets none, giving `-Py`.
fn signed_point_y(
    py: &FieldVariable,
    zero: &FieldVariable,
    flags: &[usize],
    step: usize,
) -> FieldVariable {
    let mut bit_py = zero.clone();
    for (pattern, &flag) in flags.iter().enumerate() {
        if sign_of(pattern, step) > 0 {
            bit_py = bit_py + FieldVariable::select(flag, py, zero);
        }
    }
    bit_py.int_mul(2) - py.clone()
}
