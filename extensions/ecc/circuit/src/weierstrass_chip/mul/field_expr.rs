//! Field expression for one row of the `EC_MUL` ladder.
//!
//! A row applies [`EC_MUL_STEPS_PER_ROW`] steps of `R = 2R + sigma*P` in affine coordinates. The
//! accumulator is chained between them inside the expression, so only the row's first input and
//! last output cross a row boundary.
//!
//! Signs come from the one-hot flags rather than columns of their own: step `j` uses
//! `sum_f flag_f * sigma_j(f)`, which is degree 1. The AIR reads the same sums to recover the
//! scalar's bits. Negation is free because the expression works mod `p`, so `-Py` is `0 - Py`.
//!
//! Steps run as `(2R) + sigma*P` rather than `(R + sigma*P) + R`. Both cost the same, but the
//! second needs `R != +-sigma*P` for its first addition, which fails on the first step, where
//! `R = P`.
//!
//! The doubling's `y` is not saved. It feeds only the addition's lambda and the final `y`, and
//! inlining it keeps both constraints at degree 2. That is five saved variables per step instead of
//! six, which matters because saved variables dominate the row's width.

use std::{cell::RefCell, rc::Rc};

use num_bigint::BigUint;
use openvm_circuit_primitives::var_range::VariableRangeCheckerBus;
use openvm_mod_circuit_builder::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionProgram, FieldVariable,
};

use super::{EC_MUL_SIGN_PATTERNS, EC_MUL_STEPS_PER_ROW};

/// Input indices into the expression's `inputs`.
///
/// The ordering is load-bearing: `P` must precede the accumulator. On a setup row `FieldExpr` pins
/// the leading inputs to the prime and `a`, and the doubling denominator is `2*acc_y`. With the
/// accumulator first, a setup row would carry `acc_y = a`, which is zero on three of the four
/// supported curves. That row is unsatisfiable, and also uncomputable, since
/// `SymbolicExpr::compute` inverts the denominator.
///
/// Pinning `P` instead leaves the accumulator free on setup rows for [`setup_row_inputs`] to
/// choose. It also matches the other ECC chips, whose setup operand carries `(modulus, a)`.
pub const IN_PX: usize = 0;
pub const IN_PY: usize = 1;
pub const IN_ACC_X: usize = 2;
pub const IN_ACC_Y: usize = 3;

/// The accumulator a setup row carries.
///
/// The setup check does not pin it, so it only has to keep the row's denominators nonzero — a
/// condition fewer values satisfy than it suggests. A setup row's `P` is `(prime, a)`, which is
/// the identity sentinel `(0, 0)` whenever `a = 0`. Adding it repeatedly drives many starting
/// values to `(0, 0)` as well, and then `2*acc_y` is zero; `(1, 1)` degenerates this way after one
/// step.
///
/// `(2, 1)` works on all four supported curves. `ec_mul_setup_row_is_computable` in the rvr FFI
/// checks it.
pub const SETUP_ACC: (u64, u64) = (2, 1);

/// The sign step `step` takes under `pattern`, as `+1` or `-1`.
///
/// Step 0 reads the most significant bit, matching the order a row consumes digits.
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

/// Builds a program whose public setup coefficient and formula coefficient differ.
///
/// This is deliberately available only to unit tests. It preserves the program's coarse layout
/// while changing its arithmetic, so the CUDA fast-path eligibility check can prove that it keys
/// off the exact expression rather than only dimensions and output indices.
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

/// Inputs for a setup row: the modulus, the expression's setup values, then [`SETUP_ACC`].
///
/// Only the leading inputs are compared against the first two. The accumulator is fixed here
/// because execution and trace generation must build the row identically, or the memory argument
/// will not balance.
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
        // `2*Ry` is never zero. On a compute row the multiplier is odd and below the curve order,
        // and the group has prime order so there is no 2-torsion. On a setup row the accumulator is
        // `SETUP_ACC`, picked for the same reason.
        let mut lambda_d = (acc_x.square().int_mul(3) + a.clone()) / acc_y.int_mul(2);
        let mut dx = lambda_d.square() - acc_x.int_mul(2);
        dx.save();
        // Not saved: it appears only in the two expressions below, and inlining it keeps both at
        // degree 2.
        let dy = lambda_d.clone() * (acc_x.clone() - dx.clone()) - acc_y.clone();

        // ---- R' = D + sigma*P ---------------------------------------------------------
        // `Px - Dx` is never zero. That would need `2m = +-sigma`, which the parity argument in the
        // `mul` module rules out.
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

/// `sigma_step * Py`, written as `2*(b_step * Py) - Py` for the scalar bit `b_step`.
///
/// `Select` is the only handle `FieldExpr` gives on a flag, so `b_step * Py` is a sum of
/// `Select(flag, Py, 0)` over the patterns whose `step`-th sign is positive, which is half of them.
/// Going through the bit instead of one signed term per pattern halves the number of selects.
///
/// Summing does not raise degree, so this is degree 2: one flag times one input. Exactly one flag
/// is set on a compute row, giving `+-Py`. A setup row sets none, giving `-Py`.
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
