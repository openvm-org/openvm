//! FieldExpr for EC_MUL step operation
//!
//! This module defines the algebraic constraints for one step of the double-and-add algorithm.
//! Each compute row evaluates this FieldExpr to prove the correctness of point operations.
//!
//! ## FieldExpr Column Layout
//!
//! The FieldExpr generates columns in this order:
//! ```text
//! [is_valid | inputs[0..4] | vars[0..n] | q_limbs | carry_limbs | flags[0..2]]
//!     ^           ^              ^                                    ^
//!     |           |              |                                    |
//!  Must be 0   Rx,Ry,Px,Py    Intermediate                      bit, is_setup
//!  for digest                 computations
//! ```
//!
//! ## Division by Zero Handling
//!
//! Point doubling has denominator `2*Ry`, which is 0 when:
//! - R is at infinity (Ry = 0 on first compute row)
//! - Ry happens to be 0 (should not occur for valid points on most curves)
//!
//! Point addition has denominator `Px - Dx`, which is 0 when:
//! - P = D (the doubled point equals the base point) - see POTENTIAL ISSUE #2 below
//! - P = -D (would require different formula, but shouldn't happen in normal EC_MUL)
//!
//! We use `is_setup_flag` (which is linked to `is_setup OR is_inf` in the AIR) to select
//! safe denominators (1) instead of the real denominators when division would fail.
//!
//! ## POTENTIAL ISSUE #1: Adding Point to Infinity
//!
//! When R = ∞ (is_inf = 1) and bit = 1, we want R_next = P.
//! Current formula computes:
//! - R_doubled = 2 * R (with R = (0,0), gives garbage even with safe denom)
//! - R_added = R_doubled + P (garbage + P = garbage)
//! - Output = bit ? R_added : R_doubled = garbage
//!
//! This may not correctly compute P when adding P to infinity.
//! The native execution (`ec_mul` in curves.rs) handles this correctly,
//! but the AIR constraints may not match, causing verification failure.
//!
//! **Potential Fix**: Add explicit infinity handling in the selection:
//! `R_next = is_inf && bit ? P : (bit ? R_added : R_doubled)`
//!
//! ## POTENTIAL ISSUE #2: Adding Two Equal Points (P = D)
//!
//! When the doubled point D has the same x-coordinate as the base point P (Px = Dx),
//! the addition formula has `Px - Dx = 0` in the denominator.
//!
//! This can happen when:
//! - The scalar multiplication reaches a point where 2*R = P
//! - For specific curve/scalar combinations
//!
//! Currently, safe denominators are only used when `is_setup OR is_inf`. If P = D
//! occurs during normal computation (is_inf = 0), the constraint will fail.
//!
//! **Potential Fix**: Add a `points_equal` flag that detects when `Px = Dx` and
//! uses the doubling formula instead of addition, or use safe denominators.
//! Note: If Px = Dx but Py ≠ Dy, then P = -D and the result should be infinity.

use std::{cell::RefCell, rc::Rc};

use num_bigint::BigUint;
use num_traits::One;
use openvm_circuit_primitives::var_range::VariableRangeCheckerBus;
use openvm_mod_circuit_builder::{ExprBuilder, ExprBuilderConfig, FieldExpr, FieldVariable};

/// Creates a FieldExpr that performs one EC_MUL step.
/// - Input: (Rx, Ry, Px, Py) where R is the current accumulator and P is the base point
/// - Flag 0: bit_flag (scalar bit, 0 or 1) - determines whether to add P or just double
/// - Flag 1: is_setup_flag (0 for EC_MUL, 1 for SETUP_EC_MUL) - uses safe denominators when 1
/// - Output: (R_next_x, R_next_y) = bit ? (2R + P) : 2R
///
/// This combines point doubling and conditional addition in one expression:
/// 1. Compute doubled = 2 * R
/// 2. Compute added = doubled + P
/// 3. Select based on bit: output = bit ? added : doubled
///
/// When is_setup_flag = 1, all denominators use safe values (1) to avoid division by zero.
/// This is used for SETUP_EC_MUL which validates curve parameters without doing real computation.
pub fn ec_mul_step_expr(
    config: ExprBuilderConfig,
    range_bus: VariableRangeCheckerBus,
    a_biguint: BigUint, // Curve parameter 'a' for y^2 = x^3 + ax + b
) -> FieldExpr {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    // Inputs: current accumulator (R) and base point (P)
    let mut rx = ExprBuilder::new_input(builder.clone());
    let mut ry = ExprBuilder::new_input(builder.clone());
    let px = ExprBuilder::new_input(builder.clone());
    let py = ExprBuilder::new_input(builder.clone());

    // Flag 0: scalar bit (determines whether to add P or just double)
    let bit_flag = (*builder).borrow_mut().new_flag();
    // Flag 1: setup mode (uses safe denominators when set)
    let is_setup_flag = (*builder).borrow_mut().new_flag();

    let a = ExprBuilder::new_const(builder.clone(), a_biguint.clone());
    let one = ExprBuilder::new_const(builder.clone(), BigUint::one());

    // === Step 1: Compute doubled = 2 * R ===
    // Point doubling formula:
    // lambda = (3*Rx^2 + a) / (2*Ry)
    // Dx = lambda^2 - 2*Rx
    // Dy = lambda*(Rx - Dx) - Ry

    // For doubling denominator:
    // - When is_setup_flag = 1: use safe denominator (1)
    // - When is_setup_flag = 0: use real denominator (2*Ry)
    let real_double_denom = ry.int_mul(2);
    let lambda_double_denom =
        FieldVariable::select(is_setup_flag, &one.clone(), &real_double_denom);

    let mut lambda_double = (rx.square().int_mul(3) + a) / lambda_double_denom;

    let dx = lambda_double.square() - rx.int_mul(2);
    let dy = lambda_double.clone() * (rx.clone() - dx.clone()) - ry.clone();

    // === Step 2: Compute added = doubled + P ===
    // Point addition formula:
    // lambda = (Py - Dy) / (Px - Dx)
    // Ax = lambda^2 - Dx - Px
    // Ay = lambda*(Dx - Ax) - Dy

    // For addition denominator:
    // - When is_setup_flag = 1: use safe denominator (1)
    // - When is_setup_flag = 0: use real denominator (Px - Dx)
    let real_add_denom = px.clone() - dx.clone();
    // Pad `one` to have the same expr_limbs as real_add_denom (63 limbs from dx's square)
    // by doing a trivial add/sub that doesn't change the value but increases limb count
    let one_padded = one + dx.clone() - dx.clone();
    let lambda_add_denom = FieldVariable::select(is_setup_flag, &one_padded, &real_add_denom);

    let mut lambda_add = (py.clone() - dy.clone()) / lambda_add_denom;

    let ax = lambda_add.square() - dx.clone() - px;
    let ay = lambda_add * (dx.clone() - ax.clone()) - dy.clone();

    // === Step 3: Select based on bit ===
    // If bit = 1: output = added = (Ax, Ay)
    // If bit = 0: output = doubled = (Dx, Dy)
    // (When is_setup_flag = 1, the output values don't matter as we're just validating)
    let mut out_x = FieldVariable::select(bit_flag, &ax, &dx);
    let mut out_y = FieldVariable::select(bit_flag, &ay, &dy);

    out_x.save_output();
    out_y.save_output();

    let builder = (*builder).borrow().clone();
    FieldExpr::new_with_setup_values(builder, range_bus, true, vec![a_biguint])
}
