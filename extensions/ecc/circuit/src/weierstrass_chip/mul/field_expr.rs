//! Field expression for one step of the `EC_MUL` ladder.
//!
//! A compute row constrains
//!
//! ```text
//!     R' = bit ? 2R + P : 2R
//! ```
//!
//! over affine coordinates, with the point at infinity carried as the `(0, 0)` sentinel. Both the
//! doubling denominator `2*Ry` and the addition denominator `Px - Dx` are zero for some inputs, so
//! the step is expressed as four mutually exclusive cases, each selecting its own denominator and
//! output.
//!
//! The cases are one-hot flags because [`FieldExpr`] derives `is_setup = is_valid - sum(flags)` and
//! asserts it boolean, so at most one flag may be set on a row. `is_inf` and the scalar bit
//! therefore cannot be separate flags; both are recovered from the encoding by the AIR at no
//! column cost.
//!
//! | flag | case | output |
//! |---|---|---|
//! | [`FLAG_DBL`] | `R != O`, `bit = 0` | `2R` |
//! | [`FLAG_DBL_ADD`] | `R != O`, `bit = 1` | `2R + P` |
//! | [`FLAG_INF_STAY`] | `R = O`, `bit = 0` | `(0, 0)` |
//! | [`FLAG_INF_TAKE`] | `R = O`, `bit = 1` | `P` |
//! | none set | setup | unconstrained |
//!
//! The addition branch requires `Dx != Px`, which follows from the scalar bound documented on the
//! `mul` module.

use std::{cell::RefCell, rc::Rc};

use num_bigint::BigUint;
use num_traits::{One, Zero};
use openvm_circuit_primitives::var_range::VariableRangeCheckerBus;
use openvm_mod_circuit_builder::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionProgram, FieldVariable,
};

/// `R ≠ ∞`, scalar bit `0`: output `2R`.
pub const FLAG_DBL: usize = 0;
/// `R ≠ ∞`, scalar bit `1`: output `2R + P`.
pub const FLAG_DBL_ADD: usize = 1;
/// `R = ∞`, scalar bit `0`: output stays `(0, 0)`.
pub const FLAG_INF_STAY: usize = 2;
/// `R = ∞`, scalar bit `1`: output `P`.
pub const FLAG_INF_TAKE: usize = 3;
/// Number of one-hot case flags.
pub const NUM_STEP_FLAGS: usize = 4;

/// Input indices into the expression's `inputs`.
pub const IN_RX: usize = 0;
pub const IN_RY: usize = 1;
pub const IN_PX: usize = 2;
pub const IN_PY: usize = 3;

/// `a_biguint` is the curve's `a` coefficient, folded in as a constant.
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

/// Inputs for a setup row: the modulus, the expression's setup values, then zero padding.
///
/// `FieldExpr`'s setup constraint compares the leading inputs against these, and the row's output
/// follows from them, so execution and trace generation must build them identically.
pub fn setup_row_inputs(program: &FieldExpressionProgram) -> Vec<BigUint> {
    let mut inputs = Vec::with_capacity(program.num_inputs());
    inputs.push(program.prime().clone());
    inputs.extend(program.setup_values().iter().cloned());
    inputs.resize(program.num_inputs(), BigUint::ZERO);
    inputs
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

fn build_ec_mul_step_expr(
    config: ExprBuilderConfig,
    range_max_bits: usize,
    a_biguint: &BigUint,
) -> ExprBuilder {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let mut rx = ExprBuilder::new_input(builder.clone());
    let mut ry = ExprBuilder::new_input(builder.clone());
    let px = ExprBuilder::new_input(builder.clone());
    let py = ExprBuilder::new_input(builder.clone());

    let f_dbl = (*builder).borrow_mut().new_flag();
    let f_dbl_add = (*builder).borrow_mut().new_flag();
    let f_inf_stay = (*builder).borrow_mut().new_flag();
    let _f_inf_take = (*builder).borrow_mut().new_flag();
    debug_assert_eq!(f_dbl, FLAG_DBL);
    debug_assert_eq!(f_dbl_add, FLAG_DBL_ADD);
    debug_assert_eq!(f_inf_stay, FLAG_INF_STAY);
    debug_assert_eq!(_f_inf_take, FLAG_INF_TAKE);

    let a = ExprBuilder::new_const(builder.clone(), a_biguint.clone());
    let one = ExprBuilder::new_const(builder.clone(), BigUint::one());
    let zero = ExprBuilder::new_const(builder.clone(), BigUint::zero());

    // ---- D = 2R ----------------------------------------------------------------
    // 2*Ry is a valid denominator only under FLAG_DBL or FLAG_DBL_ADD; every other case, setup
    // included, substitutes 1, since a zero denominator makes the row unsatisfiable. `select`
    // takes a flag index rather than an expression, so the disjunction is expressed by nesting.
    let two_ry = ry.int_mul(2);
    let denom_d = FieldVariable::select(
        f_dbl,
        &two_ry,
        &FieldVariable::select(f_dbl_add, &two_ry, &one),
    );
    let mut lambda_d = (rx.square().int_mul(3) + a) / denom_d;

    // `Select` requires both branches to have equal limb counts, and the output selection below
    // mixes these values with the 32-limb inputs, so they are saved to canonical width rather than
    // left as overflow expressions.
    let mut dx = lambda_d.square() - rx.int_mul(2);
    dx.save();
    let mut dy = lambda_d.clone() * (rx.clone() - dx.clone()) - ry.clone();
    dy.save();

    // ---- A = D + P -------------------------------------------------------------
    // Px − Dx is only a valid denominator under FLAG_DBL_ADD.
    let px_minus_dx = px.clone() - dx.clone();
    let denom_a = FieldVariable::select(f_dbl_add, &px_minus_dx, &one);
    let mut lambda_a = (py.clone() - dy.clone()) / denom_a;

    let mut ax = lambda_a.square() - dx.clone() - px.clone();
    ax.save();
    let mut ay = lambda_a * (dx.clone() - ax.clone()) - dy.clone();
    ay.save();

    // ---- output selection ------------------------------------------------------
    // Exactly one flag is set, so the nested selects form a 4-way one-hot mux. The fall-through
    // arm, reached when no flag is set, is the setup row, whose output is unconstrained.
    let mut out_x = FieldVariable::select(
        f_dbl,
        &dx,
        &FieldVariable::select(
            f_dbl_add,
            &ax,
            &FieldVariable::select(f_inf_stay, &zero, &px),
        ),
    );
    out_x.save_output();

    let mut out_y = FieldVariable::select(
        f_dbl,
        &dy,
        &FieldVariable::select(
            f_dbl_add,
            &ay,
            &FieldVariable::select(f_inf_stay, &zero, &py),
        ),
    );
    out_y.save_output();

    let builder = (*builder).borrow().clone();
    builder
}
