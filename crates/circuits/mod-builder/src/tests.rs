use std::{cell::RefCell, rc::Rc};

use num_bigint::BigUint;
use num_traits::One;
use openvm_circuit_primitives::{bigint::utils::*, TraceSubRowGenerator};
use openvm_stark_backend::{
    p3_air::BaseAir, p3_field::FieldAlgebra, p3_matrix::dense::RowMajorMatrix,
};
use openvm_stark_sdk::{
    any_rap_arc_vec, config::baby_bear_blake3::BabyBearBlake3Engine, engine::StarkFriEngine,
    p3_baby_bear::BabyBear,
};

use crate::{test_utils::*, ExprBuilder, FieldExpr, FieldExprCols, FieldVariable, SymbolicExpr};

const LIMB_BITS: usize = 8;

#[test]
fn test_add() {
    let prime = secp256k1_coord_prime();
    let (range_checker, builder) = setup(&prime);

    let x1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let mut x3 = x1 + x2;
    x3.save();
    let builder = builder.borrow().clone();

    let expr = FieldExpr::new(builder, range_checker.bus(), false);
    let width = BaseAir::<BabyBear>::width(&expr);

    let x = generate_random_biguint(&prime);
    let y = generate_random_biguint(&prime);
    let expected = (&x + &y) % prime;
    let inputs = vec![x, y];

    let mut row = BabyBear::zero_vec(width);
    expr.generate_subrow((&range_checker, inputs, vec![]), &mut row);
    let FieldExprCols { vars, .. } = expr.load_vars(&row);
    assert_eq!(vars.len(), 1);
    let generated = evaluate_biguint(&vars[0], LIMB_BITS);
    assert_eq!(generated, expected);

    let trace = RowMajorMatrix::new(row, width);
    let range_trace = range_checker.generate_trace();

    BabyBearBlake3Engine::run_simple_test_no_pis_fast(
        any_rap_arc_vec![expr, range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_div() {
    let prime = secp256k1_coord_prime();
    let (range_checker, builder) = setup(&prime);

    let x1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let _x3 = x1 / x2; // auto save on division.
    let builder = builder.borrow().clone();
    let expr = FieldExpr::new(builder, range_checker.bus(), false);
    let width = BaseAir::<BabyBear>::width(&expr);

    let x = generate_random_biguint(&prime);
    let y = generate_random_biguint(&prime);
    let y_inv = y.modinv(&prime).unwrap();
    let expected = (&x * &y_inv) % prime;
    let inputs = vec![x, y];

    let mut row = BabyBear::zero_vec(width);
    expr.generate_subrow((&range_checker, inputs, vec![]), &mut row);
    let FieldExprCols { vars, .. } = expr.load_vars(&row);
    assert_eq!(vars.len(), 1);
    let generated = evaluate_biguint(&vars[0], LIMB_BITS);
    assert_eq!(generated, expected);

    let trace = RowMajorMatrix::new(row, width);
    let range_trace = range_checker.generate_trace();

    BabyBearBlake3Engine::run_simple_test_no_pis_fast(
        any_rap_arc_vec![expr, range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_auto_carry_mul() {
    let prime = secp256k1_coord_prime();
    let (range_checker, builder) = setup(&prime);

    let mut x1 = ExprBuilder::new_input(builder.clone());
    let mut x2 = ExprBuilder::new_input(builder.clone());
    let mut x3 = &mut x1 * &mut x2;
    // The multiplication below will overflow, so it triggers x3 to be saved first.
    let mut x4 = &mut x3 * &mut x1;
    assert_eq!(x3.expr, SymbolicExpr::Var(0));
    x4.save();
    assert_eq!(x4.expr, SymbolicExpr::Var(1));

    let builder = builder.borrow().clone();

    let expr = FieldExpr::new(builder, range_checker.bus(), false);
    let width = BaseAir::<BabyBear>::width(&expr);
    let x = generate_random_biguint(&prime);
    let y = generate_random_biguint(&prime);
    let expected = (&x * &x * &y) % prime; // x4 = x3 * x1 = (x1 * x2) * x1
    let inputs = vec![x, y];

    let mut row = BabyBear::zero_vec(width);
    expr.generate_subrow((&range_checker, inputs, vec![]), &mut row);
    let FieldExprCols { vars, .. } = expr.load_vars(&row);
    assert_eq!(vars.len(), 2);
    let generated = evaluate_biguint(&vars[1], LIMB_BITS);
    assert_eq!(generated, expected);

    let trace = RowMajorMatrix::new(row, width);
    let range_trace = range_checker.generate_trace();

    BabyBearBlake3Engine::run_simple_test_no_pis_fast(
        any_rap_arc_vec![expr, range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_auto_carry_intmul() {
    let prime = secp256k1_coord_prime();
    let (range_checker, builder) = setup(&prime);
    let mut x1 = ExprBuilder::new_input(builder.clone());
    let mut x2 = ExprBuilder::new_input(builder.clone());
    let mut x3 = &mut x1 * &mut x2;
    // The int_mul below will overflow:
    // x3 should have max_overflow_bits = 8 + 8 + log2(32) = 21
    // The carry bits = "max_overflow_bits - limb_bits + 1" will exceed 17 if it exceeds 17 + 8 - 1
    // = 24. So it triggers x3 to be saved first.
    let mut x4 = x3.int_mul(9);
    assert_eq!(x3.expr, SymbolicExpr::Var(0));
    x4.save();
    assert_eq!(x4.expr, SymbolicExpr::Var(1));

    let builder = builder.borrow().clone();

    let expr = FieldExpr::new(builder, range_checker.bus(), false);
    let width = BaseAir::<BabyBear>::width(&expr);
    let x = generate_random_biguint(&prime);
    let y = generate_random_biguint(&prime);
    let expected = (&x * &x * BigUint::from(9u32)) % prime;
    let inputs = vec![x, y];

    let mut row = BabyBear::zero_vec(width);
    expr.generate_subrow((&range_checker, inputs, vec![]), &mut row);
    let FieldExprCols { vars, .. } = expr.load_vars(&row);
    assert_eq!(vars.len(), 2);
    let generated = evaluate_biguint(&vars[1], LIMB_BITS);
    assert_eq!(generated, expected);

    let trace = RowMajorMatrix::new(row, width);
    let range_trace = range_checker.generate_trace();

    BabyBearBlake3Engine::run_simple_test_no_pis_fast(
        any_rap_arc_vec![expr, range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_auto_carry_add() {
    let prime = secp256k1_coord_prime();
    let (range_checker, builder) = setup(&prime);

    let mut x1 = ExprBuilder::new_input(builder.clone());
    let mut x2 = ExprBuilder::new_input(builder.clone());
    let mut x3 = &mut x1 * &mut x2;
    let x4 = x3.int_mul(5);
    // Should not overflow, so x3 is not saved.
    assert_eq!(
        x3.expr,
        SymbolicExpr::Mul(
            Box::new(SymbolicExpr::Input(0)),
            Box::new(SymbolicExpr::Input(1))
        )
    );

    // Should overflow as this is 10 * x1 * x2.
    let mut x5 = x4.clone() + x4.clone();
    // cannot verify x4 as above is cloned.
    let x5_id = x5.save();
    // But x5 is var(1) implies x4 was saved as var(0).
    assert_eq!(x5.expr, SymbolicExpr::Var(1));

    let builder = builder.borrow().clone();

    let expr = FieldExpr::new(builder, range_checker.bus(), false);
    let width = BaseAir::<BabyBear>::width(&expr);

    let x = generate_random_biguint(&prime);
    let y = generate_random_biguint(&prime);
    let expected = (&x * &x * BigUint::from(10u32)) % prime;
    let inputs = vec![x, y];

    let mut row = BabyBear::zero_vec(width);
    expr.generate_subrow((&range_checker, inputs, vec![]), &mut row);
    let FieldExprCols { vars, .. } = expr.load_vars(&row);
    assert_eq!(vars.len(), 2);
    let generated = evaluate_biguint(&vars[x5_id], LIMB_BITS);
    assert_eq!(generated, expected);

    let trace = RowMajorMatrix::new(row, width);
    let range_trace = range_checker.generate_trace();

    BabyBearBlake3Engine::run_simple_test_no_pis_fast(
        any_rap_arc_vec![expr, range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_auto_carry_div() {
    let prime = secp256k1_coord_prime();
    let (range_checker, builder) = setup(&prime);

    let mut x1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    // The choice of scalar (7) needs to be such that
    // 1. the denominator 7x^2 doesn't trigger autosave, >=8 doesn't work.
    // 2. But doing a division on it triggers autosave, because of division constraint, <= 6 doesn't
    //    work.
    let mut x3 = x1.square().int_mul(7) / x2;
    x3.save();

    let builder = builder.borrow().clone();
    assert_eq!(builder.num_variables, 2); // numerator autosaved, and the final division

    let expr = FieldExpr::new(builder, range_checker.bus(), false);
    let width = BaseAir::<BabyBear>::width(&expr);

    let x = generate_random_biguint(&prime);
    let y = generate_random_biguint(&prime);
    // let expected = (&x * &x * BigUint::from(10u32)) % prime;
    let inputs = vec![x, y];

    let mut row = BabyBear::zero_vec(width);
    expr.generate_subrow((&range_checker, inputs, vec![]), &mut row);
    let FieldExprCols { vars, .. } = expr.load_vars(&row);
    assert_eq!(vars.len(), 2);
    // let generated = evaluate_biguint(&vars[x5_id], LIMB_BITS);
    // assert_eq!(generated, expected);

    let trace = RowMajorMatrix::new(row, width);
    let range_trace = range_checker.generate_trace();

    BabyBearBlake3Engine::run_simple_test_no_pis_fast(
        any_rap_arc_vec![expr, range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

fn make_addsub_chip(builder: Rc<RefCell<ExprBuilder>>) -> ExprBuilder {
    let x1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let x3 = x1.clone() + x2.clone();
    let x4 = x1.clone() - x2.clone();
    let (is_add_flag, is_sub_flag) = {
        let mut builder = builder.borrow_mut();
        let is_add = builder.new_flag();
        let is_sub = builder.new_flag();
        (is_add, is_sub)
    };
    let x5 = FieldVariable::select(is_sub_flag, &x4, &x1);
    let mut x6 = FieldVariable::select(is_add_flag, &x3, &x5);
    x6.save();
    let builder = builder.borrow().clone();
    builder
}

#[test]
fn test_select() {
    let prime = secp256k1_coord_prime();
    let (range_checker, builder) = setup(&prime);
    let builder = make_addsub_chip(builder);

    let expr = FieldExpr::new(builder, range_checker.bus(), true);
    let width = BaseAir::<BabyBear>::width(&expr);

    let x = generate_random_biguint(&prime);
    let y = generate_random_biguint(&prime);
    let expected = (&x + &prime - &y) % prime;
    let inputs = vec![x, y];
    let flags = vec![false, true];

    let mut row = BabyBear::zero_vec(width);
    expr.generate_subrow((&range_checker, inputs, flags), &mut row);
    let FieldExprCols { vars, .. } = expr.load_vars(&row);
    assert_eq!(vars.len(), 1);
    let generated = evaluate_biguint(&vars[0], LIMB_BITS);
    assert_eq!(generated, expected);

    let trace = RowMajorMatrix::new(row, width);
    let range_trace = range_checker.generate_trace();

    BabyBearBlake3Engine::run_simple_test_no_pis_fast(
        any_rap_arc_vec![expr, range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_select2() {
    let prime = secp256k1_coord_prime();
    let (range_checker, builder) = setup(&prime);
    let builder = make_addsub_chip(builder);

    let expr = FieldExpr::new(builder, range_checker.bus(), true);
    let width = BaseAir::<BabyBear>::width(&expr);

    let x = generate_random_biguint(&prime);
    let y = generate_random_biguint(&prime);
    let expected = (&x + &y) % prime;
    let inputs = vec![x, y];
    let flags = vec![true, false];

    let mut row = BabyBear::zero_vec(width);
    expr.generate_subrow((&range_checker, inputs, flags), &mut row);
    let FieldExprCols { vars, .. } = expr.load_vars(&row);
    assert_eq!(vars.len(), 1);
    let generated = evaluate_biguint(&vars[0], LIMB_BITS);
    assert_eq!(generated, expected);

    let trace = RowMajorMatrix::new(row, width);
    let range_trace = range_checker.generate_trace();

    BabyBearBlake3Engine::run_simple_test_no_pis_fast(
        any_rap_arc_vec![expr, range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

fn test_symbolic_limbs(expr: SymbolicExpr, expected_q: usize, expected_carry: usize) {
    let prime = secp256k1_coord_prime();
    let (q, carry) = expr.constraint_limbs(
        &prime,
        LIMB_BITS,
        32,
        &((BigUint::one() << 256) - BigUint::one()),
    );
    assert_eq!(q, expected_q);
    assert_eq!(carry, expected_carry);
}

#[test]
fn test_symbolic_limbs_add() {
    let expr = SymbolicExpr::Add(
        Box::new(SymbolicExpr::Var(0)),
        Box::new(SymbolicExpr::Var(1)),
    );
    // x + y = pq, q should fit in q limb.
    // x+y should have 32 limbs, pq also 32 limbs.
    let expected_q = 1;
    let expected_carry = 32;
    test_symbolic_limbs(expr, expected_q, expected_carry);
}

#[test]
fn test_symbolic_limbs_sub() {
    let expr = SymbolicExpr::Sub(
        Box::new(SymbolicExpr::Var(0)),
        Box::new(SymbolicExpr::Var(1)),
    );
    // x - y = pq, q should fit in q limb.
    // x - y should have 32 limbs, pq also 32 limbs.
    let expected_q = 1;
    let expected_carry = 32;
    test_symbolic_limbs(expr, expected_q, expected_carry);
}

#[test]
fn test_symbolic_limbs_mul() {
    let expr = SymbolicExpr::Mul(
        Box::new(SymbolicExpr::Var(0)),
        Box::new(SymbolicExpr::Var(1)),
    );
    // x * y = pq, and x,y can be up to 2^256 - 1 so q can be up to ceil((2^256 - 1)^2 / p) which
    // has 257 bits, which is 33 limbs x * y has 63 limbs, but p * q can have 64 limbs since q
    // is 33 limbs
    let expected_q = 33;
    let expected_carry = 64;
    test_symbolic_limbs(expr, expected_q, expected_carry);
}
