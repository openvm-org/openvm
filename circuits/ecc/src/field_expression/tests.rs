use std::{cell::RefCell, rc::Rc, str::FromStr, sync::Arc};

use afs_primitives::{
    bigint::{check_carry_mod_to_zero::CheckCarryModToZeroSubAir, utils::*},
    var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip},
    SubAir, TraceSubRowGenerator,
};
use afs_stark_backend::interaction::InteractionBuilder;
use ax_sdk::{
    any_rap_arc_vec, config::baby_bear_blake3::BabyBearBlake3Engine, engine::StarkFriEngine,
    utils::create_seeded_rng,
};
use lazy_static::lazy_static;
use num_bigint_dig::BigUint;
use num_traits::FromPrimitive;
use p3_air::{Air, BaseAir};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rand::RngCore;

use super::{super::test_utils::*, ExprBuilder, FieldExpr, FieldVariableConfig, SymbolicExpr};
use crate::field_expression::{FieldExprCols, FieldVariable};

const LIMB_BITS: usize = 8;

pub fn generate_random_biguint(prime: &BigUint) -> BigUint {
    let mut rng = create_seeded_rng();
    let len = 32;
    let x = (0..len).map(|_| rng.next_u32()).collect();
    let x = BigUint::new(x);
    x % prime
}

impl<AB: InteractionBuilder> Air<AB> for FieldExpr {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        SubAir::eval(self, builder, &local);
    }
}

fn setup(
    prime: &BigUint,
) -> (
    CheckCarryModToZeroSubAir,
    Arc<VariableRangeCheckerChip>,
    Rc<RefCell<ExprBuilder>>,
) {
    let field_element_bits = 30;
    let range_bus = 1;
    let range_decomp = 17; // double needs 17, rests need 16.
    let range_checker = Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
        range_bus,
        range_decomp,
    )));
    let subair = CheckCarryModToZeroSubAir::new(
        prime.clone(),
        LIMB_BITS,
        range_bus,
        range_decomp,
        field_element_bits,
    );
    let builder = ExprBuilder::new(
        prime.clone(),
        LIMB_BITS,
        32,
        range_checker.range_max_bits(),
        BabyBear::bits() - 2,
    );
    (subair, range_checker, Rc::new(RefCell::new(builder)))
}

#[derive(Clone)]
pub struct TestConfig;
impl FieldVariableConfig for TestConfig {
    fn canonical_limb_bits() -> usize {
        LIMB_BITS
    }

    fn max_limb_bits() -> usize {
        29
    }

    fn num_limbs_per_field_element() -> usize {
        32
    }
}

#[test]
fn test_add() {
    let prime = secp256k1_coord_prime();
    let (subair, range_checker, builder) = setup(&prime);

    let x1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let mut x3 = x1 + x2;
    x3.save();
    let builder = builder.borrow().clone();

    let expr = FieldExpr {
        builder,
        check_carry_mod_to_zero: subair,
        range_bus: range_checker.bus(),
    };
    let width = BaseAir::<BabyBear>::width(&expr);

    let x = generate_random_biguint(&prime);
    let y = generate_random_biguint(&prime);
    let expected = (&x + &y) % prime;
    let inputs = vec![x, y];

    let mut row = vec![BabyBear::zero(); width];
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
    let (subair, range_checker, builder) = setup(&prime);

    let x1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let _x3 = x1 / x2; // auto save on division.
    let builder = builder.borrow().clone();

    let expr = FieldExpr {
        builder,
        check_carry_mod_to_zero: subair,
        range_bus: range_checker.bus(),
    };
    let width = BaseAir::<BabyBear>::width(&expr);

    let x = generate_random_biguint(&prime);
    let y = generate_random_biguint(&prime);
    let y_inv = big_uint_mod_inverse(&y, &prime);
    let expected = (&x * &y_inv) % prime;
    let inputs = vec![x, y];

    let mut row = vec![BabyBear::zero(); width];
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
    let (subair, range_checker, builder) = setup(&prime);

    let mut x1 = ExprBuilder::new_input(builder.clone());
    let mut x2 = ExprBuilder::new_input(builder.clone());
    let mut x3 = &mut x1 * &mut x2;
    // The multiplication below will overflow, so it triggers x3 to be saved first.
    let mut x4 = &mut x3 * &mut x1;
    assert_eq!(x3.expr, SymbolicExpr::Var(0));
    x4.save();
    assert_eq!(x4.expr, SymbolicExpr::Var(1));

    let builder = builder.borrow().clone();

    let expr = FieldExpr {
        builder,
        check_carry_mod_to_zero: subair,
        range_bus: range_checker.bus(),
    };
    let width = BaseAir::<BabyBear>::width(&expr);
    let x = generate_random_biguint(&prime);
    let y = generate_random_biguint(&prime);
    let expected = (&x * &x * &y) % prime; // x4 = x3 * x1 = (x1 * x2) * x1
    let inputs = vec![x, y];

    let mut row = vec![BabyBear::zero(); width];
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
    let (subair, range_checker, builder) = setup(&prime);
    let mut x1 = ExprBuilder::new_input(builder.clone());
    let mut x2 = ExprBuilder::new_input(builder.clone());
    let mut x3 = &mut x1 * &mut x2;
    // The int_mul below will overflow:
    // x3 should have max_overflow_bits = 8 + 8 + log2(32) = 21
    // The carry bits = "max_overflow_bits - limb_bits + 1" will exceed 17 if it exceeds 17 + 8 - 1 = 24.
    // So it triggers x3 to be saved first.
    let mut x4 = x3.int_mul(9);
    assert_eq!(x3.expr, SymbolicExpr::Var(0));
    x4.save();
    assert_eq!(x4.expr, SymbolicExpr::Var(1));

    let builder = builder.borrow().clone();

    let expr = FieldExpr {
        builder,
        check_carry_mod_to_zero: subair,
        range_bus: range_checker.bus(),
    };
    let width = BaseAir::<BabyBear>::width(&expr);
    let x = generate_random_biguint(&prime);
    let y = generate_random_biguint(&prime);
    let expected = (&x * &x * BigUint::from(9u32)) % prime;
    let inputs = vec![x, y];

    let mut row = vec![BabyBear::zero(); width];
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
    let (subair, range_checker, builder) = setup(&prime);

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

    let expr = FieldExpr {
        builder,
        check_carry_mod_to_zero: subair,
        range_bus: range_checker.bus(),
    };
    let width = BaseAir::<BabyBear>::width(&expr);

    let x = generate_random_biguint(&prime);
    let y = generate_random_biguint(&prime);
    let expected = (&x * &x * BigUint::from(10u32)) % prime;
    let inputs = vec![x, y];

    let mut row = vec![BabyBear::zero(); width];
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
fn test_ec_add() {
    let prime = secp256k1_coord_prime();
    let (subair, range_checker, builder) = setup(&prime);

    let x1 = ExprBuilder::new_input(builder.clone());
    let y1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let y2 = ExprBuilder::new_input(builder.clone());
    let dx = x2.clone() - x1.clone();
    let dy = y2.clone() - y1.clone();
    let lambda = dy / dx; // auto save on division.
    let mut x3 = lambda.clone() * lambda.clone() - x1.clone() - x2;
    x3.save();
    let mut y3 = lambda * (x1 - x3) - y1;
    y3.save();
    let builder = builder.borrow().clone();

    let expr = FieldExpr {
        builder,
        check_carry_mod_to_zero: subair,
        range_bus: range_checker.bus(),
    };
    let width = BaseAir::<BabyBear>::width(&expr);
    let (x1, y1) = SampleEcPoints[0].clone();
    let (x2, y2) = SampleEcPoints[1].clone();
    let (expected_x3, expected_y3) = SampleEcPoints[2].clone();
    let inputs = vec![x1, y1, x2, y2];

    let mut row = vec![BabyBear::zero(); width];
    expr.generate_subrow((&range_checker, inputs, vec![]), &mut row);
    let FieldExprCols { vars, .. } = expr.load_vars(&row);
    assert_eq!(vars.len(), 3); // lambda, x3, y3
    let generated_x3 = evaluate_biguint(&vars[1], LIMB_BITS);
    let generated_y3 = evaluate_biguint(&vars[2], LIMB_BITS);
    assert_eq!(generated_x3, expected_x3);
    assert_eq!(generated_y3, expected_y3);

    let trace = RowMajorMatrix::new(row, width);
    let range_trace = range_checker.generate_trace();

    BabyBearBlake3Engine::run_simple_test_no_pis_fast(
        any_rap_arc_vec![expr, range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_ec_double() {
    let prime = secp256k1_coord_prime();
    let (subair, range_checker, builder) = setup(&prime);

    let x1 = ExprBuilder::new_input(builder.clone());
    let mut y1 = ExprBuilder::new_input(builder.clone());
    let nom = (x1.clone() * x1.clone()).int_mul(3);
    let denom = y1.int_mul(2);
    let lambda = nom / denom;
    let mut x3 = lambda.clone() * lambda.clone() - x1.clone() - x1.clone();
    x3.save();
    let mut y3 = lambda * (x1 - x3) - y1;
    y3.save();
    let builder = builder.borrow().clone();

    let expr = FieldExpr {
        builder,
        check_carry_mod_to_zero: subair,
        range_bus: range_checker.bus(),
    };
    let width = BaseAir::<BabyBear>::width(&expr);

    let (x1, y1) = SampleEcPoints[1].clone();
    let (expected_x3, expected_y3) = SampleEcPoints[3].clone();
    let inputs = vec![x1, y1];

    let mut row = vec![BabyBear::zero(); width];
    expr.generate_subrow((&range_checker, inputs, vec![]), &mut row);
    let FieldExprCols { vars, .. } = expr.load_vars(&row);
    assert_eq!(vars.len(), 3); // lambda, x3, y3
    let generated_x3 = evaluate_biguint(&vars[1], LIMB_BITS);
    let generated_y3 = evaluate_biguint(&vars[2], LIMB_BITS);
    assert_eq!(generated_x3, expected_x3);
    assert_eq!(generated_y3, expected_y3);

    let trace = RowMajorMatrix::new(row, width);
    let range_trace = range_checker.generate_trace();

    BabyBearBlake3Engine::run_simple_test_no_pis_fast(
        any_rap_arc_vec![expr, range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_select() {
    let prime = secp256k1_coord_prime();
    let (subair, range_checker, builder) = setup(&prime);

    let x1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let x3 = x1.clone() + x2.clone();
    let x4 = x1 - x2;
    let flag = {
        let mut builder = builder.borrow_mut();
        builder.new_flag()
    };
    let mut x5 = FieldVariable::select(flag, &x3, &x4);
    x5.save();
    let builder = builder.borrow().clone();

    let expr = FieldExpr {
        builder,
        check_carry_mod_to_zero: subair,
        range_bus: range_checker.bus(),
    };
    let width = BaseAir::<BabyBear>::width(&expr);

    let x = generate_random_biguint(&prime);
    let y = generate_random_biguint(&prime);
    let expected = (&x + &prime - &y) % prime;
    let inputs = vec![x, y];
    let flags = vec![false];

    let mut row = vec![BabyBear::zero(); width];
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
    let (subair, range_checker, builder) = setup(&prime);
    let x1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let x3 = x1.clone() + x2.clone();
    let x4 = x1 - x2;
    let flag = {
        let mut builder = builder.borrow_mut();
        builder.new_flag()
    };
    let mut x5 = FieldVariable::select(flag, &x3, &x4);
    x5.save();
    let builder = builder.borrow().clone();

    let expr = FieldExpr {
        builder,
        check_carry_mod_to_zero: subair,
        range_bus: range_checker.bus(),
    };
    let width = BaseAir::<BabyBear>::width(&expr);

    let x = generate_random_biguint(&prime);
    let y = generate_random_biguint(&prime);
    let expected = (&x + &y) % prime;
    let inputs = vec![x, y];
    let flags = vec![true];

    let mut row = vec![BabyBear::zero(); width];
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
    let (q, carry) = expr.constraint_limbs(&prime, LIMB_BITS, 32);
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
    // x * y = pq, q can be up to p so can limbs as p.
    // x * y and p * q  both have 63 limbs.
    let expected_q = 32;
    let expected_carry = 63;
    test_symbolic_limbs(expr, expected_q, expected_carry);
}

lazy_static! {
    // Sample points got from https://asecuritysite.com/ecc/ecc_points2 and
    // https://learnmeabitcoin.com/technical/cryptography/elliptic-curve/#add
    pub static ref SampleEcPoints: Vec<(BigUint, BigUint)> = {
        let x1 = BigUint::from_u32(1).unwrap();
        let y1 = BigUint::from_str(
            "29896722852569046015560700294576055776214335159245303116488692907525646231534",
        )
        .unwrap();
        let x2 = BigUint::from_u32(2).unwrap();
        let y2 = BigUint::from_str(
            "69211104694897500952317515077652022726490027694212560352756646854116994689233",
        )
        .unwrap();

        // This is the sum of (x1, y1) and (x2, y2).
        let x3 = BigUint::from_str("109562500687829935604265064386702914290271628241900466384583316550888437213118").unwrap();
        let y3 = BigUint::from_str(
            "54782835737747434227939451500021052510566980337100013600092875738315717035444",
        )
        .unwrap();

        // This is the double of (x2, y2).
        let x4 = BigUint::from_str(
            "23158417847463239084714197001737581570653996933128112807891516801581766934331").unwrap();
        let y4 = BigUint::from_str(
            "25821202496262252602076867233819373685524812798827903993634621255495124276396",
        )
        .unwrap();

        // This is the sum of (x3, y3) and (x4, y4).
        let x5 = BigUint::from_str("88733411122275068320336854419305339160905807011607464784153110222112026831518").unwrap();
        let y5 = BigUint::from_str(
            "69295025707265750480609159026651746584753914962418372690287755773539799515030",
        )
        .unwrap();

        vec![(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)]
    };
}
