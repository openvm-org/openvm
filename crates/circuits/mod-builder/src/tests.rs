use std::{cell::RefCell, rc::Rc};

use num_bigint::BigUint;
use num_traits::{One, Zero};
use openvm_circuit_primitives::{bigint::utils::*, TraceSubRowGenerator};
use openvm_stark_backend::{
    p3_air::BaseAir, p3_field::FieldAlgebra, p3_matrix::dense::RowMajorMatrix,
};
use openvm_stark_sdk::{
    any_rap_arc_vec, config::baby_bear_blake3::BabyBearBlake3Engine, engine::StarkFriEngine,
    p3_baby_bear::BabyBear,
};

use crate::{
    test_utils::*, ExprBuilder, FieldExpr, FieldExprCols, FieldExpressionCoreRecord, FieldVariable,
    SymbolicExpr,
};

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

#[test]
fn test_e4_execution_records() {
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
    let expected = (&x + &y) % &prime;
    let inputs = vec![x.clone(), y.clone()];
    let flags = vec![];

    let record = FieldExpressionCoreRecord::from_execution_data(
        100,
        200,
        0,
        &inputs,
        flags.clone(),
        expr.canonical_limb_bits(),
        expr.canonical_num_limbs(),
    );

    assert_eq!(record.num_inputs as usize, inputs.len());
    assert_eq!(record.limbs_per_input as usize, expr.canonical_num_limbs());
    assert_eq!(record.flag_states[..flags.len()], flags[..]);

    // Verify E4 input reconstruction preserves data
    let reconstructed_inputs = record.reconstruct_inputs(expr.canonical_limb_bits());
    assert_eq!(reconstructed_inputs.len(), inputs.len());
    for (original, reconstructed) in inputs.iter().zip(reconstructed_inputs.iter()) {
        assert_eq!(original, reconstructed);
    }

    // Test standard E3 execution and verification using reconstructed inputs
    let mut row = BabyBear::zero_vec(width);
    expr.generate_subrow((&range_checker, reconstructed_inputs, flags), &mut row);
    let FieldExprCols { vars, .. } = expr.load_vars(&row);
    assert_eq!(vars.len(), 1);
    let generated = evaluate_biguint(&vars[0], LIMB_BITS);
    assert_eq!(generated, expected);

    // Run STARK verification like other tests
    let trace = RowMajorMatrix::new(row, width);
    let range_trace = range_checker.generate_trace();

    BabyBearBlake3Engine::run_simple_test_no_pis_fast(
        any_rap_arc_vec![expr, range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_e3_e4_mathematical_equivalence() {
    let prime = secp256k1_coord_prime();
    let (range_checker, builder) = setup(&prime);

    // Create a complex expression to thoroughly test equivalence
    let x1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let x3 = &mut (x1.clone() * x2.clone()) + &mut (x1.clone().square());
    let mut x4 = x3.clone() / x2.clone(); // This will trigger auto-save
    x4.save();
    let builder = builder.borrow().clone();

    let expr = FieldExpr::new(builder, range_checker.bus(), false);
    let width = BaseAir::<BabyBear>::width(&expr);

    for _ in 0..10 {
        let x = generate_random_biguint(&prime);
        let y = generate_random_biguint(&prime);

        let expected = {
            let temp = (&x * &y + &x * &x) % &prime;
            let y_inv = y.modinv(&prime).unwrap();
            (temp * y_inv) % &prime
        };

        let inputs = vec![x.clone(), y.clone()];
        let flags = vec![];

        // E3 Path: Direct trace generation
        let mut e3_row = BabyBear::zero_vec(width);
        expr.generate_subrow((&range_checker, inputs.clone(), flags.clone()), &mut e3_row);

        // E4 Path: Record + deferred trace generation
        let record = FieldExpressionCoreRecord::from_execution_data(
            100,
            200,
            0,
            &inputs,
            flags.clone(),
            expr.canonical_limb_bits(),
            expr.canonical_num_limbs(),
        );

        let mut e4_row = BabyBear::zero_vec(width);
        // Simulate TraceFiller::fill_trace_row logic
        let reconstructed_inputs = record.reconstruct_inputs(expr.canonical_limb_bits());
        let reconstructed_flags = record.reconstruct_flags();
        expr.generate_subrow(
            (&range_checker, reconstructed_inputs, reconstructed_flags),
            &mut e4_row,
        );

        // CRITICAL: E3 and E4 must produce bit-identical traces
        assert_eq!(
            e3_row, e4_row,
            "E3 and E4 traces must be mathematically equivalent for inputs: {:?}",
            inputs
        );

        // Verify the actual computation is correct
        let FieldExprCols { vars, .. } = expr.load_vars(&e3_row);
        let generated = evaluate_biguint(&vars[vars.len() - 1], LIMB_BITS);
        assert_eq!(
            generated, expected,
            "Mathematical result incorrect for inputs: {:?}",
            inputs
        );
    }
}

#[test]
fn test_record_arena_allocation_patterns() {
    let prime = secp256k1_coord_prime();
    let (range_checker, builder) = setup(&prime);

    let x1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let mut x3 = x1 + x2;
    x3.save();
    let builder = builder.borrow().clone();

    let inputs = vec![
        generate_random_biguint(&prime),
        generate_random_biguint(&prime),
    ];
    let flags = vec![];

    // Test record creation with various input sizes
    let record =
        FieldExpressionCoreRecord::from_execution_data(100, 200, 0, &inputs, flags.clone(), 8, 32);

    assert_eq!(record.num_inputs as usize, inputs.len());
    assert_eq!(record.limbs_per_input, 32);
    assert_eq!(record.num_flags as usize, flags.len());
    assert_eq!(record.opcode, 0);
    assert_eq!(record.from_pc, 100);
    assert_eq!(record.from_timestamp, 200);

    let max_inputs = vec![BigUint::one(); 32]; // MAX_INPUT_LIMBS / 4
    let max_record =
        FieldExpressionCoreRecord::from_execution_data(0, 0, 0, &max_inputs, vec![], 8, 4);
    assert_eq!(max_record.num_inputs, 32);

    let max_flags = vec![true; 16];
    let flag_record =
        FieldExpressionCoreRecord::from_execution_data(0, 0, 0, &inputs, max_flags, 8, 32);
    assert_eq!(flag_record.num_flags, 16);

    let reconstructed_inputs = record.reconstruct_inputs(8);
    assert_eq!(reconstructed_inputs.len(), inputs.len());
    for (original, reconstructed) in inputs.iter().zip(reconstructed_inputs.iter()) {
        assert_eq!(original, reconstructed);
    }

    let reconstructed_flags = record.reconstruct_flags();
    assert_eq!(reconstructed_flags, flags);
}

#[test]
fn test_tracestep_tracefiller_roundtrip() {
    let prime = secp256k1_coord_prime();
    let (range_checker, builder) = setup(&prime);

    let x1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let x3 = x1.clone() * x2.clone();
    let x4 = x3.clone() + x1.clone();
    let mut x5 = x4.clone();
    x5.save();
    let builder_data = builder.borrow().clone();

    let expr = FieldExpr::new(builder_data.clone(), range_checker.bus(), false);

    let inputs = vec![
        generate_random_biguint(&prime),
        generate_random_biguint(&prime),
    ];

    let vars_direct = expr.execute(inputs.clone(), vec![]);

    let record = FieldExpressionCoreRecord::from_execution_data(
        42,
        100,
        1,
        &inputs,
        vec![],
        expr.canonical_limb_bits(),
        expr.canonical_num_limbs(),
    );

    let reconstructed_inputs = record.reconstruct_inputs(expr.canonical_limb_bits());
    let vars_reconstructed = expr.execute(reconstructed_inputs, vec![]);

    // All intermediate variables must be preserved
    assert_eq!(vars_direct.len(), vars_reconstructed.len());
    for (direct, reconstructed) in vars_direct.iter().zip(vars_reconstructed.iter()) {
        assert_eq!(
            direct, reconstructed,
            "Variable preservation failed in roundtrip"
        );
    }

    // Test with flags
    if builder_data.num_flags > 0 {
        let flags = vec![true; builder_data.num_flags];
        let vars_with_flags = expr.execute(inputs.clone(), flags.clone());

        let record_with_flags = FieldExpressionCoreRecord::from_execution_data(
            42,
            100,
            1,
            &inputs,
            flags.clone(),
            expr.canonical_limb_bits(),
            expr.canonical_num_limbs(),
        );

        let reconstructed_flags = record_with_flags.reconstruct_flags();
        assert_eq!(flags, reconstructed_flags);

        let reconstructed_inputs_flags =
            record_with_flags.reconstruct_inputs(expr.canonical_limb_bits());
        let vars_reconstructed_flags =
            expr.execute(reconstructed_inputs_flags, reconstructed_flags);

        assert_eq!(
            vars_with_flags, vars_reconstructed_flags,
            "Flag preservation failed"
        );
    }
}

#[test]
fn test_e3_e4_with_complex_operations() {
    let prime = secp256k1_coord_prime();
    let (range_checker, builder) = setup(&prime);

    let x1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let x3 = ExprBuilder::new_input(builder.clone());

    let numerator = x1.clone() * x2.clone() + x3.clone();
    let denominator = x1.clone() + x2.clone();
    let mut result = numerator / denominator;
    result.save();

    let builder_data = builder.borrow().clone();
    let expr = FieldExpr::new(builder_data, range_checker.bus(), false);
    let width = BaseAir::<BabyBear>::width(&expr);

    // Test edge cases with small and large numbers
    let test_cases = vec![
        (
            BigUint::from(1u32),
            BigUint::from(2u32),
            BigUint::from(3u32),
        ),
        (
            BigUint::from(100u32),
            BigUint::from(200u32),
            BigUint::from(300u32),
        ),
        (
            generate_random_biguint(&prime),
            generate_random_biguint(&prime),
            generate_random_biguint(&prime),
        ),
    ];

    for (x, y, z) in test_cases {
        let inputs = vec![x.clone(), y.clone(), z.clone()];

        // E3: Direct execution
        let mut e3_row = BabyBear::zero_vec(width);
        expr.generate_subrow((&range_checker, inputs.clone(), vec![]), &mut e3_row);

        // E4: Record-based execution
        let record = FieldExpressionCoreRecord::from_execution_data(
            0,
            0,
            0,
            &inputs,
            vec![],
            expr.canonical_limb_bits(),
            expr.canonical_num_limbs(),
        );

        let mut e4_row = BabyBear::zero_vec(width);
        let reconstructed_inputs = record.reconstruct_inputs(expr.canonical_limb_bits());
        expr.generate_subrow((&range_checker, reconstructed_inputs, vec![]), &mut e4_row);

        assert_eq!(
            e3_row, e4_row,
            "Complex expression E3/E4 mismatch for inputs: {:?}",
            inputs
        );

        // Verify mathematical correctness
        let expected = {
            let num = (&x * &y + &z) % &prime;
            let den_inv = (&x + &y).modinv(&prime).unwrap();
            (num * den_inv) % &prime
        };

        let FieldExprCols { vars, .. } = expr.load_vars(&e3_row);
        let computed = evaluate_biguint(&vars[vars.len() - 1], LIMB_BITS);
        assert_eq!(computed, expected, "Mathematical verification failed");
    }
}

#[test]
fn test_error_handling_and_edge_cases() {
    let prime = secp256k1_coord_prime();
    let (_range_checker, builder) = setup(&prime);

    let x1 = ExprBuilder::new_input(builder.clone());
    let mut x2 = x1.clone();
    x2.save();
    let _builder_data = builder.borrow().clone();

    let simple_inputs = vec![BigUint::from(1u32), BigUint::from(2u32)];

    let basic_record =
        FieldExpressionCoreRecord::from_execution_data(0, 0, 0, &simple_inputs, vec![], 8, 4);
    assert_eq!(basic_record.num_inputs as usize, simple_inputs.len());
    assert_eq!(basic_record.limbs_per_input, 4);

    // Test maximum flags boundary
    let max_flags = 16;
    let large_flags = vec![true; max_flags];
    let small_input = vec![BigUint::from(123u32)];
    let flag_boundary_record =
        FieldExpressionCoreRecord::from_execution_data(0, 0, 0, &small_input, large_flags, 8, 4);
    assert_eq!(flag_boundary_record.num_flags as usize, max_flags);

    // Test zero inputs edge case
    let zero_input = vec![BigUint::zero()];
    let zero_record =
        FieldExpressionCoreRecord::from_execution_data(0, 0, 0, &zero_input, vec![], 8, 4);
    let reconstructed_zero = zero_record.reconstruct_inputs(8);
    assert_eq!(reconstructed_zero[0], BigUint::zero());

    // Test small limb values that fit in the representation
    let small_value = BigUint::from(255u32); // fits in 8 bits exactly
    let small_input = vec![small_value.clone()];
    let small_record =
        FieldExpressionCoreRecord::from_execution_data(0, 0, 0, &small_input, vec![], 8, 4);
    let reconstructed_small = small_record.reconstruct_inputs(8);
    assert_eq!(reconstructed_small[0], small_value);

    // Test reconstruction preserves data exactly
    let test_inputs = vec![
        BigUint::from(42u32),
        BigUint::from(123u32),
        BigUint::from(255u32),
    ];
    let record =
        FieldExpressionCoreRecord::from_execution_data(0, 0, 0, &test_inputs, vec![], 8, 4);
    let reconstructed = record.reconstruct_inputs(8);
    for (original, reconstructed) in test_inputs.iter().zip(reconstructed.iter()) {
        assert_eq!(original, reconstructed);
    }

    // Test PC and timestamp values
    let pc_values = [0u32, 100u32, 0xFFFFFFFFu32];
    let timestamp_values = [0u32, 1000u32, 0xFFFFFFFFu32];
    let simple_input = vec![BigUint::from(1u32)];

    for &pc in &pc_values {
        for &timestamp in &timestamp_values {
            let record = FieldExpressionCoreRecord::from_execution_data(
                pc,
                timestamp,
                0,
                &simple_input,
                vec![],
                8,
                4,
            );
            assert_eq!(record.from_pc, pc);
            assert_eq!(record.from_timestamp, timestamp);

            let reconstructed = record.reconstruct_inputs(8);
            assert_eq!(reconstructed[0], simple_input[0]);
        }
    }

    // Test opcode values
    for opcode in 0..=255u8 {
        let record = FieldExpressionCoreRecord::from_execution_data(
            0,
            0,
            opcode,
            &simple_input,
            vec![],
            8,
            4,
        );
        assert_eq!(record.opcode, opcode);
    }

    // Test flag state handling
    let flag_test_cases = [
        vec![],
        vec![true],
        vec![false],
        vec![true, false, true],
        vec![false, false, false, false, false],
    ];

    for flags in flag_test_cases {
        let record = FieldExpressionCoreRecord::from_execution_data(
            0,
            0,
            0,
            &simple_input,
            flags.clone(),
            8,
            4,
        );
        let reconstructed_flags = record.reconstruct_flags();
        assert_eq!(flags, reconstructed_flags);
    }
}

#[test]
fn test_concurrent_e3_e4_simulation() {
    // Simulate mixed E3/E4 execution to ensure RecordArena abstraction works correctly
    let prime = secp256k1_coord_prime();
    let (range_checker, builder) = setup(&prime);

    let x1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let mut x3 = x1 + x2;
    x3.save();
    let builder_data = builder.borrow().clone();

    let expr = FieldExpr::new(builder_data, range_checker.bus(), false);
    let width = BaseAir::<BabyBear>::width(&expr);

    // Simulate multiple "concurrent" executions with different modes
    let execution_scenarios = vec![
        ("e3_direct", true),
        ("e4_deferred", false),
        ("e3_direct", true),
        ("e4_deferred", false),
    ];

    let mut all_traces = Vec::new();

    for (name, is_e3) in execution_scenarios {
        let inputs = vec![
            generate_random_biguint(&prime),
            generate_random_biguint(&prime),
        ];

        let mut trace = BabyBear::zero_vec(width);

        if is_e3 {
            // E3: Direct trace generation
            expr.generate_subrow((&range_checker, inputs.clone(), vec![]), &mut trace);
        } else {
            // E4: Record-based trace generation
            let record = FieldExpressionCoreRecord::from_execution_data(
                0,
                0,
                0,
                &inputs,
                vec![],
                expr.canonical_limb_bits(),
                expr.canonical_num_limbs(),
            );
            let reconstructed = record.reconstruct_inputs(expr.canonical_limb_bits());
            expr.generate_subrow((&range_checker, reconstructed, vec![]), &mut trace);
        }

        all_traces.push((name, inputs, trace));
    }

    // Verify each trace is mathematically valid
    for (name, inputs, trace) in &all_traces {
        let FieldExprCols { vars, .. } = expr.load_vars(trace);
        let generated = evaluate_biguint(&vars[0], LIMB_BITS);
        let expected = (&inputs[0] + &inputs[1]) % &prime;
        assert_eq!(
            generated, expected,
            "Concurrent execution {} failed verification",
            name
        );
    }

    // Verify that E3 and E4 with same inputs produce same results
    let same_inputs = vec![BigUint::from(123u32), BigUint::from(456u32)];

    let mut e3_trace = BabyBear::zero_vec(width);
    expr.generate_subrow((&range_checker, same_inputs.clone(), vec![]), &mut e3_trace);

    let record = FieldExpressionCoreRecord::from_execution_data(
        0,
        0,
        0,
        &same_inputs,
        vec![],
        expr.canonical_limb_bits(),
        expr.canonical_num_limbs(),
    );
    let mut e4_trace = BabyBear::zero_vec(width);
    let reconstructed = record.reconstruct_inputs(expr.canonical_limb_bits());
    expr.generate_subrow((&range_checker, reconstructed, vec![]), &mut e4_trace);

    assert_eq!(
        e3_trace, e4_trace,
        "Concurrent E3/E4 with same inputs must produce identical traces"
    );
}
