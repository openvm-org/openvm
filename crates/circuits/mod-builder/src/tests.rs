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

use crate::{
    test_utils::*, utils::biguint_to_limbs_vec, ExprBuilder, FieldExpr, FieldExprCols,
    FieldExpressionCoreRecordMut, FieldVariable, SymbolicExpr,
};

const LIMB_BITS: usize = 8;
use std::sync::Arc;

use openvm_circuit_primitives::var_range::VariableRangeCheckerChip;

fn create_field_expr_with_setup(
    builder: ExprBuilder,
) -> (FieldExpr, Arc<VariableRangeCheckerChip>, usize) {
    let prime = secp256k1_coord_prime();
    let (range_checker, _) = setup(&prime);
    let expr = FieldExpr::new(builder, range_checker.bus(), false);
    let width = BaseAir::<BabyBear>::width(&expr);
    (expr, range_checker, width)
}

fn create_field_expr_with_flags_setup(
    builder: ExprBuilder,
) -> (FieldExpr, Arc<VariableRangeCheckerChip>, usize) {
    let prime = secp256k1_coord_prime();
    let (range_checker, _) = setup(&prime);
    let expr = FieldExpr::new(builder, range_checker.bus(), true);
    let width = BaseAir::<BabyBear>::width(&expr);
    (expr, range_checker, width)
}

fn generate_direct_trace(
    expr: &FieldExpr,
    range_checker: &Arc<VariableRangeCheckerChip>,
    inputs: Vec<BigUint>,
    flags: Vec<bool>,
    width: usize,
) -> Vec<BabyBear> {
    let mut row = BabyBear::zero_vec(width);
    expr.generate_subrow((range_checker, inputs, flags), &mut row);
    row
}

fn generate_recorded_trace(
    expr: &FieldExpr,
    range_checker: &Arc<VariableRangeCheckerChip>,
    inputs: &[BigUint],
    flags: Vec<bool>,
    width: usize,
) -> Vec<BabyBear> {
    let mut buffer = vec![0u8; 1024];
    let mut record = FieldExpressionCoreRecordMut::new_from_execution_data(
        &mut buffer,
        inputs,
        expr.canonical_num_limbs(),
    );
    let data: Vec<u8> = inputs
        .iter()
        .flat_map(|x| biguint_to_limbs_vec(x, expr.canonical_num_limbs()))
        .collect();
    record.fill_from_execution_data(0, &data);

    let reconstructed_inputs: Vec<BigUint> = record
        .input_limbs
        .chunks(expr.canonical_num_limbs())
        .map(BigUint::from_bytes_le)
        .collect();

    let mut row = BabyBear::zero_vec(width);
    expr.generate_subrow((range_checker, reconstructed_inputs, flags), &mut row);
    row
}

fn verify_stark_with_traces(
    expr: FieldExpr,
    range_checker: Arc<VariableRangeCheckerChip>,
    trace: Vec<BabyBear>,
    width: usize,
) {
    let trace_matrix = RowMajorMatrix::new(trace, width);
    let range_trace = range_checker.generate_trace();
    BabyBearBlake3Engine::run_simple_test_no_pis_fast(
        any_rap_arc_vec![expr, range_checker.air],
        vec![trace_matrix, range_trace],
    )
    .expect("Verification failed");
}

fn extract_and_verify_result(
    expr: &FieldExpr,
    trace: &[BabyBear],
    expected: &BigUint,
    var_index: usize,
) {
    let FieldExprCols { vars, .. } = expr.load_vars(trace);
    assert!(var_index < vars.len(), "Variable index out of bounds");
    let generated = evaluate_biguint(&vars[var_index], LIMB_BITS);
    assert_eq!(generated, *expected);
}

fn test_trace_equivalence(
    expr: &FieldExpr,
    range_checker: &Arc<VariableRangeCheckerChip>,
    inputs: Vec<BigUint>,
    flags: Vec<bool>,
    width: usize,
) {
    let direct_trace =
        generate_direct_trace(expr, range_checker, inputs.clone(), flags.clone(), width);
    let recorded_trace = generate_recorded_trace(expr, range_checker, &inputs, flags, width);
    assert_eq!(
        direct_trace, recorded_trace,
        "Direct and recorded traces must be identical for inputs: {:?}",
        inputs
    );
}

#[test]
fn test_add() {
    let prime = secp256k1_coord_prime();
    let (_, builder) = setup(&prime);

    let x1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let mut x3 = x1 + x2;
    x3.save();
    let builder = builder.borrow().clone();

    let (expr, range_checker, width) = create_field_expr_with_setup(builder);

    let x = generate_random_biguint(&prime);
    let y = generate_random_biguint(&prime);
    let expected = (&x + &y) % &prime;
    let inputs = vec![x, y];

    let trace = generate_direct_trace(&expr, &range_checker, inputs, vec![], width);
    extract_and_verify_result(&expr, &trace, &expected, 0);
    verify_stark_with_traces(expr, range_checker, trace, width);
}

#[test]
fn test_div() {
    let prime = secp256k1_coord_prime();
    let (_, builder) = setup(&prime);

    let x1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let _x3 = x1 / x2; // auto save on division.
    let builder = builder.borrow().clone();

    let (expr, range_checker, width) = create_field_expr_with_setup(builder);

    let x = generate_random_biguint(&prime);
    let y = generate_random_biguint(&prime);
    let y_inv = y.modinv(&prime).unwrap();
    let expected = (&x * &y_inv) % &prime;
    let inputs = vec![x, y];

    let trace = generate_direct_trace(&expr, &range_checker, inputs, vec![], width);
    extract_and_verify_result(&expr, &trace, &expected, 0);
    verify_stark_with_traces(expr, range_checker, trace, width);
}

#[test]
fn test_auto_carry_mul() {
    let prime = secp256k1_coord_prime();
    let (_, builder) = setup(&prime);

    let mut x1 = ExprBuilder::new_input(builder.clone());
    let mut x2 = ExprBuilder::new_input(builder.clone());
    let mut x3 = &mut x1 * &mut x2;
    // The multiplication below will overflow, so it triggers x3 to be saved first.
    let mut x4 = &mut x3 * &mut x1;
    assert_eq!(x3.expr, SymbolicExpr::Var(0));
    x4.save();
    assert_eq!(x4.expr, SymbolicExpr::Var(1));

    let builder = builder.borrow().clone();
    let (expr, range_checker, width) = create_field_expr_with_setup(builder);

    let x = generate_random_biguint(&prime);
    let y = generate_random_biguint(&prime);
    let expected = (&x * &x * &y) % &prime; // x4 = x3 * x1 = (x1 * x2) * x1
    let inputs = vec![x, y];

    let trace = generate_direct_trace(&expr, &range_checker, inputs, vec![], width);
    let FieldExprCols { vars, .. } = expr.load_vars(&trace);
    assert_eq!(vars.len(), 2);
    extract_and_verify_result(&expr, &trace, &expected, 1);
    verify_stark_with_traces(expr, range_checker, trace, width);
}

#[test]
fn test_auto_carry_intmul() {
    let prime = secp256k1_coord_prime();
    let (_, builder) = setup(&prime);
    let mut x1: FieldVariable = ExprBuilder::new_input(builder.clone());
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
    let (expr, range_checker, width) = create_field_expr_with_setup(builder);

    let x = generate_random_biguint(&prime);
    let y = generate_random_biguint(&prime);
    let expected = (&x * &x * BigUint::from(9u32)) % &prime;
    let inputs = vec![x, y];

    let trace = generate_direct_trace(&expr, &range_checker, inputs, vec![], width);
    let FieldExprCols { vars, .. } = expr.load_vars(&trace);
    assert_eq!(vars.len(), 2);
    extract_and_verify_result(&expr, &trace, &expected, 1);
    verify_stark_with_traces(expr, range_checker, trace, width);
}

#[test]
fn test_auto_carry_add() {
    let prime = secp256k1_coord_prime();
    let (_, builder) = setup(&prime);

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
    let (expr, range_checker, width) = create_field_expr_with_setup(builder);

    let x = generate_random_biguint(&prime);
    let y = generate_random_biguint(&prime);
    let expected = (&x * &x * BigUint::from(10u32)) % &prime;
    let inputs = vec![x, y];

    let trace = generate_direct_trace(&expr, &range_checker, inputs, vec![], width);
    let FieldExprCols { vars, .. } = expr.load_vars(&trace);
    assert_eq!(vars.len(), 2);
    extract_and_verify_result(&expr, &trace, &expected, x5_id);
    verify_stark_with_traces(expr, range_checker, trace, width);
}

#[test]
fn test_auto_carry_div() {
    let prime = secp256k1_coord_prime();
    let (_, builder) = setup(&prime);

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

    let (expr, range_checker, width) = create_field_expr_with_setup(builder);

    let x = generate_random_biguint(&prime);
    let y = generate_random_biguint(&prime);
    let inputs = vec![x, y];

    let trace = generate_direct_trace(&expr, &range_checker, inputs, vec![], width);
    let FieldExprCols { vars, .. } = expr.load_vars(&trace);
    assert_eq!(vars.len(), 2);
    verify_stark_with_traces(expr, range_checker, trace, width);
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
    let (_, builder) = setup(&prime);
    let builder = make_addsub_chip(builder);

    let (expr, range_checker, width) = create_field_expr_with_flags_setup(builder);

    let x = generate_random_biguint(&prime);
    let y = generate_random_biguint(&prime);
    let expected = (&x + &prime - &y) % &prime;
    let inputs = vec![x, y];
    let flags: Vec<bool> = vec![false, true];

    let trace = generate_direct_trace(&expr, &range_checker, inputs, flags, width);
    extract_and_verify_result(&expr, &trace, &expected, 0);
    verify_stark_with_traces(expr, range_checker, trace, width);
}

#[test]
fn test_select2() {
    let prime = secp256k1_coord_prime();
    let (_, builder) = setup(&prime);
    let builder = make_addsub_chip(builder);

    let (expr, range_checker, width) = create_field_expr_with_flags_setup(builder);

    let x = generate_random_biguint(&prime);
    let y = generate_random_biguint(&prime);
    let expected = (&x + &y) % &prime;
    let inputs = vec![x, y];
    let flags: Vec<bool> = vec![true, false];

    let trace = generate_direct_trace(&expr, &range_checker, inputs, flags, width);
    extract_and_verify_result(&expr, &trace, &expected, 0);
    verify_stark_with_traces(expr, range_checker, trace, width);
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
fn test_recorded_execution_records() {
    let prime = secp256k1_coord_prime();
    let (_, builder) = setup(&prime);

    let x1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let mut x3 = x1 + x2;
    x3.save();
    let builder = builder.borrow().clone();

    let (expr, range_checker, width) = create_field_expr_with_setup(builder);

    let x = generate_random_biguint(&prime);
    let y = generate_random_biguint(&prime);
    let expected = (&x + &y) % &prime;
    let inputs = vec![x.clone(), y.clone()];
    let flags: Vec<bool> = vec![];

    // Test record creation and reconstruction
    let mut buffer = vec![0u8; 1024];
    let mut record = FieldExpressionCoreRecordMut::new_from_execution_data(
        &mut buffer,
        &inputs,
        expr.canonical_num_limbs(),
    );
    let data: Vec<u8> = inputs
        .iter()
        .flat_map(|x| biguint_to_limbs_vec(x, expr.canonical_num_limbs()))
        .collect();
    record.fill_from_execution_data(0, &data);
    assert_eq!(*record.opcode, 0);

    // Verify input reconstruction preserves data
    let reconstructed_inputs: Vec<BigUint> = record
        .input_limbs
        .chunks(expr.canonical_num_limbs())
        .map(BigUint::from_bytes_le)
        .collect();
    assert_eq!(reconstructed_inputs.len(), inputs.len());
    for (original, reconstructed) in inputs.iter().zip(reconstructed_inputs.iter()) {
        assert_eq!(original, reconstructed);
    }

    // Test standard execution and verification using reconstructed inputs
    let trace = generate_direct_trace(&expr, &range_checker, reconstructed_inputs, flags, width);
    extract_and_verify_result(&expr, &trace, &expected, 0);
    verify_stark_with_traces(expr, range_checker, trace, width);
}

#[test]
fn test_trace_mathematical_equivalence() {
    let prime = secp256k1_coord_prime();
    let (_, builder) = setup(&prime);

    let x1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let x3 = &mut (x1.clone() * x2.clone()) + &mut (x1.clone().square());
    let mut x4 = x3.clone() / x2.clone(); // This will trigger auto-save
    x4.save();
    let builder = builder.borrow().clone();

    let (expr, range_checker, width) = create_field_expr_with_setup(builder);

    for _ in 0..10 {
        let x = generate_random_biguint(&prime);
        let y = generate_random_biguint(&prime);

        let expected = {
            let temp = (&x * &y + &x * &x) % &prime;
            let y_inv = y.modinv(&prime).unwrap();
            (temp * y_inv) % &prime
        };

        let inputs = vec![x.clone(), y.clone()];
        let flags: Vec<bool> = vec![];

        // Test direct/recorded equivalence
        test_trace_equivalence(&expr, &range_checker, inputs.clone(), flags.clone(), width);

        // Verify the actual computation is correct
        let direct_row = generate_direct_trace(&expr, &range_checker, inputs.clone(), flags, width);
        let FieldExprCols { vars, .. } = expr.load_vars(&direct_row);
        extract_and_verify_result(&expr, &direct_row, &expected, vars.len() - 1);
    }
}

#[test]
fn test_record_arena_allocation_patterns() {
    let prime = secp256k1_coord_prime();
    let (_, builder) = setup(&prime);

    let x1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let mut x3 = x1 + x2;
    x3.save();
    let builder = builder.borrow().clone();

    let (expr, _range_checker, _width) = create_field_expr_with_setup(builder);

    let inputs = vec![
        generate_random_biguint(&prime),
        generate_random_biguint(&prime),
    ];

    // Test record creation with various input sizes
    let mut buffer = vec![0u8; 1024];
    let mut record = FieldExpressionCoreRecordMut::new_from_execution_data(
        &mut buffer,
        &inputs,
        expr.canonical_num_limbs(),
    );
    let data: Vec<u8> = inputs
        .iter()
        .flat_map(|x| biguint_to_limbs_vec(x, expr.canonical_num_limbs()))
        .collect();
    record.fill_from_execution_data(0, &data);
    assert_eq!(*record.opcode, 0);

    // Test with maximum inputs
    let max_inputs = vec![BigUint::one(); 40]; // MAX_INPUT_LIMBS / 4
    let mut max_buffer = vec![0u8; 2048];
    let max_record =
        FieldExpressionCoreRecordMut::new_from_execution_data(&mut max_buffer, &max_inputs, 4);
    assert_eq!(*max_record.opcode, 0);

    // Test input reconstruction
    let reconstructed_inputs: Vec<BigUint> = record
        .input_limbs
        .chunks(expr.canonical_num_limbs())
        .map(BigUint::from_bytes_le)
        .collect();
    assert_eq!(reconstructed_inputs.len(), inputs.len());
    for (original, reconstructed) in inputs.iter().zip(reconstructed_inputs.iter()) {
        assert_eq!(original, reconstructed);
    }
}

#[test]
fn test_tracestep_tracefiller_roundtrip() {
    let prime = secp256k1_coord_prime();
    let (_, builder) = setup(&prime);

    let x1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let x3 = x1.clone() * x2.clone();
    let x4 = x3.clone() + x1.clone();
    let mut x5 = x4.clone();
    x5.save();
    let builder_data = builder.borrow().clone();

    let (expr, _range_checker, _width) = create_field_expr_with_setup(builder_data);

    let inputs = vec![
        generate_random_biguint(&prime),
        generate_random_biguint(&prime),
    ];

    let vars_direct = expr.execute(inputs.clone(), vec![]);

    // Test record creation and reconstruction roundtrip
    let mut buffer = vec![0u8; 1024];
    let mut record = FieldExpressionCoreRecordMut::new_from_execution_data(
        &mut buffer,
        &inputs,
        expr.canonical_num_limbs(),
    );
    let data: Vec<u8> = inputs
        .iter()
        .flat_map(|x| biguint_to_limbs_vec(x, expr.canonical_num_limbs()))
        .collect();
    record.fill_from_execution_data(0, &data);

    let reconstructed_inputs: Vec<BigUint> = record
        .input_limbs
        .chunks(expr.canonical_num_limbs())
        .map(BigUint::from_bytes_le)
        .collect();
    let vars_reconstructed = expr.execute(reconstructed_inputs, vec![]);

    // All intermediate variables must be preserved
    assert_eq!(vars_direct.len(), vars_reconstructed.len());
    for (direct, reconstructed) in vars_direct.iter().zip(vars_reconstructed.iter()) {
        assert_eq!(
            direct, reconstructed,
            "Variable preservation failed in roundtrip"
        );
    }
}

#[test]
fn test_direct_recorded_with_complex_operations() {
    let prime = secp256k1_coord_prime();
    let (_, builder) = setup(&prime);

    let x1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let x3 = ExprBuilder::new_input(builder.clone());

    let numerator = x1.clone() * x2.clone() + x3.clone();
    let denominator = x1.clone() + x2.clone();
    let mut result = numerator / denominator;
    result.save();

    let builder_data = builder.borrow().clone();
    let (expr, range_checker, width) = create_field_expr_with_setup(builder_data);

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
        let flags = vec![];

        // Test direct/recorded equivalence
        test_trace_equivalence(&expr, &range_checker, inputs.clone(), flags.clone(), width);

        // Verify mathematical correctness
        let expected = {
            let num = (&x * &y + &z) % &prime;
            let den_inv = (&x + &y).modinv(&prime).unwrap();
            (num * den_inv) % &prime
        };

        let direct_row = generate_direct_trace(&expr, &range_checker, inputs, flags, width);
        let FieldExprCols { vars, .. } = expr.load_vars(&direct_row);
        extract_and_verify_result(&expr, &direct_row, &expected, vars.len() - 1);
    }
}

#[test]
fn test_concurrent_direct_recorded_simulation() {
    // Simulate mixed direct/recorded execution to ensure RecordArena abstraction works correctly
    let prime = secp256k1_coord_prime();
    let (_, builder) = setup(&prime);

    let x1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let mut x3 = x1 + x2;
    x3.save();
    let builder_data = builder.borrow().clone();

    let (expr, range_checker, width) = create_field_expr_with_setup(builder_data);

    // Simulate multiple "concurrent" executions with different modes
    let execution_scenarios = vec![
        ("direct", true),
        ("recorded", false),
        ("direct", true),
        ("recorded", false),
    ];

    let mut all_traces = Vec::new();

    for (name, is_direct) in execution_scenarios {
        let inputs = vec![
            generate_random_biguint(&prime),
            generate_random_biguint(&prime),
        ];

        let trace = if is_direct {
            generate_direct_trace(&expr, &range_checker, inputs.clone(), vec![], width)
        } else {
            generate_recorded_trace(&expr, &range_checker, &inputs, vec![], width)
        };

        all_traces.push((name, inputs, trace));
    }

    // Verify each trace is mathematically valid
    for (_, inputs, trace) in &all_traces {
        let expected = (&inputs[0] + &inputs[1]) % &prime;
        extract_and_verify_result(&expr, trace, &expected, 0);
    }

    // Verify that direct and recorded with same inputs produce same results
    let same_inputs = vec![BigUint::from(123u32), BigUint::from(456u32)];
    test_trace_equivalence(&expr, &range_checker, same_inputs, vec![], width);
}
