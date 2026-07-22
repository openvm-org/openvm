use std::{cell::RefCell, rc::Rc};

use num_bigint::BigUint;
use num_traits::One;
use openvm_circuit_primitives::{bigint::utils::*, TraceSubRowGenerator};
use openvm_stark_backend::{
    any_air_arc_vec, p3_air::BaseAir, p3_field::PrimeCharacteristicRing,
    p3_matrix::dense::RowMajorMatrix, prover::AirProvingContext, StarkEngine, SystemParams,
};
use openvm_stark_sdk::{config::baby_bear_poseidon2::*, p3_baby_bear::BabyBear};

use crate::{
    test_utils::*, utils::biguint_to_limbs_vec, ExprBuilder, ExprBuilderConfig, FieldExpr,
    FieldExprCols, FieldExpressionCoreRecordMut, FieldExpressionProgram, FieldVariable,
    SymbolicExpr,
};

const LIMB_BITS: usize = 8;
use std::sync::Arc;

use openvm_circuit_primitives::var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip};

#[test]
#[should_panic(expected = "expression and range bus must use the same range capacity")]
fn field_expr_rejects_mismatched_range_capacity() {
    let config = ExprBuilderConfig {
        modulus: secp256k1_coord_prime(),
        num_limbs: 32,
        limb_bits: 8,
    };
    let program = FieldExpressionProgram::new(ExprBuilder::new(config, 16), false);
    FieldExpr::new(program, VariableRangeCheckerBus::new(0, 15));
}

fn create_field_expr_with_setup(
    builder: ExprBuilder,
) -> (FieldExpr, Arc<VariableRangeCheckerChip>, usize) {
    let prime = secp256k1_coord_prime();
    let (range_checker, _) = setup(&prime);
    let program = FieldExpressionProgram::new(builder, false);
    let expr = FieldExpr::new(program, range_checker.bus());
    let width = BaseAir::<BabyBear>::width(&expr);
    (expr, range_checker, width)
}

fn create_field_expr_with_flags_setup(
    builder: ExprBuilder,
) -> (FieldExpr, Arc<VariableRangeCheckerChip>, usize) {
    let prime = secp256k1_coord_prime();
    let (range_checker, _) = setup(&prime);
    let program = FieldExpressionProgram::new(builder, true);
    let expr = FieldExpr::new(program, range_checker.bus());
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
        expr.program().canonical_num_limbs(),
    );
    let data: Vec<u8> = inputs
        .iter()
        .flat_map(|x| biguint_to_limbs_vec(x, expr.program().canonical_num_limbs()))
        .collect();
    record.fill_from_execution_data(0, &data);

    let reconstructed_inputs: Vec<BigUint> = record
        .input_limbs
        .chunks(expr.program().canonical_num_limbs())
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
    let engine: BabyBearPoseidon2CpuEngine =
        BabyBearPoseidon2CpuEngine::new(SystemParams::new_for_testing(20));
    let ctxs = vec![
        AirProvingContext::simple_no_pis(trace_matrix),
        AirProvingContext::simple_no_pis(range_trace),
    ];
    engine
        .run_test(any_air_arc_vec![expr, range_checker.air], ctxs)
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
        "Direct and recorded traces must be identical for inputs: {inputs:?}"
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
        expr.program().canonical_num_limbs(),
    );
    let data: Vec<u8> = inputs
        .iter()
        .flat_map(|x| biguint_to_limbs_vec(x, expr.program().canonical_num_limbs()))
        .collect();
    record.fill_from_execution_data(0, &data);
    assert_eq!(*record.opcode, 0);

    // Verify input reconstruction preserves data
    let reconstructed_inputs: Vec<BigUint> = record
        .input_limbs
        .chunks(expr.program().canonical_num_limbs())
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
        expr.program().canonical_num_limbs(),
    );
    let data: Vec<u8> = inputs
        .iter()
        .flat_map(|x| biguint_to_limbs_vec(x, expr.program().canonical_num_limbs()))
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
        .chunks(expr.program().canonical_num_limbs())
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

    let vars_direct = expr.program().execute(&inputs, &[]);

    // Test record creation and reconstruction roundtrip
    let mut buffer = vec![0u8; 1024];
    let mut record = FieldExpressionCoreRecordMut::new_from_execution_data(
        &mut buffer,
        &inputs,
        expr.program().canonical_num_limbs(),
    );
    let data: Vec<u8> = inputs
        .iter()
        .flat_map(|x| biguint_to_limbs_vec(x, expr.program().canonical_num_limbs()))
        .collect();
    record.fill_from_execution_data(0, &data);

    let reconstructed_inputs: Vec<BigUint> = record
        .input_limbs
        .chunks(expr.program().canonical_num_limbs())
        .map(BigUint::from_bytes_le)
        .collect();
    let vars_reconstructed = expr.program().execute(&reconstructed_inputs, &[]);

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

#[test]
#[ignore]
fn bench_tracegen_ec_add_ne_shape() {
    use std::time::Instant;
    let prime = secp256k1_coord_prime();
    let (_, builder) = setup(&prime);

    // Same expression as ec_add_ne_expr in extensions/ecc weierstrass chip
    let x1 = ExprBuilder::new_input(builder.clone());
    let y1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let y2 = ExprBuilder::new_input(builder.clone());
    let mut lambda = (y2 - y1.clone()) / (x2.clone() - x1.clone());
    let mut x3 = lambda.square() - x1.clone() - x2;
    x3.save_output();
    let mut y3 = lambda * (x1 - x3.clone()) - y1;
    y3.save_output();
    let builder = builder.borrow().clone();

    let (expr, range_checker, width) = create_field_expr_with_flags_setup(builder);
    println!(
        "width = {width}, num_vars = {}, num_limbs = {}",
        expr.program().num_vars(),
        expr.program().canonical_num_limbs()
    );

    const N: usize = 4096;
    let mut state = 0x12345678u64;
    let mut next_biguint = |p: &BigUint| -> BigUint {
        let bytes: Vec<u8> = (0..32)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 33) as u8
            })
            .collect();
        BigUint::from_bytes_le(&bytes) % p
    };
    let rows_inputs: Vec<Vec<BigUint>> = (0..N)
        .map(|_| (0..4).map(|_| next_biguint(&prime)).collect())
        .collect();
    let flags = vec![true];

    let t0 = Instant::now();
    for inp in &rows_inputs {
        std::hint::black_box(expr.program().execute(inp, &flags));
    }
    let exec_time = t0.elapsed();

    let mut trace = BabyBear::zero_vec(width * N);
    let t1 = Instant::now();
    for (i, inp) in rows_inputs.iter().enumerate() {
        expr.generate_subrow(
            (&range_checker, inp.clone(), flags.clone()),
            &mut trace[i * width..(i + 1) * width],
        );
    }
    let subrow_time = t1.elapsed();
    let per_row = (exec_time + subrow_time) / N as u32;
    println!(
        "execute (run_field_expression part): {:?}/row",
        exec_time / N as u32
    );
    println!("generate_subrow: {:?}/row", subrow_time / N as u32);
    println!(
        "total fill: {:?}/row => {:.0} rows/s single-thread, {:.1} ms for 2^17 rows at 32 threads",
        per_row,
        1.0 / per_row.as_secs_f64(),
        (1 << 17) as f64 * per_row.as_secs_f64() * 1000.0 / 32.0
    );
}
