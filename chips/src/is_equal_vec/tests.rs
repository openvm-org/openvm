use crate::is_equal_vec::IsEqualVecChip;
use p3_baby_bear::BabyBear;

use afs_stark_backend::prover::USE_DEBUG_BUILDER;
use afs_stark_backend::verifier::VerificationError;
use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;
use p3_field::AbstractField;

#[test]
fn test_single_is_equal_vec() {
    // Public inputs:
    let x = vec![1, 2, 3];
    let y = vec![1, 2, 3];

    type Val = BabyBear;
    // let air = IsEqualVecChip;
    let chip = IsEqualVecChip::new(0, vec![x], vec![y]); // Create an instance of IsEqualChip

    let trace = chip.generate_trace_rows::<Val>(); // Use the instance

    run_simple_test_no_pis(vec![&chip], vec![trace]).expect("Verification failed");
}

#[test]
fn test_single_is_equal_vec2() {
    // Public inputs:
    let x = vec![2, 2, 7];
    let y = vec![3, 5, 1];

    type Val = BabyBear;
    // let air = IsEqualVecChip;
    let chip = IsEqualVecChip::new(0, vec![x], vec![y]); // Create an instance of IsEqualChip

    let trace = chip.generate_trace_rows::<Val>(); // Use the instance

    run_simple_test_no_pis(vec![&chip], vec![trace]).expect("Verification failed");
}

#[test]
fn test_single_is_equal_vec3() {
    // Public inputs:
    let x = vec![17, 23, 4];
    let y = vec![17, 23, 4];

    type Val = BabyBear;
    // let air = IsEqualVecChip;
    let chip = IsEqualVecChip::new(0, vec![x], vec![y]); // Create an instance of IsEqualChip

    let trace = chip.generate_trace_rows::<Val>(); // Use the instance

    run_simple_test_no_pis(vec![&chip], vec![trace]).expect("Verification failed");
}

#[test]
fn test_single_is_equal_vec4() {
    // Public inputs:
    let x1 = vec![1, 2, 3];
    let y1 = vec![1, 2, 1];
    let x2 = vec![2, 2, 7];
    let y2 = vec![3, 5, 1];
    let x3 = vec![17, 23, 4];
    let y3 = vec![17, 23, 4];
    let x4 = vec![1, 2, 3];
    let y4 = vec![1, 2, 1];

    type Val = BabyBear;
    // let air = IsEqualVecChip;
    let chip = IsEqualVecChip::new(0, vec![x1, x2, x3, x4], vec![y1, y2, y3, y4]); // Create an instance of IsEqualChip

    let trace = chip.generate_trace_rows::<Val>(); // Use the instance

    run_simple_test_no_pis(vec![&chip], vec![trace]).expect("Verification failed");
}

#[test]
fn test_single_is_equal_vec_fail() {
    // Public inputs:
    let x = vec![1, 2, 3];
    let y = vec![1, 2, 1];

    type Val = BabyBear;
    let chip = IsEqualVecChip::new(0, vec![x], vec![y]); // Create an instance of IsEqualChip

    let mut trace = chip.generate_trace_rows::<Val>(); // Use the instance

    // let mut trace_values = trace.values;
    trace.values[0] = AbstractField::from_canonical_u32(2);
    // let trace = DenseMatrix::new(trace_values, 1, trace.len());

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(vec![&chip], vec![trace]),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected constraint to fail"
    );
}

#[test]
fn test_single_is_equal_vec_fail2() {
    // Public inputs:
    let x = vec![1, 2, 3];
    let y = vec![1, 2, 1];

    type Val = BabyBear;
    let chip = IsEqualVecChip::new(0, vec![x], vec![y]); // Create an instance of IsEqualChip

    let mut trace = chip.generate_trace_rows::<Val>(); // Use the instance

    // let mut trace_values = trace.values;
    trace.values[8] = AbstractField::from_canonical_u32(1);
    // let trace = DenseMatrix::new(trace_values, 1, trace.len());

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(vec![&chip], vec![trace]),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected constraint to fail"
    );
}
