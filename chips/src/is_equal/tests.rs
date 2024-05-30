use super::IsEqualChip;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;

use afs_stark_backend::prover::USE_DEBUG_BUILDER;
use afs_stark_backend::verifier::VerificationError;
use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;

#[test]
fn test_single_is_equal() {
    // Public inputs:
    let x = 97u32;
    let y = 97u32;

    type Val = BabyBear;
    let chip = IsEqualChip::new(vec![x], vec![y]); // Create an instance of IsEqualChip

    let trace = chip.generate_trace_rows::<Val>(); // Use the instance

    run_simple_test_no_pis(vec![&chip], vec![trace]).expect("Verification failed");
}

#[test]
fn test_single_is_equal2() {
    // Public inputs:
    let x = 127u32;
    let y = 74u32;

    type Val = BabyBear;
    let chip = IsEqualChip::new(vec![x], vec![y]); // Create an instance of IsEqualChip

    let trace = chip.generate_trace_rows::<Val>(); // Use the instance

    run_simple_test_no_pis(vec![&chip], vec![trace]).expect("Verification failed");
}

#[test]
fn test_single_is_zero_fail() {
    // Public inputs:
    let x = 187u32;
    let y = 123u32;

    type Val = BabyBear;
    let chip = IsEqualChip::new(vec![x], vec![y]); // Create an instance of IsEqualChip

    let mut trace = chip.generate_trace_rows::<Val>(); // Use the instance
    trace.values[2] = AbstractField::from_canonical_u32(1);

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
fn test_single_is_zero_fail2() {
    // Public inputs:
    let x = 123u32;
    let y = 123u32;

    type Val = BabyBear;
    let chip = IsEqualChip::new(vec![x], vec![y]); // Create an instance of IsEqualChip

    let mut trace = chip.generate_trace_rows::<Val>(); // Use the instance
    trace.values[2] = AbstractField::from_canonical_u32(0);

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(vec![&chip], vec![trace]),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected constraint to fail"
    );
}
