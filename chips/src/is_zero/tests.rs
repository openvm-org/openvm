use super::IsZeroChip;
use p3_baby_bear::BabyBear;

use afs_stark_backend::prover::USE_DEBUG_BUILDER;
use afs_stark_backend::verifier::VerificationError;
use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;
use p3_field::AbstractField;

#[test]
fn test_single_is_zero() {
    // Public inputs:
    let x = 97u32;

    type Val = BabyBear;
    let chip = IsZeroChip::new(vec![x]); // Create an instance of IsZeroChip

    let trace = chip.generate_trace::<Val>(); // Use the instance

    assert_eq!(trace.values[1], AbstractField::from_canonical_u32(0));

    run_simple_test_no_pis(vec![&chip], vec![trace]).expect("Verification failed");
}

#[test]
fn test_single_is_zero2() {
    // Public inputs:
    let x = 0u32;

    type Val = BabyBear;
    let chip = IsZeroChip::new(vec![x]); // Create an instance of IsZeroChip

    let trace = chip.generate_trace::<Val>(); // Use the instance

    assert_eq!(trace.values[1], AbstractField::from_canonical_u32(1));

    run_simple_test_no_pis(vec![&chip], vec![trace]).expect("Verification failed");
}

#[test]
fn test_single_is_zero_fail() {
    // Public inputs:
    let x = 187u32;

    type Val = BabyBear;

    let chip = IsZeroChip::new(vec![x]); // Create an instance of IsZeroChip

    let mut trace = chip.generate_trace::<Val>(); // Use the instance
    trace.values[1] = AbstractField::from_canonical_u32(1);

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
    let x = 0u32;

    type Val = BabyBear;

    let chip = IsZeroChip::new(vec![x]); // Create an instance of IsZeroChip

    let mut trace = chip.generate_trace::<Val>(); // Use the instance
    trace.values[1] = AbstractField::from_canonical_u32(0);

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(vec![&chip], vec![trace]),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected constraint to fail"
    );
}
