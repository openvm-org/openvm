use super::IsZeroChip;
// use crate::is_zero::air::IsZeroChip;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;

use afs_stark_backend::prover::USE_DEBUG_BUILDER;
use afs_stark_backend::verifier::VerificationError;
use afs_test_utils::config::baby_bear_poseidon2::run_simple_test;

#[test]
fn test_single_is_zero() {
    // Public inputs:
    let x = 97u32;
    let is_zero = 0u32;

    type Val = BabyBear;
    let pis = [x, is_zero].map(BabyBear::from_canonical_u32).to_vec();
    let chip = IsZeroChip::new(vec![x]); // Create an instance of IsZeroChip

    let trace = chip.generate_trace::<Val>(); // Use the instance

    run_simple_test(vec![&chip], vec![trace], vec![pis]).expect("Verification failed");
}

#[test]
fn test_single_is_zero2() {
    // Public inputs:
    let x = 0u32;
    let is_zero = 1u32;

    type Val = BabyBear;
    let pis = [x, is_zero].map(BabyBear::from_canonical_u32).to_vec();
    let chip = IsZeroChip::new(vec![x]); // Create an instance of IsZeroChip

    let trace = chip.generate_trace::<Val>(); // Use the instance

    run_simple_test(vec![&chip], vec![trace], vec![pis]).expect("Verification failed");
}

#[test]
fn test_single_is_zero_fail() {
    // Public inputs:
    let x = 187u32;
    let is_zero = 1u32;

    type Val = BabyBear;
    let pis = [x, is_zero].map(BabyBear::from_canonical_u32).to_vec();
    let chip = IsZeroChip::new(vec![x]); // Create an instance of IsZeroChip

    let trace = chip.generate_trace::<Val>(); // Use the instance

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test(vec![&chip], vec![trace], vec![pis]),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected constraint to fail"
    );
}

#[test]
fn test_single_is_zero_fail2() {
    // Public inputs:
    let x = 0u32;
    let is_zero = 0u32;

    type Val = BabyBear;
    let pis = [x, is_zero].map(BabyBear::from_canonical_u32).to_vec();
    let chip = IsZeroChip::new(vec![x]); // Create an instance of IsZeroChip

    let trace = chip.generate_trace::<Val>(); // Use the instance

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test(vec![&chip], vec![trace], vec![pis]),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected constraint to fail"
    );
}
