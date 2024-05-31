use super::IsZeroChip;

use afs_stark_backend::prover::USE_DEBUG_BUILDER;
use afs_stark_backend::verifier::VerificationError;
use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;
use p3_field::AbstractField;

#[test]
fn test_single_is_zero() {
    let x = 97u32;

    let chip = IsZeroChip::new(vec![AbstractField::from_canonical_u32(x)]);

    let trace = chip.generate_trace();

    assert_eq!(trace.values[1], AbstractField::from_canonical_u32(0));

    run_simple_test_no_pis(vec![&chip], vec![trace]).expect("Verification failed");
}

#[test]
fn test_single_is_zero2() {
    let x = 0u32;

    let chip = IsZeroChip::new(vec![AbstractField::from_canonical_u32(x)]);

    let trace = chip.generate_trace();

    assert_eq!(trace.values[1], AbstractField::from_canonical_u32(1));

    run_simple_test_no_pis(vec![&chip], vec![trace]).expect("Verification failed");
}

#[test]
fn test_single_is_zero_fail() {
    let x = 187u32;

    let chip = IsZeroChip::new(vec![AbstractField::from_canonical_u32(x)]);

    let mut trace = chip.generate_trace();
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
    let x = 0u32;

    let chip = IsZeroChip::new(vec![AbstractField::from_canonical_u32(x)]);

    let mut trace = chip.generate_trace();
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
