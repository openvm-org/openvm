use super::IsEqualChip;
use p3_field::AbstractField;

use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;

#[test]
fn test_single_is_equal() {
    let x = AbstractField::from_canonical_u32(97);
    let y = AbstractField::from_canonical_u32(97);

    let chip = IsEqualChip {};

    let trace = chip.generate_trace(vec![x], vec![y]);

    run_simple_test_no_pis(vec![&chip], vec![trace]).expect("Verification failed");
}

#[test]
fn test_single_is_equal2() {
    let x = AbstractField::from_canonical_u32(127);
    let y = AbstractField::from_canonical_u32(74);

    let chip = IsEqualChip {};

    let trace = chip.generate_trace(vec![x], vec![y]);

    run_simple_test_no_pis(vec![&chip], vec![trace]).expect("Verification failed");
}

#[test]
fn test_single_is_zero_fail() {
    let x = AbstractField::from_canonical_u32(187);
    let y = AbstractField::from_canonical_u32(123);

    let chip = IsEqualChip {};

    let mut trace = chip.generate_trace(vec![x], vec![y]);
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
    let x = AbstractField::from_canonical_u32(123);
    let y = AbstractField::from_canonical_u32(123);

    let chip = IsEqualChip {};

    let mut trace = chip.generate_trace(vec![x], vec![y]);
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
