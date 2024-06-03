use super::IsZeroChip;

use afs_stark_backend::prover::USE_DEBUG_BUILDER;
use afs_stark_backend::verifier::VerificationError;
use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;
use p3_field::AbstractField;

#[test]
fn test_single_is_zero() {
    let x = AbstractField::from_canonical_u32(97);

    let chip = IsZeroChip {};

    let trace = chip.generate_trace(vec![x]);

    assert_eq!(trace.values[1], AbstractField::from_canonical_u32(0));

    run_simple_test_no_pis(vec![&chip], vec![trace]).expect("Verification failed");
}

#[test]
fn test_single_is_zero2() {
    let x_vec = [0, 1, 2, 7]
        .iter()
        .map(|x| AbstractField::from_canonical_u32(*x))
        .collect();

    let chip = IsZeroChip {};

    let trace = chip.generate_trace(x_vec);

    assert_eq!(trace.values[1], AbstractField::from_canonical_u32(1));
    assert_eq!(trace.values[4], AbstractField::from_canonical_u32(0));
    assert_eq!(trace.values[7], AbstractField::from_canonical_u32(0));
    assert_eq!(trace.values[10], AbstractField::from_canonical_u32(0));

    run_simple_test_no_pis(vec![&chip], vec![trace]).expect("Verification failed");
}

#[test]
fn test_single_is_zero_fail() {
    let x = AbstractField::from_canonical_u32(187);

    let chip = IsZeroChip {};

    let mut trace = chip.generate_trace(vec![x]);
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
    let x_vec = [1, 2, 7, 0]
        .iter()
        .map(|x| AbstractField::from_canonical_u32(*x))
        .collect();

    let chip = IsZeroChip {};

    let mut trace = chip.generate_trace(x_vec);
    trace.values[1] = AbstractField::from_canonical_u32(1);

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(vec![&chip], vec![trace.clone()]),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected constraint to fail"
    );

    trace.values[1] = AbstractField::from_canonical_u32(0);
    trace.values[10] = AbstractField::from_canonical_u32(0);

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(vec![&chip], vec![trace.clone()]),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected constraint to fail"
    );
}
