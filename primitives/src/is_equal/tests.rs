use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;
use p3_field::AbstractField;
use test_case::test_matrix;

use super::IsEqualAir;

// #[test]
#[test_matrix(
    [0,97,127],
    [0,23,97]
)]
fn test_single_is_equal(x: u32, y: u32) {
    let x = AbstractField::from_canonical_u32(x);
    let y = AbstractField::from_canonical_u32(y);

    let chip = IsEqualAir {};

    let trace = chip.generate_trace(vec![x], vec![y]);

    run_simple_test_no_pis(vec![&chip], vec![trace]).expect("Verification failed");
}

#[test_matrix(
    [0,97,127],
    [0,23,97]
)]
fn test_single_is_zero_fail(x: u32, y: u32) {
    let x = AbstractField::from_canonical_u32(x);
    let y = AbstractField::from_canonical_u32(y);

    let chip = IsEqualAir {};

    let mut trace = chip.generate_trace(vec![x], vec![y]);
    trace.values[2] = if trace.values[2] == AbstractField::one() {
        AbstractField::zero()
    } else {
        AbstractField::one()
    };

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(vec![&chip], vec![trace]),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected constraint to fail"
    );
}
