use std::{borrow::BorrowMut, sync::Arc};

use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use ax_sdk::{
    any_rap_vec, config::baby_bear_poseidon2::BabyBearPoseidon2Engine, engine::StarkFriEngine,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::DenseMatrix;

use super::{super::assert_less_than::AssertLessThanChip, columns::AssertLessThanCols};
use crate::var_range::{bus::VariableRangeCheckerBus, VariableRangeCheckerChip};

#[test]
fn test_borrow_mut_roundtrip() {
    const AUX_LEN: usize = 2; // number of auxilliary columns is two

    let num_cols = AssertLessThanCols::<usize, AUX_LEN>::width();
    let mut all_cols = (0..num_cols).collect::<Vec<usize>>();

    let lt_cols: &mut AssertLessThanCols<_, AUX_LEN> = all_cols[..].borrow_mut();

    lt_cols.io.x = 2;
    lt_cols.io.y = 8;
    lt_cols.aux.lower_decomp[0] = 1;
    lt_cols.aux.lower_decomp[1] = 0;

    assert_eq!(all_cols[0], 2);
    assert_eq!(all_cols[1], 8);
    assert_eq!(all_cols[2], 1);
    assert_eq!(all_cols[3], 0);
}

#[test]
fn test_assert_less_than_chip_lt() {
    let max_bits: usize = 16;
    let decomp: usize = 8;
    let bus = VariableRangeCheckerBus::new(0, decomp);
    const AUX_LEN: usize = 2;

    let range_checker = Arc::new(VariableRangeCheckerChip::new(bus));

    let chip = AssertLessThanChip::<AUX_LEN>::new(bus, max_bits, range_checker);
    let trace = chip.generate_trace(vec![(14321, 26883), (0, 1), (28, 120), (337, 456)]);
    let range_trace: DenseMatrix<BabyBear> = chip.range_checker.generate_trace();

    BabyBearPoseidon2Engine::run_simple_test_no_pis(
        &any_rap_vec![&chip.air, &chip.range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_lt_chip_decomp_does_not_divide() {
    let max_bits: usize = 29;
    let decomp: usize = 8;
    let bus = VariableRangeCheckerBus::new(0, decomp);
    const AUX_LEN: usize = 4;

    let range_checker = Arc::new(VariableRangeCheckerChip::new(bus));

    let chip = AssertLessThanChip::<AUX_LEN>::new(bus, max_bits, range_checker);
    let trace = chip.generate_trace(vec![(14321, 26883), (0, 1), (28, 120), (337, 456)]);
    let range_trace: DenseMatrix<BabyBear> = chip.range_checker.generate_trace();

    BabyBearPoseidon2Engine::run_simple_test_no_pis(
        &any_rap_vec![&chip.air, &chip.range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_assert_less_than_negative_1() {
    let max_bits: usize = 16;
    let decomp: usize = 8;
    let bus = VariableRangeCheckerBus::new(0, decomp);
    const AUX_LEN: usize = 2;

    let range_checker = Arc::new(VariableRangeCheckerChip::new(bus));

    let chip = AssertLessThanChip::<AUX_LEN>::new(bus, max_bits, range_checker);
    let mut trace = chip.generate_trace(vec![(28, 29)]);
    let range_trace = chip.range_checker.generate_trace();

    // Make the trace invalid
    trace.values.swap(0, 1);

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        BabyBearPoseidon2Engine::run_simple_test_no_pis(
            &any_rap_vec![&chip.air, &chip.range_checker.air],
            vec![trace, range_trace],
        )
        .err(),
        Some(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}

#[test]
fn test_assert_less_than_negative_2() {
    let max_bits: usize = 29;
    let decomp: usize = 8;
    let bus = VariableRangeCheckerBus::new(0, decomp);
    const AUX_LEN: usize = 4;
    let range_checker = Arc::new(VariableRangeCheckerChip::new(bus));

    let chip = AssertLessThanChip::<AUX_LEN>::new(bus, max_bits, range_checker);
    let mut trace = chip.generate_trace(vec![(28, 29)]);
    let range_trace = chip.range_checker.generate_trace();

    // Make the trace invalid
    trace.values[2] = AbstractField::from_canonical_u64(1 << decomp as u64);

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        BabyBearPoseidon2Engine::run_simple_test_no_pis(
            &any_rap_vec![&chip.air, &chip.range_checker.air],
            vec![trace, range_trace],
        )
        .err(),
        Some(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}
