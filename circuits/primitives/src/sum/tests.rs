use std::sync::Arc;

use afs_stark_backend::{utils::disable_debug_builder, verifier::VerificationError};
use ax_sdk::{
    any_rap_arc_vec, config::baby_bear_poseidon2::BabyBearPoseidon2Engine,
    dummy_airs::interaction::dummy_interaction_air::DummyInteractionAir, engine::StarkFriEngine,
    utils::to_field_vec,
};
use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};

use crate::{
    is_less_than::columns::IsLessThanCols,
    sub_chip::LocalTraceInstructions,
    sum::SumChip,
    var_range::{bus::VariableRangeCheckerBus, VariableRangeCheckerChip},
};

const INPUT_BUS: usize = 0;
const OUTPUT_BUS: usize = 1;
const RANGE_BUS: usize = 2;
const RANGE_MAX_BITS: usize = 4;

fn assert_verification_error(
    result: impl FnOnce() -> Result<(), VerificationError>,
    expected_error: VerificationError,
) {
    disable_debug_builder();
    assert_eq!(result(), Err(expected_error));
}

/// Tests whether a trace passes the interal constraints of sum air verification (i.e., not bus constraints).
fn run_sum_air_trace_test(sum_trace_u32: &[(u32, u32, u32, u32)]) -> Result<(), VerificationError> {
    let sender_air = DummyInteractionAir::new(2, true, INPUT_BUS);
    let sender_trace = RowMajorMatrix::new(
        to_field_vec(
            sum_trace_u32
                .iter()
                .flat_map(|&(key, val, _, _)| [1, key, val])
                .collect(),
        ),
        sender_air.field_width() + 1,
    );

    let receiver_air = DummyInteractionAir::new(2, false, OUTPUT_BUS);
    let receiver_trace = RowMajorMatrix::new(
        to_field_vec(
            sum_trace_u32
                .iter()
                .flat_map(|&(key, _, sum, is_final)| [is_final, key, sum])
                .collect(),
        ),
        receiver_air.field_width() + 1,
    );

    let range_checker = Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
        RANGE_BUS,
        RANGE_MAX_BITS,
    )));
    let sum_chip = SumChip::new(INPUT_BUS, OUTPUT_BUS, RANGE_MAX_BITS, range_checker);

    let mut rows: Vec<Vec<BabyBear>> = Vec::new();
    for i in 0..sum_trace_u32.len() {
        let partial_row = sum_trace_u32[i];
        let (key, val, sum, is_final) = partial_row;
        let next_key = sum_trace_u32[(i + 1) % sum_trace_u32.len()].0;

        let row = [key, val, sum, is_final];
        let mut row: Vec<BabyBear> = row.into_iter().map(BabyBear::from_canonical_u32).collect();

        let is_less_than_row: IsLessThanCols<BabyBear> = LocalTraceInstructions::generate_trace_row(
            &sum_chip.air.is_lt_air,
            (key, next_key, sum_chip.range_checker.clone()),
        );
        row.extend(is_less_than_row.aux.flatten());
        rows.push(row);
    }

    let width = BaseAir::<BabyBear>::width(&sum_chip.air);
    let sum_trace = DenseMatrix::new(rows.concat(), width);

    let range_checker_trace = sum_chip.range_checker.generate_trace();

    BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(
        any_rap_arc_vec![
            sum_chip.air,
            sum_chip.range_checker.air,
            sender_air,
            receiver_air
        ],
        vec![sum_trace, range_checker_trace, sender_trace, receiver_trace],
    )
    .map(|_| ())
}

#[test]
fn test_sum_air_trace_one_key() {
    let sum_trace = &[
        (0, 1, 1, 0),
        (0, 2, 3, 0),
        (0, 3, 6, 0),
        (0, 4, 10, 0),
        (0, 5, 15, 0),
        (0, 6, 21, 0),
        (0, 7, 28, 0),
        (0, 8, 36, 1),
    ];
    run_sum_air_trace_test(sum_trace).expect("Verification failed");
}

#[test]
fn test_sum_air_trace_many_keys() {
    let sum_trace = &[
        (0, 1, 1, 0),
        (0, 2, 3, 0),
        (0, 3, 6, 1),
        (3, 4, 4, 0),
        (3, 5, 9, 0),
        (3, 6, 15, 1),
        (8, 7, 7, 1),
        (10, 8, 8, 1),
    ];
    run_sum_air_trace_test(sum_trace).expect("Verification failed");
}

#[test]
fn test_sum_air_trace_one_key_wrong_sum() {
    let sum_trace = &[
        (0, 1, 1, 0),
        (0, 2, 3, 0),
        (0, 3, 6, 0),
        (0, 4, 14, 1), // wrong
    ];
    assert_verification_error(
        || run_sum_air_trace_test(sum_trace),
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn test_sum_air_trace_one_key_initial_sum_wrong() {
    let sum_trace = &[
        (0, 1, 0, 0), // wrong
        (0, 2, 2, 0),
        (0, 3, 5, 0),
        (0, 4, 9, 1),
    ];
    assert_verification_error(
        || run_sum_air_trace_test(sum_trace),
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn test_sum_air_trace_one_key_no_final() {
    let sum_trace = &[
        (0, 1, 1, 0),
        (0, 2, 3, 0),
        (0, 3, 6, 0),
        (0, 4, 10, 0),
        (0, 5, 15, 0),
        (0, 6, 21, 0),
        (0, 7, 28, 0),
        (0, 8, 36, 0), // wrong: is_final not set
    ];
    assert_verification_error(
        || run_sum_air_trace_test(sum_trace),
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn test_sum_air_trace_is_final_not_bool() {
    let sum_trace = &[(0, 1, 1, 0), (0, 1, 2, 2), (1, 2, 2, 0), (1, 3, 5, 1)];
    assert_verification_error(
        || run_sum_air_trace_test(sum_trace),
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn test_sum_air_trace_many_keys_wrong_sum() {
    let sum_trace = &[
        (0, 1, 1, 0),
        (0, 2, 3, 0),
        (0, 3, 6, 1),
        (3, 4, 4, 0),
        (3, 5, 9, 0),
        (3, 6, 9, 1), // wrong
        (8, 7, 7, 1),
        (10, 8, 8, 1),
    ];
    assert_verification_error(
        || run_sum_air_trace_test(sum_trace),
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn test_sum_air_trace_two_groups_same_key() {
    let sum_trace = &[
        (0, 1, 1, 1), // wrong
        (0, 1, 1, 1),
    ];
    assert_verification_error(
        || run_sum_air_trace_test(sum_trace),
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn test_sum_air_trace_keys_increasing() {
    let sum_trace = &[
        (10, 1, 1, 1), // wrong
        (5, 1, 1, 1),
    ];
    assert_verification_error(
        || run_sum_air_trace_test(sum_trace),
        VerificationError::OodEvaluationMismatch,
    )
}
