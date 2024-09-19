use std::sync::Arc;

use afs_stark_backend::{utils::disable_debug_builder, verifier::VerificationError};
use ax_sdk::{
    any_rap_vec, config::baby_bear_poseidon2::BabyBearPoseidon2Engine, engine::StarkFriEngine,
};
use p3_field::AbstractField;

use super::super::is_less_than_tuple::IsLessThanTupleChip;
use crate::{
    is_less_than_tuple::{columns::IsLessThanTupleCols, IsLessThanTupleAir},
    var_range::{bus::VariableRangeCheckerBus, VariableRangeCheckerChip},
};

fn get_range_bus() -> VariableRangeCheckerBus {
    let range_max_bits: usize = 8;
    VariableRangeCheckerBus::new(0, range_max_bits)
}
fn get_tester_range_chip() -> Arc<VariableRangeCheckerChip> {
    let bus = get_range_bus();
    Arc::new(VariableRangeCheckerChip::new(bus))
}

#[test]
fn test_flatten_fromslice_roundtrip() {
    let limb_bits = vec![16, 8, 20, 20];
    let bus = get_range_bus();

    let lt_air = IsLessThanTupleAir::new(bus, limb_bits.clone());

    let num_cols = IsLessThanTupleCols::<usize>::width(&lt_air);
    let all_cols = (0..num_cols).collect::<Vec<usize>>();

    let cols_numbered = IsLessThanTupleCols::<usize>::from_slice(&all_cols, &lt_air);
    let flattened = cols_numbered.flatten();

    for (i, col) in flattened.iter().enumerate() {
        assert_eq!(*col, all_cols[i]);
    }

    assert_eq!(num_cols, flattened.len());
}

#[test]
fn test_is_less_than_tuple_chip() {
    let limb_bits: Vec<usize> = vec![16, 8];

    let range_checker = get_tester_range_chip();
    let chip = IsLessThanTupleChip::new(limb_bits, range_checker);
    let range_checker = chip.range_checker.as_ref();

    let trace = chip.generate_trace(vec![
        (vec![14321, 123], vec![26678, 233]),
        (vec![26678, 244], vec![14321, 233]),
        (vec![14321, 244], vec![14321, 244]),
        (vec![26678, 233], vec![14321, 244]),
    ]);
    let range_checker_trace = range_checker.generate_trace();
    BabyBearPoseidon2Engine::run_simple_test_no_pis(
        &any_rap_vec![&chip.air, &range_checker.air],
        vec![trace, range_checker_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_is_less_than_tuple_chip_negative() {
    let limb_bits: Vec<usize> = vec![16, 8];

    let range_checker = get_tester_range_chip();
    let chip = IsLessThanTupleChip::new(limb_bits, range_checker);
    let range_checker = chip.range_checker.as_ref();
    let mut trace = chip.generate_trace(vec![(vec![14321, 123], vec![26678, 233])]);
    let range_checker_trace = range_checker.generate_trace();

    trace.values[2] = AbstractField::from_canonical_u64(0);

    disable_debug_builder();
    assert_eq!(
        BabyBearPoseidon2Engine::run_simple_test_no_pis(
            &any_rap_vec![&chip.air, &range_checker.air],
            vec![trace, range_checker_trace]
        )
        .err(),
        Some(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}
