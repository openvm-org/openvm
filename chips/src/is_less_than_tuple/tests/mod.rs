use std::sync::Arc;

use crate::is_less_than_tuple::columns::IsLessThanTupleCols;
use crate::range_gate::RangeCheckerGateChip;

use super::super::is_less_than_tuple::IsLessThanTupleChip;

use afs_stark_backend::prover::USE_DEBUG_BUILDER;
use afs_stark_backend::verifier::VerificationError;
use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;
use p3_field::AbstractField;

#[test]
fn test_flatten_fromslice_roundtrip() {
    let limb_bits = vec![16, 8, 20, 20];
    let decomp = 8;

    let num_cols = IsLessThanTupleCols::<usize>::get_width(&limb_bits, decomp);
    let all_cols = (0..num_cols).collect::<Vec<usize>>();

    let cols_numbered = IsLessThanTupleCols::<usize>::from_slice(&all_cols, &limb_bits, decomp);
    let flattened = cols_numbered.flatten();

    for (i, col) in flattened.iter().enumerate() {
        assert_eq!(*col, all_cols[i]);
    }

    assert_eq!(num_cols, flattened.len());
}

#[test]
fn test_is_less_than_tuple_chip() {
    let bus_index: usize = 0;
    let limb_bits: Vec<usize> = vec![16, 8];
    let decomp: usize = 6;
    let range_max: u32 = 1 << decomp;

    let range_checker = Arc::new(RangeCheckerGateChip::new(bus_index, range_max));

    let chip = IsLessThanTupleChip::new(bus_index, limb_bits, decomp, range_checker);
    let range_checker = chip.range_checker.as_ref();

    let trace = chip.generate_trace(vec![
        (vec![14321, 123], vec![26678, 233]),
        (vec![26678, 244], vec![14321, 233]),
        (vec![14321, 244], vec![14321, 244]),
        (vec![26678, 233], vec![14321, 244]),
    ]);
    let range_checker_trace = range_checker.generate_trace();
    run_simple_test_no_pis(
        vec![&chip.air, &range_checker.air],
        vec![trace, range_checker_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_is_less_than_tuple_chip_negative() {
    let bus_index: usize = 0;
    let limb_bits: Vec<usize> = vec![16, 8];
    let decomp: usize = 8;
    let range_max: u32 = 1 << decomp;

    let range_checker = Arc::new(RangeCheckerGateChip::new(bus_index, range_max));

    let chip = IsLessThanTupleChip::new(bus_index, limb_bits, decomp, range_checker);
    let range_checker = chip.range_checker.as_ref();
    let mut trace = chip.generate_trace(vec![(vec![14321, 123], vec![26678, 233])]);
    let range_checker_trace = range_checker.generate_trace();

    trace.values[2] = AbstractField::from_canonical_u64(0);

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(
            vec![&chip.air, &range_checker.air],
            vec![trace, range_checker_trace]
        ),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}
