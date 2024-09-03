use std::sync::Arc;

use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use ax_sdk::config::baby_bear_poseidon2::run_simple_test_no_pis;
use p3_field::AbstractField;

use super::super::is_less_than_tuple::IsLessThanTupleChip;
use crate::{
    is_less_than_tuple::{columns::IsLessThanTupleCols, IsLessThanTupleAir},
    range::bus::RangeCheckBus,
    range_gate::RangeCheckerGateChip,
};

#[test]
fn test_flatten_fromslice_roundtrip() {
    let limb_bits = vec![16, 8, 20, 20];
    let decomp = 8;
    let bus = RangeCheckBus::new(0, 1 << decomp);

    let lt_air = IsLessThanTupleAir::new(bus, limb_bits.clone(), decomp);

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
    let decomp: usize = 6;
    let bus = RangeCheckBus::new(0, 1 << decomp);

    let range_checker = Arc::new(RangeCheckerGateChip::new(bus));

    let chip = IsLessThanTupleChip::new(limb_bits, decomp, range_checker);
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
    let limb_bits: Vec<usize> = vec![16, 8];
    let decomp: usize = 8;
    let bus = RangeCheckBus::new(0, 1 << decomp);

    let range_checker = Arc::new(RangeCheckerGateChip::new(bus));

    let chip = IsLessThanTupleChip::new(limb_bits, decomp, range_checker);
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
