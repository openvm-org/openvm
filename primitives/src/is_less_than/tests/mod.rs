use std::sync::Arc;

use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use ax_sdk::config::baby_bear_poseidon2::run_simple_test_no_pis;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::DenseMatrix;

use super::{super::is_less_than::IsLessThanChip, columns::IsLessThanCols};
use crate::{is_less_than::IsLessThanAir, range_gate::RangeCheckerGateChip};

#[test]
fn test_flatten_fromslice_roundtrip() {
    let lt_air = IsLessThanAir::new(0, 16, 8);

    let num_cols = IsLessThanCols::<usize>::width(&lt_air);
    let all_cols = (0..num_cols).collect::<Vec<usize>>();

    let cols_numbered = IsLessThanCols::<usize>::from_slice(&all_cols);
    let flattened = cols_numbered.flatten();

    for (i, col) in flattened.iter().enumerate() {
        assert_eq!(*col, all_cols[i]);
    }

    assert_eq!(num_cols, flattened.len());
}

#[test]
fn test_is_less_than_chip_lt() {
    let bus_index: usize = 0;
    let max_bits: usize = 16;
    let decomp: usize = 8;
    let range_max: u32 = 1 << decomp;

    let range_checker = Arc::new(RangeCheckerGateChip::new(bus_index, range_max));

    let chip = IsLessThanChip::new(bus_index, max_bits, decomp, range_checker);
    let trace = chip.generate_trace(vec![(14321, 26883), (1, 0), (773, 773), (337, 456)]);
    let range_trace: DenseMatrix<BabyBear> = chip.range_checker.generate_trace();

    run_simple_test_no_pis(
        vec![&chip.air, &chip.range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_lt_chip_decomp_does_not_divide() {
    let bus_index: usize = 0;
    let max_bits: usize = 30;
    let decomp: usize = 8;
    let range_max: u32 = 1 << decomp;

    let range_checker = Arc::new(RangeCheckerGateChip::new(bus_index, range_max));

    let chip = IsLessThanChip::new(bus_index, max_bits, decomp, range_checker);
    let trace = chip.generate_trace(vec![(14321, 26883), (1, 0), (773, 773), (337, 456)]);
    let range_trace: DenseMatrix<BabyBear> = chip.range_checker.generate_trace();

    run_simple_test_no_pis(
        vec![&chip.air, &chip.range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_is_less_than_negative() {
    let bus_index: usize = 0;
    let max_bits: usize = 16;
    let decomp: usize = 8;
    let range_max: u32 = 1 << decomp;

    let range_checker = Arc::new(RangeCheckerGateChip::new(bus_index, range_max));

    let chip = IsLessThanChip::new(bus_index, max_bits, decomp, range_checker);
    let mut trace = chip.generate_trace(vec![(446, 553)]);
    let range_trace = chip.range_checker.generate_trace();

    trace.values[2] = AbstractField::from_canonical_u64(0);

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(
            vec![&chip.air, &chip.range_checker.air],
            vec![trace, range_trace],
        ),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}
