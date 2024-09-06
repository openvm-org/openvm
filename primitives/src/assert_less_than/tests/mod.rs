use std::sync::Arc;

use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use ax_sdk::config::baby_bear_poseidon2::run_simple_test_no_pis;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::DenseMatrix;

use super::{super::assert_less_than::AssertLessThanChip, columns::AssertLessThanCols};
use crate::{
    assert_less_than::AssertLessThanAir, range::bus::RangeCheckBus, range_gate::RangeCheckerGateChip,
};

#[test]
fn test_flatten_fromslice_roundtrip() {
    const AUX_LEN: usize = 2; // number of auxilliary columns is two

    let lt_air = AssertLessThanAir::<AUX_LEN>::new(RangeCheckBus::new(0, 1 << 8), 16, 8);

    let num_cols = AssertLessThanCols::<usize, AUX_LEN>::width(&lt_air);
    let all_cols = (0..num_cols).collect::<Vec<usize>>();

    let cols_numbered = AssertLessThanCols::<usize, AUX_LEN>::from_slice(&all_cols);
    let flattened = cols_numbered.flatten();

    for (i, col) in flattened.iter().enumerate() {
        assert_eq!(*col, all_cols[i]);
    }

    assert_eq!(num_cols, flattened.len());
}

#[test]
fn test_assert_less_than_chip_lt() {
    let max_bits: usize = 16;
    let decomp: usize = 8;
    let range_max: u32 = 1 << decomp;
    let bus = RangeCheckBus::new(0, range_max);
    const AUX_LEN: usize = 2;

    let range_checker = Arc::new(RangeCheckerGateChip::new(bus));
    
    let chip = AssertLessThanChip::<AUX_LEN>::new(bus, max_bits, decomp, range_checker);
    let trace = chip.generate_trace(vec![(14321, 26883), (0, 1), (28, 120), (337, 456)]);
    let range_trace: DenseMatrix<BabyBear> = chip.range_checker.generate_trace();
    
    run_simple_test_no_pis(
        vec![&chip.air, &chip.range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_lt_chip_decomp_does_not_divide() {
    let max_bits: usize = 29;
    let decomp: usize = 8;
    let range_max: u32 = 1 << decomp;
    let bus = RangeCheckBus::new(0, range_max);
    const AUX_LEN: usize = 5;

    let range_checker = Arc::new(RangeCheckerGateChip::new(bus));
    
    let chip = AssertLessThanChip::<AUX_LEN>::new(bus, max_bits, decomp, range_checker);
    let trace = chip.generate_trace(vec![(14321, 26883), (0, 1), (28, 120), (337, 456)]);
    let range_trace: DenseMatrix<BabyBear> = chip.range_checker.generate_trace();
    
    run_simple_test_no_pis(
        vec![&chip.air, &chip.range_checker.air],
        vec![trace, range_trace],
        )
        .expect("Verification failed");
}

#[test]
fn test_assert_less_than_negative_1() {
    let max_bits: usize = 16;
    let decomp: usize = 8;
    let range_max: u32 = 1 << decomp;
    let bus = RangeCheckBus::new(0, range_max);
    const AUX_LEN: usize = 2;

    let range_checker = Arc::new(RangeCheckerGateChip::new(bus));
    
    let chip = AssertLessThanChip::<AUX_LEN>::new(bus, max_bits, decomp, range_checker);
    let mut trace = chip.generate_trace(vec![(28, 29)]);
    let range_trace = chip.range_checker.generate_trace();

    // Make the trace invalid  
    trace.values.swap(0, 1);

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

#[test]
fn test_assert_less_than_negative_2() {
    let max_bits: usize = 29;
    let decomp: usize = 8;
    let range_max: u32 = 1 << decomp;
    let bus = RangeCheckBus::new(0, range_max);
    const AUX_LEN: usize = 5;
    let range_checker = Arc::new(RangeCheckerGateChip::new(bus));
    
    let chip = AssertLessThanChip::<AUX_LEN>::new(bus, max_bits, decomp, range_checker);
    let mut trace = chip.generate_trace(vec![(28, 29)]);
    let range_trace = chip.range_checker.generate_trace();

    // Make the trace invalid
    trace.values[2] = AbstractField::from_canonical_u64(range_max as u64);

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