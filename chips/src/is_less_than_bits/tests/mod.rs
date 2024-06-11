use super::columns::IsLessThanBitsCols;
use super::IsLessThanBitsChip;

use afs_stark_backend::prover::USE_DEBUG_BUILDER;
use afs_stark_backend::verifier::VerificationError;
use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;
use p3_field::AbstractField;

#[test]
fn test_flatten_fromslice_roundtrip() {
    let limb_bits = 16;

    let num_cols = IsLessThanBitsCols::<usize>::get_width(limb_bits);
    let all_cols = (0..num_cols).collect::<Vec<usize>>();

    let cols_numbered = IsLessThanBitsCols::<usize>::from_slice(limb_bits, &all_cols);
    let flattened = cols_numbered.flatten();

    for (i, col) in flattened.iter().enumerate() {
        assert_eq!(*col, all_cols[i]);
    }

    assert_eq!(num_cols, flattened.len());
}

#[test]
fn test_is_less_than_bits_chip_lt() {
    let limb_bits: usize = 16;

    let chip = IsLessThanBitsChip::new(limb_bits);
    let trace = chip.generate_trace(vec![(14321, 26883), (1, 0), (773, 773), (337, 456)]);
    //let trace = chip.generate_trace(vec![(0, 1)]);

    run_simple_test_no_pis(vec![&chip.air], vec![trace]).expect("Verification failed");
}

#[test]
fn test_is_less_than_negative_1() {
    let limb_bits: usize = 16;

    let chip = IsLessThanBitsChip::new(limb_bits);
    let mut trace = chip.generate_trace(vec![(446, 553)]);

    trace.values[2] = AbstractField::from_canonical_u64(0);

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(vec![&chip.air], vec![trace],),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}

#[test]
fn test_is_less_than_negative_2() {
    let limb_bits: usize = 16;

    let chip = IsLessThanBitsChip::new(limb_bits);
    let mut trace = chip.generate_trace(vec![(446, 447)]);

    trace.values[2] = AbstractField::from_canonical_u64(0);
    for d in 3 + (2 * limb_bits)..3 + (3 * limb_bits) {
        trace.values[d] = AbstractField::from_canonical_u64(0);
    }

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(vec![&chip.air], vec![trace],),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}

#[test]
fn test_is_less_than_negative_3() {
    let limb_bits: usize = 2;

    let chip = IsLessThanBitsChip::new(limb_bits);
    let mut trace = chip.generate_trace(vec![(0, 2)]);

    trace.values[3 + limb_bits] = AbstractField::from_canonical_u64(2);
    trace.values[3 + limb_bits + 1] = AbstractField::from_canonical_u64(0);

    trace.values[3 + (2 * limb_bits)] = AbstractField::from_canonical_u64(2);
    trace.values[3 + (2 * limb_bits) + 1] = AbstractField::from_canonical_u64(2);

    trace.values[2] = AbstractField::from_canonical_u64(2);

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(vec![&chip.air], vec![trace],),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}
