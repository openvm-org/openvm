use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use ax_sdk::config::baby_bear_poseidon2::run_simple_test_no_pis;
use p3_field::AbstractField;

use super::{columns::IsLessThanBitsCols, IsLessThanBitsAir};

#[test]
fn test_flatten_fromslice_roundtrip() {
    let limb_bits = 16;

    let num_cols = IsLessThanBitsCols::<usize>::get_width(limb_bits);
    let all_cols = (0..num_cols).collect::<Vec<usize>>();

    let cols_numbered = IsLessThanBitsCols::<usize>::from_slice(&all_cols);
    let flattened = cols_numbered.flatten();

    for (i, col) in flattened.iter().enumerate() {
        assert_eq!(*col, all_cols[i]);
    }

    assert_eq!(num_cols, flattened.len());
}

#[test]
fn test_is_less_than_bits_chip_lt() {
    std::env::set_var("RUST_BACKTRACE", "1");
    let limb_bits: usize = 16;

    let air = IsLessThanBitsAir { limb_bits };
    let trace = air.generate_trace(vec![(14321, 26883), (1, 0), (773, 773), (337, 456)]);
    //let trace = chip.generate_trace(vec![(0, 1)]);

    run_simple_test_no_pis(vec![&air], vec![trace]).expect("Verification failed");
}

#[test]
fn test_is_less_than_negative_1() {
    let limb_bits: usize = 16;

    let air = IsLessThanBitsAir { limb_bits };
    let mut trace = air.generate_trace(vec![(446, 553)]);

    trace.row_mut(0)[2] = AbstractField::from_canonical_u64(0);

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(vec![&air], vec![trace],),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}

#[test]
fn test_is_less_than_negative_2() {
    let limb_bits: usize = 16;

    let air = IsLessThanBitsAir { limb_bits };
    let mut trace = air.generate_trace(vec![(446, 342)]);

    trace.row_mut(0)[2] = AbstractField::from_canonical_u64(1);
    for d in 3..=3 + limb_bits {
        trace.row_mut(0)[d] = AbstractField::from_canonical_u64(0);
    }

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(vec![&air], vec![trace],),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}

#[test]
fn test_is_less_than_negative_3() {
    let limb_bits: usize = 1;

    let air = IsLessThanBitsAir { limb_bits };
    let mut trace = air.generate_trace(vec![(1, 1)]);

    trace.row_mut(0)[2] = AbstractField::from_canonical_u64(1);
    trace.row_mut(0)[3] = AbstractField::from_canonical_u64(2);
    trace.row_mut(0)[4] = AbstractField::from_canonical_u64(0);

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(vec![&air], vec![trace],),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}

/*#[test]
fn test_is_less_than_negative_2() {
    let limb_bits: usize = 16;

    let air = IsLessThanBitsAir { limb_bits };
    let mut trace = air.generate_trace(vec![(446, 447)]);

    trace.row_mut(0)[2] = AbstractField::from_canonical_u64(0);
    for d in 3 + limb_bits..3 + (2 * limb_bits) {
        trace.row_mut(0)[d] = AbstractField::from_canonical_u64(0);
    }

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(vec![&air], vec![trace],),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}

#[test]
fn test_is_less_than_negative_3() {
    let limb_bits: usize = 2;

    let air = IsLessThanBitsAir { limb_bits };
    let mut trace = air.generate_trace(vec![(0, 2)]);

    trace.row_mut(0)[3] = AbstractField::from_canonical_u64(2);
    trace.row_mut(0)[3 + 1] = AbstractField::from_canonical_u64(0);

    trace.row_mut(0)[3 + limb_bits] = AbstractField::from_canonical_u64(2);
    trace.row_mut(0)[3 + limb_bits + 1] = AbstractField::from_canonical_u64(2);

    trace.row_mut(0)[2] = AbstractField::from_canonical_u64(3);

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(vec![&air], vec![trace],),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}

#[test]
fn test_is_less_than_negative_4() {
    let limb_bits: usize = 2;

    let air = IsLessThanBitsAir { limb_bits };
    let mut trace = air.generate_trace(vec![(1, 0)]);

    trace.row_mut(0)[3 + limb_bits + 1] = AbstractField::from_canonical_u64(1);

    trace.row_mut(0)[2] = AbstractField::from_canonical_u64(1);

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(vec![&air], vec![trace],),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}*/
