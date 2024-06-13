use super::columns::IsLessThanTupleBitsCols;

use super::IsLessThanTupleBitsAir;

use afs_stark_backend::prover::USE_DEBUG_BUILDER;
use afs_stark_backend::verifier::VerificationError;
use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;
use p3_field::AbstractField;

#[test]
fn test_flatten_fromslice_roundtrip() {
    std::env::set_var("RUST_BACKTRACE", "1");
    let limb_bits = vec![16, 8, 20, 20];
    let tuple_len = 4;

    let num_cols = IsLessThanTupleBitsCols::<usize>::get_width(limb_bits.clone(), tuple_len);
    let all_cols = (0..num_cols).collect::<Vec<usize>>();

    let cols_numbered =
        IsLessThanTupleBitsCols::<usize>::from_slice(&all_cols, limb_bits.clone(), tuple_len);
    let flattened = cols_numbered.flatten();

    for (i, col) in flattened.iter().enumerate() {
        assert_eq!(*col, all_cols[i]);
    }

    assert_eq!(num_cols, flattened.len());
}

#[test]
fn test_is_less_than_tuple_chip() {
    let limb_bits: Vec<usize> = vec![16, 8];

    let air = IsLessThanTupleBitsAir::new(limb_bits);

    let trace = air.generate_trace(vec![
        (vec![14321, 123], vec![26678, 233]),
        (vec![26678, 244], vec![14321, 233]),
        (vec![14321, 244], vec![14321, 244]),
        (vec![26678, 233], vec![14321, 244]),
    ]);

    run_simple_test_no_pis(vec![&air], vec![trace]).expect("Verification failed");
}

#[test]
fn test_is_less_than_tuple_chip_negative() {
    let limb_bits: Vec<usize> = vec![16, 8];
    let air = IsLessThanTupleBitsAir::new(limb_bits);
    let mut trace = air.generate_trace(vec![(vec![14321, 123], vec![26678, 233])]);

    trace.row_mut(0)[2] = AbstractField::from_canonical_u64(0);

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(vec![&air], vec![trace]),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}
