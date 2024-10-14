use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use ax_sdk::{
    any_rap_arc_vec, config::baby_bear_poseidon2::BabyBearPoseidon2Engine, engine::StarkFriEngine,
};
use p3_field::AbstractField;

use super::{columns::IsLessThanTupleBitsCols, IsLessThanTupleBitsAir};

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
    BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(any_rap_arc_vec![air], vec![trace])
        .expect("Verification failed");
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
        BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(any_rap_arc_vec![air], vec![trace])
            .err(),
        Some(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}
