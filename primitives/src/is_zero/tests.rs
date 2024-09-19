use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use ax_sdk::{
    any_rap_vec, config::baby_bear_poseidon2::BabyBearPoseidon2Engine, engine::StarkFriEngine,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::Matrix;
use test_case::test_case;

use super::IsZeroAir;

#[test_case(97 ; "97 => 0")]
#[test_case(0 ; "0 => 1")]
fn test_single_is_zero(x: u32) {
    let x = AbstractField::from_canonical_u32(x);

    let chip = IsZeroAir {};

    let trace = chip.generate_trace(vec![x]);

    assert_eq!(
        trace.get(0, 1),
        AbstractField::from_bool(x == AbstractField::zero())
    );

    BabyBearPoseidon2Engine::run_simple_test_no_pis(&any_rap_vec![&chip], vec![trace])
        .expect("Verification failed");
}

#[test_case([0, 1, 2, 7], [1, 0, 0, 0] ; "0, 1, 2, 7 => 1, 0, 0, 0")]
#[test_case([97, 23, 179, 0], [0, 0, 0, 1] ; "97, 23, 179, 0 => 0, 0, 0, 1")]
fn test_vec_is_zero(x_vec: [u32; 4], expected: [u32; 4]) {
    let x_vec = x_vec
        .iter()
        .map(|x| AbstractField::from_canonical_u32(*x))
        .collect();

    let chip = IsZeroAir {};

    let trace = chip.generate_trace(x_vec);

    for (i, value) in expected.iter().enumerate() {
        assert_eq!(
            trace.values[3 * i + 1],
            AbstractField::from_canonical_u32(*value)
        );
    }

    BabyBearPoseidon2Engine::run_simple_test_no_pis(&any_rap_vec![&chip], vec![trace])
        .expect("Verification failed");
}

#[test_case(97 ; "97 => 0")]
#[test_case(0 ; "0 => 1")]
fn test_single_is_zero_fail(x: u32) {
    let x = AbstractField::from_canonical_u32(x);

    let chip = IsZeroAir {};

    let mut trace = chip.generate_trace(vec![x]);
    trace.values[1] = BabyBear::one() - trace.values[1];

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        BabyBearPoseidon2Engine::run_simple_test_no_pis(&any_rap_vec![&chip], vec![trace]).err(),
        Some(VerificationError::OodEvaluationMismatch),
        "Expected constraint to fail"
    );
}

#[test_case([1, 2, 7, 0], [0, 0, 0, 1] ; "1, 2, 7, 0 => 0, 0, 0, 1")]
#[test_case([97, 0, 179, 0], [0, 1, 0, 1] ; "97, 0, 179, 0 => 0, 1, 0, 1")]
fn test_vec_is_zero_fail(x_vec: [u32; 4], expected: [u32; 4]) {
    let x_vec: Vec<BabyBear> = x_vec
        .iter()
        .map(|x| BabyBear::from_canonical_u32(*x))
        .collect();

    let chip = IsZeroAir {};

    let mut trace = chip.generate_trace(x_vec);

    for (i, _value) in expected.iter().enumerate() {
        trace.row_mut(i)[1] = BabyBear::one() - trace.row_mut(i)[1];
        USE_DEBUG_BUILDER.with(|debug| {
            *debug.lock().unwrap() = false;
        });
        assert_eq!(
            BabyBearPoseidon2Engine::run_simple_test_no_pis(
                &any_rap_vec![&chip],
                vec![trace.clone()]
            )
            .err(),
            Some(VerificationError::OodEvaluationMismatch),
            "Expected constraint to fail"
        );
        trace.row_mut(i)[1] = BabyBear::one() - trace.row_mut(i)[1];
    }
}
