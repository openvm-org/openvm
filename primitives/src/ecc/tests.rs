use std::sync::Arc;

use afs_stark_backend::{utils::disable_debug_builder, verifier::VerificationError};
use ax_sdk::{any_rap_vec, config::baby_bear_blake3::BabyBearBlake3Engine, engine::StarkFriEngine};
use num_bigint_dig::BigUint;
use num_traits::{FromPrimitive, Zero};
use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;

use super::{
    air::{EcAddUnequalAir, EcAirConfig, EcDoubleAir},
    columns::EcDoubleCols,
    SampleEcPoints,
};
use crate::{
    bigint::{utils::secp256k1_prime, DefaultLimbConfig, LimbConfig},
    ecc::columns::EcAddCols,
    sub_chip::LocalTraceInstructions,
    var_range::{bus::VariableRangeCheckerBus, VariableRangeCheckerChip},
};

fn evaluate_bigint(limbs: &[BabyBear], limb_bits: usize) -> BigUint {
    let mut res = BigUint::zero();
    let base = BigUint::from_u64(1 << limb_bits).unwrap();
    for limb in limbs.iter().rev() {
        res = res * base.clone() + BigUint::from_u64(limb.as_canonical_u64()).unwrap();
    }
    res
}

fn get_air_config_and_range_checker() -> (EcAirConfig, Arc<VariableRangeCheckerChip>) {
    let prime = secp256k1_prime();
    let b = BigUint::from_u32(7).unwrap();
    let range_bus = 1;
    let range_decomp = 18;
    let range_checker = Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
        range_bus,
        range_decomp,
    )));
    let limb_bits = DefaultLimbConfig::limb_bits();
    let field_element_bits = 30;
    let config = EcAirConfig::new(
        prime,
        b,
        range_bus,
        range_decomp,
        limb_bits,
        field_element_bits,
    );

    (config, range_checker)
}

fn test_ec_add(p1: (BigUint, BigUint), p2: (BigUint, BigUint), expected_p3: (BigUint, BigUint)) {
    let (config, range_checker) = get_air_config_and_range_checker();
    let air = EcAddUnequalAir { config };
    let input = (p1, p2, range_checker.clone());
    let cols = air.generate_trace_row(input);
    let EcAddCols { io, aux: _ } = cols.clone();
    let generated_x = evaluate_bigint(&io.p3.x.limbs, 8);
    let generated_y = evaluate_bigint(&io.p3.y.limbs, 8);
    assert_eq!(generated_x, expected_p3.0);
    assert_eq!(generated_y, expected_p3.1);

    let row = cols.flatten();
    let trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&air));
    let range_trace = range_checker.generate_trace();

    BabyBearBlake3Engine::run_simple_test_no_pis(
        &any_rap_vec![&air, &range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_ec_add1() {
    let p1 = SampleEcPoints[0].clone();
    let p2 = SampleEcPoints[1].clone();
    let p3 = SampleEcPoints[2].clone();
    test_ec_add(p1, p2, p3);
}

#[test]
fn test_ec_add2() {
    let p1 = SampleEcPoints[2].clone();
    let p2 = SampleEcPoints[3].clone();
    let p3 = SampleEcPoints[4].clone();
    test_ec_add(p1, p2, p3);
}

#[test]
fn test_ec_add_fail() {
    let p1 = SampleEcPoints[0].clone();
    let p2 = SampleEcPoints[1].clone();
    let (config, range_checker) = get_air_config_and_range_checker();
    let air = EcAddUnequalAir { config };
    let input = (p1, p2, range_checker.clone());
    let cols = air.generate_trace_row(input);

    let row = cols.flatten();
    let mut trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&air));
    let range_trace = range_checker.generate_trace();
    trace.row_mut(0)[0] += BabyBear::one();

    disable_debug_builder();
    assert_eq!(
        BabyBearBlake3Engine::run_simple_test_no_pis(
            &any_rap_vec![&air, &range_checker.air],
            vec![trace, range_trace]
        )
        .err(),
        Some(VerificationError::OodEvaluationMismatch),
        "Expected constraint to fail"
    );
}

#[test]
fn test_ec_double() {
    let p1 = SampleEcPoints[1].clone();
    let expected_double = SampleEcPoints[3].clone();
    let (config, range_checker) = get_air_config_and_range_checker();
    let air = EcDoubleAir { config };
    let input = (p1, range_checker.clone());
    let cols = air.generate_trace_row(input);

    let EcDoubleCols { io, aux: _ } = cols.clone();
    let generated_x = evaluate_bigint(&io.p2.x.limbs, 8);
    let generated_y = evaluate_bigint(&io.p2.y.limbs, 8);
    assert_eq!(generated_x, expected_double.0);
    assert_eq!(generated_y, expected_double.1);
    let row = cols.flatten();
    let trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&air));
    let range_trace = range_checker.generate_trace();

    BabyBearBlake3Engine::run_simple_test_no_pis(
        &any_rap_vec![&air, &range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_ec_double_wrong_trace() {
    let p1 = SampleEcPoints[3].clone();
    let (config, range_checker) = get_air_config_and_range_checker();
    let air = EcDoubleAir { config };
    let input = (p1, range_checker.clone());
    let cols = air.generate_trace_row(input);

    let row = cols.flatten();
    let mut trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&air));
    trace.row_mut(0)[0] += BabyBear::one();
    let range_trace = range_checker.generate_trace();

    disable_debug_builder();
    assert_eq!(
        BabyBearBlake3Engine::run_simple_test_no_pis(
            &any_rap_vec![&air, &range_checker.air],
            vec![trace, range_trace]
        )
        .err(),
        Some(VerificationError::OodEvaluationMismatch),
        "Expected constraint to fail"
    );
}
