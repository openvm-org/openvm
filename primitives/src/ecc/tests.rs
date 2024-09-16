use std::{str::FromStr, sync::Arc};

use afs_stark_backend::{utils::disable_debug_builder, verifier::VerificationError};
use ax_sdk::{any_rap_vec, config::baby_bear_blake3::BabyBearBlake3Engine, engine::StarkFriEngine};
use lazy_static::lazy_static;
use num_bigint_dig::BigUint;
use num_traits::{FromPrimitive, Zero};
use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;

use super::{
    air::{EcAddUnequalAir, EcAirConfig, EccDoubleAir},
    columns::EcDoubleCols,
};
use crate::{
    bigint::{utils::secp256k1_prime, DefaultLimbConfig, LimbConfig},
    ecc::columns::EcAddCols,
    sub_chip::LocalTraceInstructions,
    var_range::{bus::VariableRangeCheckerBus, VariableRangeCheckerChip},
};

lazy_static! {
    // Sample points got from https://asecuritysite.com/ecc/ecc_points2 and
    // https://learnmeabitcoin.com/technical/cryptography/elliptic-curve/#add
    static ref EcPoints: Vec<(BigUint, BigUint)> = {
        let x1 = BigUint::from_u32(1).unwrap();
        let y1 = BigUint::from_str(
            "29896722852569046015560700294576055776214335159245303116488692907525646231534",
        )
        .unwrap();
        let x2 = BigUint::from_u32(2).unwrap();
        let y2 = BigUint::from_str(
            "69211104694897500952317515077652022726490027694212560352756646854116994689233",
        )
        .unwrap();

        // This is the sum of (x1, y1) and (x2, y2).
        let x3 = BigUint::from_str("109562500687829935604265064386702914290271628241900466384583316550888437213118").unwrap();
        let y3 = BigUint::from_str(
            "54782835737747434227939451500021052510566980337100013600092875738315717035444",
        )
        .unwrap();

        // This is the double of (x2, y2).
        let x4 = BigUint::from_str(
            "23158417847463239084714197001737581570653996933128112807891516801581766934331").unwrap();
        let y4 = BigUint::from_str(
            "25821202496262252602076867233819373685524812798827903993634621255495124276396",
        )
        .unwrap();

        // This is the sum of (x3, y3) and (x4, y4).
        let x5 = BigUint::from_str("88733411122275068320336854419305339160905807011607464784153110222112026831518").unwrap();
        let y5 = BigUint::from_str(
            "69295025707265750480609159026651746584753914962418372690287755773539799515030",
        )
        .unwrap();

        vec![(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)]
    };
}

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
    let p1 = EcPoints[0].clone();
    let p2 = EcPoints[1].clone();
    let p3 = EcPoints[2].clone();
    test_ec_add(p1, p2, p3);
}

#[test]
fn test_ec_add2() {
    let p1 = EcPoints[2].clone();
    let p2 = EcPoints[3].clone();
    let p3 = EcPoints[4].clone();
    test_ec_add(p1, p2, p3);
}

#[test]
fn test_ec_add_fail() {
    let p1 = EcPoints[0].clone();
    let p2 = EcPoints[1].clone();
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
    let p1 = EcPoints[1].clone();
    let expected_double = EcPoints[3].clone();
    let (config, range_checker) = get_air_config_and_range_checker();
    let air = EccDoubleAir { config };
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
    let p1 = EcPoints[3].clone();
    let (config, range_checker) = get_air_config_and_range_checker();
    let air = EccDoubleAir { config };
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
