use std::{str::FromStr, sync::Arc};

use ax_sdk::config::baby_bear_blake3::run_simple_test_no_pis;
use lazy_static::lazy_static;
use num_bigint_dig::BigUint;
use num_traits::FromPrimitive;
use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;

use super::air::EccAir;
use crate::{
    bigint::{utils::secp256k1_prime, DefaultLimbConfig, LimbConfig},
    sub_chip::LocalTraceInstructions,
    var_range::{bus::VariableRangeCheckerBus, VariableRangeCheckerChip},
};

lazy_static! {
    // Sample points got from https://asecuritysite.com/ecc/ecc_points2.
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
        let x3 = BigUint::from_u32(3).unwrap();
        let y3 = BigUint::from_str(
            "94471189679404635060807731153122836805497974241028285133722790318709222555876",
        )
        .unwrap();
        let x4 = BigUint::from_u32(20).unwrap();
        let y4 = BigUint::from_str(
            "95115947350322555212584100192293494006877237570979160767752142956238074546829",
        )
        .unwrap();
        vec![(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    };
}

fn get_air_and_range_checker() -> (EccAir, Arc<VariableRangeCheckerChip>) {
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
    let air = EccAir::new(
        prime,
        b,
        range_bus,
        range_decomp,
        limb_bits,
        field_element_bits,
    );

    (air, range_checker)
}

fn test_ec_add(p1: (BigUint, BigUint), p2: (BigUint, BigUint)) {
    let (air, range_checker) = get_air_and_range_checker();
    let input = (p1, p2, range_checker.clone());
    let cols = air.generate_trace_row(input);

    let row = cols.flatten();
    let trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&air));
    let range_trace = range_checker.generate_trace();

    run_simple_test_no_pis(vec![&air, &range_checker.air], vec![trace, range_trace])
        .expect("Verification failed");
}

#[test]
fn test_ec_add1() {
    let p1 = EcPoints[0].clone();
    let p2 = EcPoints[1].clone();
    test_ec_add(p1, p2);
}

#[test]
fn test_ec_add2() {
    let p1 = EcPoints[1].clone();
    let p2 = EcPoints[0].clone();
    test_ec_add(p1, p2);
}

#[test]
fn test_ec_add3() {
    let p1 = EcPoints[2].clone();
    let p2 = EcPoints[3].clone();
    test_ec_add(p1, p2);
}

#[test]
fn test_ec_add4() {
    let p1 = EcPoints[3].clone();
    let p2 = EcPoints[1].clone();
    test_ec_add(p1, p2);
}

#[test]
#[should_panic]
fn test_ec_add_fail() {
    let p1 = EcPoints[0].clone();
    let p2 = EcPoints[1].clone();
    let (air, range_checker) = get_air_and_range_checker();
    let input = (p1, p2, range_checker.clone());
    let cols = air.generate_trace_row(input);

    let row = cols.flatten();
    let mut trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&air));
    let range_trace = range_checker.generate_trace();
    trace.row_mut(0)[0] += BabyBear::one();

    run_simple_test_no_pis(vec![&air, &range_checker.air], vec![trace, range_trace])
        .expect("Verification failed");
}
