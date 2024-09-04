use std::{str::FromStr, sync::Arc};

use ax_sdk::config::baby_bear_blake3::run_simple_test_no_pis;
use num_bigint_dig::BigUint;
use num_traits::FromPrimitive;
use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use p3_matrix::dense::RowMajorMatrix;

use super::EccAir;
use crate::{
    bigint::{utils::secp256k1_prime, DefaultLimbConfig, LimbConfig},
    range::bus::RangeCheckBus,
    range_gate::RangeCheckerGateChip,
    sub_chip::LocalTraceInstructions,
};

fn get_air_and_range_checker() -> (EccAir, Arc<RangeCheckerGateChip>) {
    let prime = secp256k1_prime();
    let b = BigUint::from_u32(7).unwrap();
    let range_bus = 1;
    let range_decomp = 18;
    let range_checker = Arc::new(RangeCheckerGateChip::new(RangeCheckBus::new(
        range_bus,
        1 << range_decomp,
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

#[test]
fn test_ec_add() {
    let (air, range_checker) = get_air_and_range_checker();

    // Sample points got from https://asecuritysite.com/ecc/ecc_points2.
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

    let input = ((x1, y1), (x2, y2), range_checker.clone());
    let cols = air.generate_trace_row(input);

    let row = cols.flatten();
    let trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&air));
    let range_trace = range_checker.generate_trace();

    run_simple_test_no_pis(vec![&air, &range_checker.air], vec![trace, range_trace])
        .expect("Verification failed");
}
