use std::sync::Arc;

use ax_sdk::{config::baby_bear_blake3::run_simple_test_no_pis, utils::create_seeded_rng};
use num_bigint_dig::BigUint;
use num_traits::{One, Zero};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::RngCore;

use crate::{
    modular_multiplication::bigint::{
        air::ModularArithmeticBigIntAir, columns::ModularArithmeticBigIntCols,
    },
    var_range::{bus::VariableRangeCheckerBus, VariableRangeCheckerChip},
};

fn secp256k1_prime() -> BigUint {
    let mut result = BigUint::one() << 256;
    for power in [32, 9, 8, 7, 6, 4, 0] {
        result -= BigUint::one() << power;
    }
    result
}

fn default_air() -> ModularArithmeticBigIntAir {
    ModularArithmeticBigIntAir::new(secp256k1_prime(), 256, 16, 0, 30, 30, 10, 16, 1 << 15)
}

#[test]
fn test_flatten_fromslice_roundtrip() {
    let air = default_air();

    let num_cols = ModularArithmeticBigIntCols::<usize>::get_width(&air);
    let all_cols = (0..num_cols).collect::<Vec<usize>>();

    let cols_numbered = ModularArithmeticBigIntCols::<usize>::from_slice(&all_cols, &air);
    let flattened = cols_numbered.flatten();

    for (i, col) in flattened.iter().enumerate() {
        assert_eq!(*col, all_cols[i]);
    }

    assert_eq!(num_cols, flattened.len());
}

#[test]
fn test_modular_multiplication_bigint_1() {
    let air = default_air();
    let num_digits = 8;
    let range_checker = Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
        air.range_bus,
        air.decomp,
    )));

    let mut rng = create_seeded_rng();
    let a_digits = (0..num_digits).map(|_| rng.next_u32()).collect();
    let a = BigUint::new(a_digits);
    let b_digits = (0..num_digits).map(|_| rng.next_u32()).collect();
    let b = BigUint::new(b_digits);
    // if these are not true then trace is not guaranteed to be verifiable
    assert!(a < secp256k1_prime());
    assert!(b < secp256k1_prime());

    let trace = air.generate_trace(vec![(a, b)], range_checker.clone());
    let range_trace = range_checker.generate_trace();
    run_simple_test_no_pis(vec![&air, &range_checker.air], vec![trace, range_trace])
        .expect("Verification failed");
}

#[test]
fn test_modular_multiplication_bigint_2() {
    let air = default_air();
    let num_digits = 8;
    let range_checker = Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
        air.range_bus,
        air.decomp,
    )));

    let trace_degree = 16;
    let mut rng = create_seeded_rng();

    let inputs = (0..trace_degree)
        .map(|_| {
            let a_digits = (0..num_digits).map(|_| rng.next_u32()).collect();
            let a = BigUint::new(a_digits);
            let b_digits = (0..num_digits).map(|_| rng.next_u32()).collect();
            let b = BigUint::new(b_digits);
            // if these are not true then trace is not guaranteed to be verifiable
            assert!(a < secp256k1_prime());
            assert!(b < secp256k1_prime());
            (a, b)
        })
        .collect();

    let trace = air.generate_trace(inputs, range_checker.clone());
    let range_trace = range_checker.generate_trace();
    run_simple_test_no_pis(vec![&air, &range_checker.air], vec![trace, range_trace])
        .expect("Verification failed");
}

#[test]
fn test_modular_multiplication_bigint_zero() {
    let air = default_air();
    let range_checker = Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
        air.range_bus,
        air.decomp,
    )));

    let trace = air.generate_trace(
        vec![(BigUint::zero(), BigUint::zero())],
        range_checker.clone(),
    );
    let range_trace = range_checker.generate_trace();
    run_simple_test_no_pis(vec![&air, &range_checker.air], vec![trace, range_trace])
        .expect("Verification failed");
}

#[test]
#[should_panic]
fn test_modular_multiplication_bigint_negative() {
    std::env::set_var("RUST_BACKTRACE", "1");
    let air = default_air();
    let num_digits = 8;
    let range_checker = Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
        air.range_bus,
        air.decomp,
    )));

    let digits = (0..num_digits).map(|_| u32::MAX).collect();
    let a = BigUint::new(digits);
    println!("{}", a);

    let trace = air.generate_trace(vec![(a.clone(), a)], range_checker.clone());
    let range_trace = range_checker.generate_trace();
    run_simple_test_no_pis(vec![&air, &range_checker.air], vec![trace, range_trace])
        .expect("Verification failed");
}

#[test]
#[should_panic]
fn test_modular_multiplication_bigint_negative_2() {
    let air = default_air();
    let num_digits = 8;
    let range_checker = Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
        air.range_bus,
        air.decomp,
    )));

    let mut rng = create_seeded_rng();
    let a_digits = (0..num_digits).map(|_| rng.next_u32()).collect();
    let a = BigUint::new(a_digits);
    let b_digits = (0..num_digits).map(|_| rng.next_u32()).collect();
    let b = BigUint::new(b_digits);
    // if these are not true then trace is not guaranteed to be verifiable
    assert!(a < secp256k1_prime());
    assert!(b < secp256k1_prime());

    let mut trace = air.generate_trace(vec![(a, b)], range_checker.clone());
    trace.row_mut(0)[0] += BabyBear::one();
    let range_trace = range_checker.generate_trace();
    run_simple_test_no_pis(vec![&air, &range_checker.air], vec![trace, range_trace])
        .expect("Verification failed");
}
