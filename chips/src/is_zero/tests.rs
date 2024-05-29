use crate::is_zero::air::IsZeroAir;
use crate::is_zero::trace::generate_trace_rows;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;

use afs_test_utils::config::baby_bear_poseidon2::run_simple_test;

#[test]
fn test_single_is_zero() {
    // Public inputs:
    let x = 97u32;
    let is_zero = 0u32;

    type Val = BabyBear;
    let pis = [x, is_zero].map(BabyBear::from_canonical_u32).to_vec();
    let air = IsZeroAir;

    let trace = generate_trace_rows::<Val>(x);

    run_simple_test(vec![&air], vec![trace], vec![pis]).expect("Verification failed");
}

#[test]
fn test_single_is_zero2() {
    // Public inputs:
    let x = 0u32;
    let is_zero = 1u32;

    type Val = BabyBear;
    let pis = [x, is_zero].map(BabyBear::from_canonical_u32).to_vec();
    let air = IsZeroAir;

    let trace = generate_trace_rows::<Val>(x);

    run_simple_test(vec![&air], vec![trace], vec![pis]).expect("Verification failed");
}

#[test]
fn test_single_is_zero_fail() {
    // Public inputs:
    let x = 187u32;
    let is_zero = 1u32;

    type Val = BabyBear;
    let pis = [x, is_zero].map(BabyBear::from_canonical_u32).to_vec();
    let air = IsZeroAir;

    let trace = generate_trace_rows::<Val>(x);

    run_simple_test(vec![&air], vec![trace], vec![pis]).expect("Verification failed");
}

#[test]
fn test_single_is_zero_fail2() {
    // Public inputs:
    let x = 0u32;
    let is_zero = 0u32;

    type Val = BabyBear;
    let pis = [x, is_zero].map(BabyBear::from_canonical_u32).to_vec();
    let air = IsZeroAir;

    let trace = generate_trace_rows::<Val>(x);

    run_simple_test(vec![&air], vec![trace], vec![pis]).expect("Verification failed");
}
