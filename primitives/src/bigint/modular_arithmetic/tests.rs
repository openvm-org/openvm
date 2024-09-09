use std::sync::Arc;

use ax_sdk::{config::baby_bear_blake3::run_simple_test_no_pis, utils::create_seeded_rng};
use num_bigint_dig::BigUint;
use num_traits::{FromPrimitive, One, Zero};
use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_ceil_usize;
use rand::RngCore;

use super::{
    super::utils::secp256k1_prime, add::*, mul::*, sub::*, ModularArithmeticAir,
    ModularArithmeticCols,
};
use crate::{
    range::bus::RangeCheckBus, range_gate::RangeCheckerGateChip, sub_chip::LocalTraceInstructions,
};
// 256 bit prime, 10 limb bits -> 26 limbs.
const LIMB_BITS: usize = 10;
const NUM_LIMB: usize = 26;

fn evaluate_bigint(limbs: &[BabyBear], limb_bits: usize) -> BigUint {
    let mut res = BigUint::zero();
    let base = BigUint::from_u64(1 << limb_bits).unwrap();
    for limb in limbs.iter().rev() {
        res = res * base.clone() + BigUint::from_u64(limb.as_canonical_u64()).unwrap();
    }
    res
}

fn get_air_and_range_checker(
    prime: BigUint,
    limb_bits: usize,
    num_limbs: usize,
    is_mul_div: bool,
) -> (ModularArithmeticAir, Arc<RangeCheckerGateChip>) {
    let limb_max_abs = if is_mul_div {
        // The equation: x*y - p*q - r, or y*r - x - pq. with num_limbs N = 26
        // Abs of each limb of the equation can be as much as 2^10 * 2^10 * N * 2 + 2^10
        (1 << (2 * limb_bits)) * num_limbs * 2 + (1 << limb_bits)
    } else {
        // x +- y -r -pq
        (1 << (2 * limb_bits)) + (1 << limb_bits) * 3
    };
    // overflow bits: log(max_abs) => 26
    let max_overflow_bits = log2_ceil_usize(limb_max_abs);

    let range_bus = 1;
    let range_decomp = 17;
    let range_checker = Arc::new(RangeCheckerGateChip::new(RangeCheckBus::new(
        range_bus,
        1 << range_decomp,
    )));
    let q_limbs = if is_mul_div { num_limbs } else { 1 };
    let carry_limbs = if is_mul_div {
        2 * num_limbs - 1
    } else {
        num_limbs
    };
    let air = ModularArithmeticAir::new(
        prime,
        limb_bits,
        max_overflow_bits,
        num_limbs,
        q_limbs,
        carry_limbs,
        range_bus,
        range_decomp,
    );
    (air, range_checker)
}

fn generate_xy() -> (BigUint, BigUint) {
    let mut rng = create_seeded_rng();
    let len = 8; // in bytes -> 256 bits.
    let x = (0..len).map(|_| rng.next_u32()).collect();
    let x = BigUint::new(x);
    let y = (0..len).map(|_| rng.next_u32()).collect();
    let y = BigUint::new(y);
    (x, y)
}

#[test]
fn test_x_mul_y() {
    let prime = secp256k1_prime();
    let (x, y) = generate_xy();

    let (arithmetic, range_checker) =
        get_air_and_range_checker(prime.clone(), LIMB_BITS, NUM_LIMB, true);
    let air = ModularMultiplicationAir { arithmetic };
    let expected_r = x.clone() * y.clone() % prime.clone();
    let expected_q = x.clone() * y.clone() / prime;
    let cols = air.generate_trace_row((x, y, range_checker.clone()));
    let ModularArithmeticCols {
        x: _x,
        y: _y,
        q,
        r,
        carries: _carries,
    } = cols.clone();
    let generated_r = evaluate_bigint(&r, LIMB_BITS);
    let generated_q = evaluate_bigint(&q, LIMB_BITS);
    assert_eq!(generated_r, expected_r);
    assert_eq!(generated_q, expected_q);

    let row = cols.flatten();
    let trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&air));
    let range_trace = range_checker.generate_trace();

    run_simple_test_no_pis(vec![&air, &range_checker.air], vec![trace, range_trace])
        .expect("Verification failed");
}

#[test]
fn test_x_mul_zero() {
    let prime = secp256k1_prime();
    let (x, _) = generate_xy();
    let y = BigUint::zero();

    let (arithmetic, range_checker) =
        get_air_and_range_checker(prime.clone(), LIMB_BITS, NUM_LIMB, true);
    let air = ModularMultiplicationAir { arithmetic };
    let cols = air.generate_trace_row((x, y, range_checker.clone()));

    let row = cols.flatten();
    let trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&air));
    let range_trace = range_checker.generate_trace();

    run_simple_test_no_pis(vec![&air, &range_checker.air], vec![trace, range_trace])
        .expect("Verification failed");
}

#[test]
fn test_x_mul_one() {
    let prime = secp256k1_prime();
    let (x, _) = generate_xy();
    let y = BigUint::one();

    let (arithmetic, range_checker) =
        get_air_and_range_checker(prime.clone(), LIMB_BITS, NUM_LIMB, true);
    let air = ModularMultiplicationAir { arithmetic };
    let cols = air.generate_trace_row((x, y, range_checker.clone()));

    let row = cols.flatten();
    let trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&air));
    let range_trace = range_checker.generate_trace();

    run_simple_test_no_pis(vec![&air, &range_checker.air], vec![trace, range_trace])
        .expect("Verification failed");
}

#[test]
#[should_panic]
fn test_x_mul_y_wrong_trace() {
    let prime = secp256k1_prime();
    let (x, y) = generate_xy();

    let (arithmetic, range_checker) =
        get_air_and_range_checker(prime.clone(), LIMB_BITS, NUM_LIMB, true);
    let air = ModularMultiplicationAir { arithmetic };
    let cols = air.generate_trace_row((x, y, range_checker.clone()));

    let row = cols.flatten();
    let mut trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&air));
    trace.row_mut(0)[0] += BabyBear::one();
    let range_trace = range_checker.generate_trace();

    run_simple_test_no_pis(vec![&air, &range_checker.air], vec![trace, range_trace])
        .expect("Verification failed");
}

/*
#[test]
fn test_x_div_y() {
    let prime = secp256k1_prime();
    let (x, y) = generate_xy();
    let exp = prime.clone() - BigUint::from_u8(2).unwrap();
    let y_inv = y.modpow(&exp, &prime);

    let expected_r = x.clone() * y_inv.clone() % prime.clone();

    let (arithmetic, range_checker) =
        get_air_and_range_checker(prime.clone(), LIMB_BITS, NUM_LIMB, true);
    let air = ModularDivisionAir { arithmetic };
    let expected_q = (y.clone() * expected_r.clone() - x.clone()) / prime.clone();
    let cols = air.generate_trace_row((x.clone(), y.clone(), range_checker.clone()));
    let ModularArithmeticCols {
        x: _x,
        y: _y,
        q,
        r,
        carries: _carries,
    } = cols.clone();
    let generated_r = evaluate_bigint(&r, LIMB_BITS);
    let generated_q = evaluate_bigint(&q, LIMB_BITS);
    assert_eq!(generated_r, expected_r);
    assert_eq!(generated_q, expected_q);

    let row = cols.flatten();
    let trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&air));
    let range_trace = range_checker.generate_trace();

    run_simple_test_no_pis(vec![&air, &range_checker.air], vec![trace, range_trace])
        .expect("Verification failed");
}

#[test]
#[should_panic]
fn test_x_div_zero() {
    let prime = secp256k1_prime();
    let (x, _) = generate_xy();
    let y = BigUint::zero();

    let (arithmetic, range_checker) =
        get_air_and_range_checker(prime.clone(), LIMB_BITS, NUM_LIMB, true);
    let air = ModularDivisionAir { arithmetic };
    let cols = air.generate_trace_row((x.clone(), y.clone(), range_checker.clone()));
    let row = cols.flatten();
    let trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&air));
    let range_trace = range_checker.generate_trace();

    run_simple_test_no_pis(vec![&air, &range_checker.air], vec![trace, range_trace])
        .expect("Verification failed");
}

#[test]
fn test_x_div_one() {
    let prime = secp256k1_prime();
    let (x, _) = generate_xy();
    let y = BigUint::one();

    let (arithmetic, range_checker) =
        get_air_and_range_checker(prime.clone(), LIMB_BITS, NUM_LIMB, true);
    let air = ModularDivisionAir { arithmetic };
    let cols = air.generate_trace_row((x.clone(), y.clone(), range_checker.clone()));
    let row = cols.flatten();
    let trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&air));
    let range_trace = range_checker.generate_trace();

    run_simple_test_no_pis(vec![&air, &range_checker.air], vec![trace, range_trace])
        .expect("Verification failed");
}

#[test]
#[should_panic]
fn test_x_div_y_wrong_trace() {
    let prime = secp256k1_prime();
    let (x, y) = generate_xy();

    let (arithmetic, range_checker) =
        get_air_and_range_checker(prime.clone(), LIMB_BITS, NUM_LIMB, true);
    let air = ModularDivisionAir { arithmetic };
    let cols = air.generate_trace_row((x.clone(), y.clone(), range_checker.clone()));
    let row = cols.flatten();
    let mut trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&air));
    trace.row_mut(0)[0] += BabyBear::one();
    let range_trace = range_checker.generate_trace();

    run_simple_test_no_pis(vec![&air, &range_checker.air], vec![trace, range_trace])
        .expect("Verification failed");
}
*/

#[test]
fn test_x_add_y() {
    let prime = secp256k1_prime();
    let (x, y) = generate_xy();

    let (arithmetic, range_checker) =
        get_air_and_range_checker(prime.clone(), LIMB_BITS, NUM_LIMB, false);
    let air = ModularAdditionAir { arithmetic };
    let expected_r = (x.clone() + y.clone()) % prime.clone();
    let expected_q = (x.clone() + y.clone() - expected_r.clone()) / prime;
    let cols = air.generate_trace_row((x, y, range_checker.clone()));
    let ModularArithmeticCols { q, r, .. } = cols.clone();
    let generated_r = evaluate_bigint(&r, LIMB_BITS);
    let generated_q = evaluate_bigint(&q, LIMB_BITS);
    assert_eq!(generated_r, expected_r);
    assert_eq!(generated_q, expected_q);

    let row = cols.flatten();
    let trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&air));
    let range_trace = range_checker.generate_trace();

    run_simple_test_no_pis(vec![&air, &range_checker.air], vec![trace, range_trace])
        .expect("Verification failed");
}

#[test]
#[should_panic]
fn test_x_add_y_wrong_trace() {
    let prime = secp256k1_prime();
    let (x, y) = generate_xy();

    let (arithmetic, range_checker) =
        get_air_and_range_checker(prime.clone(), LIMB_BITS, NUM_LIMB, false);
    let air = ModularAdditionAir { arithmetic };
    let cols = air.generate_trace_row((x, y, range_checker.clone()));

    let row = cols.flatten();
    let mut trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&air));
    trace.row_mut(0)[0] += BabyBear::one();
    let range_trace = range_checker.generate_trace();

    run_simple_test_no_pis(vec![&air, &range_checker.air], vec![trace, range_trace])
        .expect("Verification failed");
}

#[test]
fn test_x_sub_y() {
    let prime = secp256k1_prime();
    let (x, y) = generate_xy();

    let (arithmetic, range_checker) =
        get_air_and_range_checker(prime.clone(), LIMB_BITS, NUM_LIMB, false);
    let air = ModularSubtractionAir { arithmetic };
    let mut xp = x.clone();
    while xp < y {
        xp += prime.clone();
    }
    let expected_r = xp - y.clone();
    let expected_q = BigUint::zero();
    let cols = air.generate_trace_row((x.clone(), y.clone(), range_checker.clone()));
    let ModularArithmeticCols {
        x: _x,
        y: _y,
        q,
        r,
        carries: _carries,
    } = cols.clone();
    let generated_r = evaluate_bigint(&r, LIMB_BITS);
    let generated_q = evaluate_bigint(&q, LIMB_BITS);
    assert_eq!(generated_r, expected_r);
    assert_eq!(generated_q, expected_q);

    let row = cols.flatten();
    let trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&air));
    let range_trace = range_checker.generate_trace();

    run_simple_test_no_pis(vec![&air, &range_checker.air], vec![trace, range_trace])
        .expect("Verification failed");
}

#[test]
fn test_x_sub_bigger_y() {
    let prime = secp256k1_prime();
    // x > y from the fixed randomness, so swap x and y.
    let (y, x) = generate_xy();

    let (arithmetic, range_checker) =
        get_air_and_range_checker(prime.clone(), LIMB_BITS, NUM_LIMB, false);
    let air = ModularSubtractionAir { arithmetic };

    let cols = air.generate_trace_row((x.clone(), y.clone(), range_checker.clone()));
    let row = cols.flatten();
    let trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&air));
    let range_trace = range_checker.generate_trace();

    run_simple_test_no_pis(vec![&air, &range_checker.air], vec![trace, range_trace])
        .expect("Verification failed");
}

#[test]
#[should_panic]
fn test_x_sub_y_wrong_trace() {
    let prime = secp256k1_prime();
    let (x, y) = generate_xy();

    let (arithmetic, range_checker) =
        get_air_and_range_checker(prime.clone(), LIMB_BITS, NUM_LIMB, false);
    let air = ModularSubtractionAir { arithmetic };
    let cols = air.generate_trace_row((x, y, range_checker.clone()));

    let row = cols.flatten();
    let mut trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&air));
    trace.row_mut(0)[0] += BabyBear::one();
    let range_trace = range_checker.generate_trace();

    run_simple_test_no_pis(vec![&air, &range_checker.air], vec![trace, range_trace])
        .expect("Verification failed");
}
