use num_bigint_dig::{BigInt, BigUint, Sign};
use p3_field::PrimeField64;

use super::EcModularConfig;
use crate::bigint::{utils::big_int_to_num_limbs, CanonicalUint, DefaultLimbConfig, OverflowInt};

pub(super) fn vec_isize_to_f<F: PrimeField64>(x: Vec<isize>) -> Vec<F> {
    x.iter()
        .map(|x| {
            F::from_canonical_usize(x.unsigned_abs())
                * if x >= &0 { F::one() } else { F::neg_one() }
        })
        .collect()
}

pub(super) fn to_canonical(
    x: &BigUint,
    num_limbs: usize,
) -> CanonicalUint<isize, DefaultLimbConfig> {
    CanonicalUint::<isize, DefaultLimbConfig>::from_big_uint(x, Some(num_limbs))
}

pub(super) fn to_canonical_f<F: PrimeField64>(
    x: &BigUint,
    num_limbs: usize,
) -> CanonicalUint<F, DefaultLimbConfig> {
    let limbs = vec_isize_to_f(to_canonical(x, num_limbs).limbs);
    CanonicalUint::<F, DefaultLimbConfig>::from_vec(limbs)
}

pub(super) fn to_overflow_int(x: &BigUint, num_limbs: usize) -> OverflowInt<isize> {
    OverflowInt::<isize>::from(to_canonical(x, num_limbs))
}

pub(super) fn to_overflow_q(q_limbs: Vec<isize>, limb_bits: usize) -> OverflowInt<isize> {
    OverflowInt {
        limbs: q_limbs,
        max_overflow_bits: limb_bits + 1,
        limb_max_abs: (1 << limb_bits),
    }
}

pub(super) fn compute_lambda_q_limbs(
    config: &EcModularConfig,
    x1: &BigUint,
    x2: &BigUint,
    y1: &BigUint,
    y2: &BigUint,
    lambda: &BigUint,
) -> Vec<isize> {
    // λ * (x2 - x1) - y2 + y1 = q * p
    let lambda_signed = BigInt::from_biguint(Sign::Plus, lambda.clone());
    let x1_signed = BigInt::from_biguint(Sign::Plus, x1.clone());
    let x2_signed = BigInt::from_biguint(Sign::Plus, x2.clone());
    let y1_signed = BigInt::from_biguint(Sign::Plus, y1.clone());
    let y2_signed = BigInt::from_biguint(Sign::Plus, y2.clone());
    let prime_signed = BigInt::from_biguint(Sign::Plus, config.prime.clone());
    let lambda_q_signed: BigInt =
        (lambda_signed * (x2_signed - x1_signed) - y2_signed + y1_signed) / prime_signed;
    big_int_to_num_limbs(lambda_q_signed, config.limb_bits, config.num_limbs)
}

pub(super) fn compute_lambda_carries(
    config: &EcModularConfig,
    x1: &BigUint,
    x2: &BigUint,
    y1: &BigUint,
    y2: &BigUint,
    lambda: &BigUint,
    lambda_q_limbs: Vec<isize>,
) -> (Vec<isize>, usize) {
    // λ * (x2 - x1) - y2 + y1 - q * p
    let lambda_overflow = to_overflow_int(lambda, config.num_limbs);
    let x1_overflow = to_overflow_int(x1, config.num_limbs);
    let x2_overflow = to_overflow_int(x2, config.num_limbs);
    let y1_overflow = to_overflow_int(y1, config.num_limbs);
    let y2_overflow = to_overflow_int(y2, config.num_limbs);
    let lambda_q_overflow = to_overflow_q(lambda_q_limbs, config.limb_bits);
    let prime_overflow = to_overflow_int(&config.prime, config.num_limbs);
    let expr = lambda_overflow * (x2_overflow - x1_overflow) - y2_overflow + y1_overflow;
    let expr = expr - lambda_q_overflow * prime_overflow;
    (
        expr.calculate_carries(config.limb_bits),
        expr.max_overflow_bits,
    )
}

pub(super) fn compute_x3_q_limbs(
    config: &EcModularConfig,
    x1: &BigUint,
    x2: &BigUint,
    x3: &BigUint,
    lambda: &BigUint,
) -> Vec<isize> {
    // λ * λ - x1 - x2 - x3 = x3_q * p
    let lambda_signed = BigInt::from_biguint(Sign::Plus, lambda.clone());
    let x1_signed = BigInt::from_biguint(Sign::Plus, x1.clone());
    let x2_signed = BigInt::from_biguint(Sign::Plus, x2.clone());
    let x3_signed = BigInt::from_biguint(Sign::Plus, x3.clone());
    let prime_signed = BigInt::from_biguint(Sign::Plus, config.prime.clone());
    let x3_q_signed =
        (lambda_signed.clone() * lambda_signed - x1_signed - x2_signed - x3_signed) / prime_signed;
    big_int_to_num_limbs(x3_q_signed, config.limb_bits, config.num_limbs)
}

pub(super) fn compute_x3_carries(
    config: &EcModularConfig,
    x1: &BigUint,
    x2: &BigUint,
    x3: &BigUint,
    lambda: &BigUint,
    x3_q_limbs: Vec<isize>,
) -> (Vec<isize>, usize) {
    // λ * λ - x1 - x2 - x3 - x3_q * p
    let lambda_overflow = to_overflow_int(lambda, config.num_limbs);
    let x1_overflow = to_overflow_int(x1, config.num_limbs);
    let x2_overflow = to_overflow_int(x2, config.num_limbs);
    let x3_overflow = to_overflow_int(x3, config.num_limbs);
    let x3_q_overflow = to_overflow_q(x3_q_limbs, config.limb_bits);
    let prime_overflow = to_overflow_int(&config.prime, config.num_limbs);
    let expr = lambda_overflow.clone() * lambda_overflow - x1_overflow - x2_overflow - x3_overflow;
    let expr = expr - x3_q_overflow * prime_overflow;
    (
        expr.calculate_carries(config.limb_bits),
        expr.max_overflow_bits,
    )
}

pub(super) fn compute_y3_q_limbs(
    config: &EcModularConfig,
    x1: &BigUint,
    x3: &BigUint,
    y1: &BigUint,
    y3: &BigUint,
    lambda: &BigUint,
) -> Vec<isize> {
    // y3 + λ * x3 + y1 - λ * x1 = q * p
    let lambda_signed = BigInt::from_biguint(Sign::Plus, lambda.clone());
    let x1_signed = BigInt::from_biguint(Sign::Plus, x1.clone());
    let x3_signed = BigInt::from_biguint(Sign::Plus, x3.clone());
    let y1_signed = BigInt::from_biguint(Sign::Plus, y1.clone());
    let y3_signed = BigInt::from_biguint(Sign::Plus, y3.clone());
    let prime_signed = BigInt::from_biguint(Sign::Plus, config.prime.clone());
    let y3_q_signed = (y3_signed + lambda_signed.clone() * x3_signed + y1_signed
        - lambda_signed * x1_signed)
        / prime_signed;
    big_int_to_num_limbs(y3_q_signed, config.limb_bits, config.num_limbs)
}

pub(super) fn compute_y3_carries(
    config: &EcModularConfig,
    x1: &BigUint,
    x3: &BigUint,
    y1: &BigUint,
    y3: &BigUint,
    lambda: &BigUint,
    y3_q_limbs: Vec<isize>,
) -> (Vec<isize>, usize) {
    // y3 + λ * x3 + y1 - λ * x1 - q * p
    let lambda_overflow = to_overflow_int(lambda, config.num_limbs);
    let x1_overflow = to_overflow_int(x1, config.num_limbs);
    let x3_overflow = to_overflow_int(x3, config.num_limbs);
    let y1_overflow = to_overflow_int(y1, config.num_limbs);
    let y3_overflow = to_overflow_int(y3, config.num_limbs);
    let y3_q_overflow = to_overflow_q(y3_q_limbs, config.limb_bits);
    let prime_overflow = to_overflow_int(&config.prime, config.num_limbs);
    let expr = y3_overflow + lambda_overflow.clone() * x3_overflow + y1_overflow
        - lambda_overflow * x1_overflow;
    let expr = expr - y3_q_overflow * prime_overflow;
    (
        expr.calculate_carries(config.limb_bits),
        expr.max_overflow_bits,
    )
}
