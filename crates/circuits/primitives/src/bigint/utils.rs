use std::{cmp::max, collections::VecDeque, iter::repeat, ops::Neg, str::FromStr};

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{FromPrimitive, Num, ToPrimitive, Zero};
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder},
    p3_field::PrimeField64,
};

use crate::var_range::VariableRangeCheckerBus;

// Checks that the given expression is within bits number of bits.
pub fn range_check<AB: InteractionBuilder>(
    builder: &mut AB,
    range_bus: BusIndex, // The bus number for range checker.
    decomp: usize,       // The ranger checker checks the numbers are within decomp bits.
    bits: usize,
    into_expr: impl Into<AB::Expr>,
    count: impl Into<AB::Expr>,
) {
    assert!(bits <= decomp, "range_check: bits {bits} > decomp {decomp}");
    let expr = into_expr.into();
    let bus = VariableRangeCheckerBus::new(range_bus, decomp);
    bus.range_check(expr, bits).eval(builder, count);
}

pub fn secp256k1_coord_prime() -> BigUint {
    BigUint::from_str_radix(
        "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f",
        16,
    )
    .unwrap()
}

pub fn secp256k1_scalar_prime() -> BigUint {
    BigUint::from_str(
        "115792089237316195423570985008687907852837564279074904382605163141518161494337",
    )
    .unwrap()
}

// aka P256
pub fn secp256r1_coord_prime() -> BigUint {
    BigUint::from_str_radix(
        "ffffffff00000001000000000000000000000000ffffffffffffffffffffffff",
        16,
    )
    .unwrap()
}

pub fn secp256r1_scalar_prime() -> BigUint {
    BigUint::from_str_radix(
        "ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551",
        16,
    )
    .unwrap()
}

pub fn big_int_abs(x: &BigInt) -> BigUint {
    if x.sign() == Sign::Minus {
        x.neg().to_biguint().unwrap()
    } else {
        x.to_biguint().unwrap()
    }
}

pub fn big_uint_sub(x: BigUint, y: BigUint) -> BigInt {
    match x.cmp(&y) {
        std::cmp::Ordering::Less => BigInt::from_biguint(Sign::Minus, y - x),
        std::cmp::Ordering::Equal => BigInt::zero(),
        std::cmp::Ordering::Greater => BigInt::from_biguint(Sign::Plus, x - y),
    }
}

// little endian.
pub fn big_uint_to_limbs(x: &BigUint, limb_bits: usize) -> Vec<usize> {
    let mut result = Vec::new();
    let mut x = x.clone();
    let base = BigUint::from_u32(1 << limb_bits).unwrap();
    while x > BigUint::zero() {
        result.push((x.clone() % &base).to_usize().unwrap());
        x /= &base;
    }
    result
}

pub fn big_uint_to_num_limbs(x: &BigUint, limb_bits: usize, num_limbs: usize) -> Vec<usize> {
    let limbs = big_uint_to_limbs(x, limb_bits);
    let num_limbs = max(num_limbs, limbs.len());
    limbs
        .iter()
        .chain(repeat(&0))
        .take(num_limbs)
        .copied()
        .collect()
}

pub fn big_int_to_limbs(x: &BigInt, limb_bits: usize) -> Vec<isize> {
    let x_sign = x.sign();
    let limbs = big_uint_to_limbs(&big_int_abs(x), limb_bits);
    if x_sign == Sign::Minus {
        limbs.iter().map(|&x| -(x as isize)).collect()
    } else {
        limbs.iter().map(|&x| x as isize).collect()
    }
}

pub fn big_int_to_num_limbs(x: &BigInt, limb_bits: usize, num_limbs: usize) -> Vec<isize> {
    let limbs = big_int_to_limbs(x, limb_bits);
    let num_limbs = max(num_limbs, limbs.len());
    limbs
        .iter()
        .chain(repeat(&0))
        .take(num_limbs)
        .copied()
        .collect()
}

pub fn take_limb(deque: &mut VecDeque<usize>, limb_size: usize) -> usize {
    deque
        .drain(..limb_size.min(deque.len()))
        .enumerate()
        .map(|(i, bit)| bit << i)
        .sum()
}

pub fn vec_isize_to_f<F: PrimeField64>(x: Vec<isize>) -> Vec<F> {
    x.iter()
        .map(|x| {
            F::from_canonical_usize(x.unsigned_abs()) * if x >= &0 { F::ONE } else { F::NEG_ONE }
        })
        .collect()
}
