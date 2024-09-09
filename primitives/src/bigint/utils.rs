use std::{collections::VecDeque, ops::Neg};

use afs_stark_backend::interaction::InteractionBuilder;
use num_bigint_dig::{BigInt, BigUint, Sign};
use num_traits::{One, Zero};
use p3_field::AbstractField;

use crate::var_range::bus::VariableRangeCheckerBus;

// Checks that the given expression is within bits number of bits.
pub fn range_check<AB: InteractionBuilder>(
    builder: &mut AB,
    range_bus: usize, // The bus number for range checker.
    decomp: usize,    // The ranger checker checks the numbers are within decomp bits.
    bits: usize,
    into_expr: impl Into<AB::Expr>,
) {
    assert!(bits <= decomp);
    let expr = into_expr.into();
    let bus = VariableRangeCheckerBus::new(range_bus, decomp);
    bus.range_check(expr, bits).eval(builder, AB::F::one());
}

pub fn secp256k1_prime() -> BigUint {
    let mut result = BigUint::one() << 256;
    for power in [32, 9, 8, 7, 6, 4, 0] {
        result -= BigUint::one() << power;
    }
    result
}

pub fn big_int_abs(x: BigInt) -> BigUint {
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

// Convert a big uint bits by first conerting to bytes (little endian).
// So the number of bits is multiple of 8.
pub fn big_uint_to_bits(x: BigUint) -> VecDeque<usize> {
    let mut result = VecDeque::new();
    for byte in x.to_bytes_le() {
        for i in 0..8 {
            result.push_back(((byte >> i) as usize) & 1);
        }
    }
    result
}

pub fn big_uint_to_limbs(x: BigUint, limb_bits: usize) -> Vec<usize> {
    let total_limbs = (x.bits() + limb_bits - 1) / limb_bits;
    let mut modulus_bits = big_uint_to_bits(x);

    (0..total_limbs)
        .map(|_| take_limb(&mut modulus_bits, limb_bits))
        .collect()
}

pub fn big_int_to_limbs(x: BigInt, limb_bits: usize) -> Vec<isize> {
    let x_sign = x.sign();
    let limbs = big_uint_to_limbs(big_int_abs(x), limb_bits);
    if x_sign == Sign::Minus {
        limbs.iter().map(|&x| -(x as isize)).collect()
    } else {
        limbs.iter().map(|&x| x as isize).collect()
    }
}

pub fn take_limb(deque: &mut VecDeque<usize>, limb_size: usize) -> usize {
    deque
        .drain(..limb_size.min(deque.len()))
        .enumerate()
        .map(|(i, bit)| bit << i)
        .sum()
}
