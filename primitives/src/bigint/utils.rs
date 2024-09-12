use std::{borrow::Cow, collections::VecDeque, iter::repeat, ops::Neg, str::FromStr};

use afs_stark_backend::interaction::InteractionBuilder;
use num_bigint_dig::{algorithms::mod_inverse, BigInt, BigUint, Sign};
use num_traits::{One, Zero};

use super::modular_arithmetic::ModularArithmeticAir;
use crate::var_range::bus::VariableRangeCheckerBus;

// Checks that the given expression is within bits number of bits.
pub fn range_check<AB: InteractionBuilder>(
    builder: &mut AB,
    range_bus: usize, // The bus number for range checker.
    decomp: usize,    // The ranger checker checks the numbers are within decomp bits.
    bits: usize,
    into_expr: impl Into<AB::Expr>,
    count: impl Into<AB::Expr>,
) {
    assert!(bits <= decomp);
    let expr = into_expr.into();
    let bus = VariableRangeCheckerBus::new(range_bus, decomp);
    bus.range_check(expr, bits).eval(builder, count);
}

pub fn secp256k1_prime() -> BigUint {
    let mut result = BigUint::one() << 256;
    for power in [32, 9, 8, 7, 6, 4, 0] {
        result -= BigUint::one() << power;
    }
    result
}

pub fn secp256k1_coord_prime() -> BigUint {
    let mut result = BigUint::one() << 256;
    for power in [32, 9, 8, 7, 6, 4, 0] {
        result -= BigUint::one() << power;
    }
    result
}

pub fn secp256k1_scalar_prime() -> BigUint {
    BigUint::from_str(
        "115792089237316195423570985008687907852837564279074904382605163141518161494337",
    )
    .unwrap()
}

pub fn get_arithmetic_air(
    prime: BigUint,
    limb_bits: usize,
    field_element_bits: usize,
    num_limbs: usize,
    is_mul_div: bool,
    range_bus: usize,
    range_decomp: usize,
) -> ModularArithmeticAir {
    let q_limbs = if is_mul_div { num_limbs } else { 1 };
    let carry_limbs = if is_mul_div {
        2 * num_limbs - 1
    } else {
        num_limbs
    };
    ModularArithmeticAir::new(
        prime,
        limb_bits,
        field_element_bits,
        num_limbs,
        q_limbs,
        carry_limbs,
        range_bus,
        range_decomp,
    )
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

pub fn big_uint_mod_inverse(x: &BigUint, modulus: &BigUint) -> BigUint {
    mod_inverse(Cow::Borrowed(x), Cow::Borrowed(modulus))
        .unwrap()
        .to_biguint()
        .unwrap()
}

// Convert a big uint bits by first conerting to bytes (little endian).
// So the number of bits is multiple of 8.
pub fn big_uint_to_bits(x: &BigUint) -> VecDeque<usize> {
    let mut result = VecDeque::new();
    for byte in x.to_bytes_le() {
        for i in 0..8 {
            result.push_back(((byte >> i) as usize) & 1);
        }
    }
    result
}

pub fn big_uint_to_limbs(x: &BigUint, limb_bits: usize) -> Vec<usize> {
    let total_limbs = (x.bits() + limb_bits - 1) / limb_bits;
    let mut modulus_bits = big_uint_to_bits(x);

    (0..total_limbs)
        .map(|_| take_limb(&mut modulus_bits, limb_bits))
        .collect()
}

pub fn big_uint_to_num_limbs(x: BigUint, limb_bits: usize, num_limbs: usize) -> Vec<usize> {
    let limbs = big_uint_to_limbs(&x, limb_bits);
    limbs
        .iter()
        .chain(repeat(&0))
        .take(num_limbs)
        .copied()
        .collect()
}

pub fn big_int_to_limbs(x: BigInt, limb_bits: usize) -> Vec<isize> {
    let x_sign = x.sign();
    let limbs = big_uint_to_limbs(&big_int_abs(x), limb_bits);
    if x_sign == Sign::Minus {
        limbs.iter().map(|&x| -(x as isize)).collect()
    } else {
        limbs.iter().map(|&x| x as isize).collect()
    }
}

pub fn big_int_to_num_limbs(x: BigInt, limb_bits: usize, num_limbs: usize) -> Vec<isize> {
    let limbs = big_int_to_limbs(x, limb_bits);
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
