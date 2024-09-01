use std::collections::VecDeque;

use afs_stark_backend::interaction::InteractionBuilder;
use num_bigint_dig::BigUint;
use num_traits::One;
use p3_field::AbstractField;

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
    if bits == decomp {
        builder.push_send(range_bus, [expr], AB::F::one());
    } else {
        builder.push_send(range_bus, [expr.clone()], AB::F::one());
        builder.push_send(
            range_bus,
            [expr + AB::F::from_canonical_usize((1 << decomp) - (1 << bits))],
            AB::F::one(),
        );
    }
}

pub fn secp256k1_prime() -> BigUint {
    let mut result = BigUint::one() << 256;
    for power in [32, 9, 8, 7, 6, 4, 0] {
        result -= BigUint::one() << power;
    }
    result
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

pub fn take_limb(deque: &mut VecDeque<usize>, limb_size: usize) -> usize {
    deque
        .drain(..limb_size.min(deque.len()))
        .enumerate()
        .map(|(i, bit)| bit << i)
        .sum()
}
