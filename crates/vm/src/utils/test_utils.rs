use std::array;

use ax_circuit_primitives::bigint::utils::big_uint_to_limbs;
use num_bigint_dig::BigUint;
use num_traits::{FromPrimitive, ToPrimitive, Zero};
use p3_field::PrimeField32;
use rand::{rngs::StdRng, Rng};

pub fn i32_to_f<F: PrimeField32>(val: i32) -> F {
    if val.signum() == -1 {
        -F::from_canonical_u32(val.unsigned_abs())
    } else {
        F::from_canonical_u32(val as u32)
    }
}

// little endian.
// Warning: This function only returns the last NUM_LIMBS*LIMB_BITS bits of
//          the input, while the input can have more than that.
pub fn biguint_to_limbs<const NUM_LIMBS: usize>(
    mut x: BigUint,
    limb_size: usize,
) -> [u32; NUM_LIMBS] {
    let mut result = [0; NUM_LIMBS];
    let base = BigUint::from_u32(1 << limb_size).unwrap();
    for r in result.iter_mut() {
        *r = (x.clone() % &base).to_u32().unwrap();
        x /= &base;
    }
    assert!(x.is_zero());
    result
}

pub fn generate_long_number<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    rng: &mut StdRng,
) -> [u32; NUM_LIMBS] {
    array::from_fn(|_| rng.gen_range(0..(1 << LIMB_BITS)))
}

pub fn generate_field_element<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    modulus: &BigUint,
    rng: &mut StdRng,
) -> [u32; NUM_LIMBS] {
    let x = generate_long_number::<NUM_LIMBS, LIMB_BITS>(rng);
    let bigint = BigUint::new(x.to_vec()) % modulus;
    let vec = big_uint_to_limbs(&bigint, LIMB_BITS);
    array::from_fn(|i| if i < vec.len() { vec[i] as u32 } else { 0 })
}
// in little endian
pub fn u32_into_limbs<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    num: u32,
) -> [u32; NUM_LIMBS] {
    array::from_fn(|i| (num >> (LIMB_BITS * i)) & ((1 << LIMB_BITS) - 1))
}

pub fn u32_sign_extend<const IMM_BITS: usize>(num: u32) -> u32 {
    if num & (1 << (IMM_BITS - 1)) != 0 {
        num | (u32::MAX - (1 << IMM_BITS) + 1)
    } else {
        num
    }
}
