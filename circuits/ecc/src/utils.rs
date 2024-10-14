use num_bigint_dig::BigUint;
use num_traits::{FromPrimitive, Zero};
use p3_baby_bear::BabyBear;
use p3_field::PrimeField64;

pub fn evaluate_biguint(limbs: &[BabyBear], limb_bits: usize) -> BigUint {
    let mut res = BigUint::zero();
    let base = BigUint::from_u64(1 << limb_bits).unwrap();
    for limb in limbs.iter().rev() {
        res = res * base.clone() + BigUint::from_u64(limb.as_canonical_u64()).unwrap();
    }
    res
}
