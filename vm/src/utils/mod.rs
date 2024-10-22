use num_bigint_dig::BigUint;
use num_traits::{FromPrimitive, ToPrimitive, Zero};

// little endian.
pub fn limbs_to_biguint(x: &[u32], limb_size: usize) -> BigUint {
    let mut result = BigUint::zero();
    let base = BigUint::from_u32(1 << limb_size).unwrap();
    for limb in x.iter().rev() {
        result = result * &base + BigUint::from_u32(*limb).unwrap();
    }
    result
}

#[allow(dead_code)]
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

// Use this when num_limbs is not a constant.
// little endian.
// Warning: This function only returns the last NUM_LIMBS*LIMB_SIZE bits of
//          the input, while the input can have more than that.
pub fn biguint_to_limbs_vec(mut x: BigUint, limb_size: usize, num_limbs: usize) -> Vec<u32> {
    let mut result = vec![0; num_limbs];
    let base = BigUint::from_u32(1 << limb_size).unwrap();
    for r in result.iter_mut() {
        *r = (x.clone() % &base).to_u32().unwrap();
        x /= &base;
    }
    assert!(x.is_zero());
    result
}
