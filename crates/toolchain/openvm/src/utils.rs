#[cfg(not(openvm_intrinsics))]
use num_bigint::BigUint;

#[inline]
#[cfg(not(openvm_intrinsics))]
#[allow(dead_code)]
/// Convert a `BigUint` to a `[u8; NUM_LIMBS]` in little-endian format.
pub fn biguint_to_limbs<const NUM_LIMBS: usize>(x: &BigUint) -> [u8; NUM_LIMBS] {
    let mut sm = x.to_bytes_le();
    sm.resize(NUM_LIMBS, 0);
    sm.try_into().unwrap()
}
