#[cfg(not(target_os = "zkvm"))]
use num_bigint_dig::BigUint;

#[inline]
#[cfg(not(target_os = "zkvm"))]
pub(super) fn biguint_to_limbs<const NUM_LIMBS: usize>(x: &BigUint) -> [u8; NUM_LIMBS] {
    let mut sm = x.to_bytes_le();
    sm.resize(NUM_LIMBS, 0);
    sm.try_into().unwrap()
}
