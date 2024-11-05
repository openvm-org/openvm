#[cfg(not(target_os = "zkvm"))]
use num_bigint_dig::BigUint;

#[inline]
#[cfg(not(target_os = "zkvm"))]
pub(super) fn biguint_to_limbs<const NUM_LIMBS: usize>(x: BigUint) -> [u8; NUM_LIMBS] {
    x.to_bytes_le().try_into().unwrap()
}
