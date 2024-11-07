#[cfg(not(target_os = "zkvm"))]
use num_bigint_dig::{traits::ModInverse, BigUint, Sign, ToBigInt};

#[inline]
#[cfg(not(target_os = "zkvm"))]
#[allow(dead_code)]
/// Convert a `BigUint` to a `[u8; NUM_LIMBS]`.
pub fn biguint_to_limbs<const NUM_LIMBS: usize>(x: &BigUint) -> [u8; NUM_LIMBS] {
    let mut sm = x.to_bytes_le();
    sm.resize(NUM_LIMBS, 0);
    sm.try_into().unwrap()
}

#[inline]
#[cfg(not(target_os = "zkvm"))]
#[allow(dead_code)]
/// Find the modular inverse of BigUint 'x'
pub fn uint_mod_inverse(x: &BigUint, modulus: &BigUint) -> BigUint {
    let signed_inv = x.mod_inverse(modulus).unwrap();
    if signed_inv.sign() == Sign::Minus {
        modulus.to_bigint().unwrap() + signed_inv
    } else {
        signed_inv
    }
    .to_biguint()
    .unwrap()
}
