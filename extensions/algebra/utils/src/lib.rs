//! Shared number-theory helpers for the OpenVM algebra extension.

use num_bigint::BigUint;
use num_traits::{FromPrimitive, One};
use rand::RngCore;

/// Returns a non-quadratic residue in the field of order `modulus`.
///
/// Uses fixed shortcuts when `modulus % 4 == 3` (return `-1`) or
/// `modulus % 8 == 5` (return `2`), otherwise rejection-samples uniformly
/// over `[2, modulus - 1)` and tests with the Euler criterion.
pub fn find_non_qr(modulus: &BigUint, rng: &mut impl RngCore) -> BigUint {
    if modulus % 4u32 == BigUint::from(3u8) {
        // p = 3 mod 4 then -1 is a non-quadratic residue
        modulus - BigUint::one()
    } else if modulus % 8u32 == BigUint::from(5u8) {
        // p = 5 mod 8 then 2 is a non-quadratic residue
        // since 2^((p-1)/2) = (-1)^((p^2-1)/8)
        BigUint::from_u8(2u8).unwrap()
    } else {
        // Sample uniformly from [2, modulus - 1) using rejection sampling
        let range = modulus - 3u32; // number of values in [2, modulus-1)
        let mut buf = vec![0u8; modulus.to_bytes_be().len()];
        let exponent = (modulus - BigUint::one()) >> 1;
        loop {
            // Rejection sample for uniform distribution
            rng.fill_bytes(&mut buf);
            let val = BigUint::from_bytes_be(&buf);
            if val >= range {
                continue;
            }
            let non_qr = val + 2u32;
            // To check if non_qr is a quadratic nonresidue, we compute non_qr^((p-1)/2)
            // If the result is p-1, then non_qr is a quadratic nonresidue
            if non_qr.modpow(&exponent, modulus) == modulus - BigUint::one() {
                return non_qr;
            }
        }
    }
}
