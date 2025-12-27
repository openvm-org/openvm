use alloc::vec::Vec;

use num_bigint::BigUint;
use num_traits::{One, ToPrimitive, Zero};

/// Convert a positive integer into a Non-Adjacent Form (NAF) digit vector (LSB-first).
pub fn biguint_to_naf(n: &BigUint) -> Vec<i8> {
    if n.is_zero() {
        return Vec::new();
    }

    let mut k = n.clone();
    let one = BigUint::one();
    let three = &one + &one + &one;
    let mut naf = Vec::new();

    while !k.is_zero() {
        if (&k & &one) == one {
            let rem = (&k & &three).to_u8().unwrap();
            let digit = 2i8 - rem as i8;
            naf.push(if digit == 2 { -1 } else { digit });
            if *naf.last().unwrap() > 0 {
                k -= &one;
            } else {
                k += &one;
            }
        } else {
            naf.push(0);
        }
        k >>= 1usize;
    }
    naf
}
