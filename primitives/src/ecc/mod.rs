use num_bigint_dig::BigUint;

use crate::bigint::{
    check_carry_mod_to_zero::CheckCarryModToZeroSubAir, CanonicalUint, LimbConfig,
};

pub mod air;
pub mod columns;
pub mod trace;

#[cfg(test)]
mod tests;

pub struct EcPoint<T, C: LimbConfig> {
    pub x: CanonicalUint<T, C>,
    pub y: CanonicalUint<T, C>,
}

pub struct EccAir {
    // e.g. secp256k1 is 2^256 - 2^32 - 977.
    pub prime: BigUint,

    // y^2 = x^3 + b. b=7 for secp256k1.
    pub b: BigUint,

    // The limb config for the EcPoint coordinates.
    pub limb_bits: usize,
    // Number of limbs of the prime and the coordinates.
    pub num_limbs: usize,
    // pub limb_config: C,

    // The max overflow bits, see OverflowInt for details.
    // pub max_overflow_bits: usize,

    // The subair to constrain big integer operations.
    pub check_carry: CheckCarryModToZeroSubAir,
    // For range checker interactions.
    // pub range_checker_bus: usize,
    // Range checker decomp bits.
    pub decomp: usize,
    // lambda_carries_bits: usize ?? I think each carry bit is different?
    // If carry bits are different for each equation, we need different subair...
}

impl EccAir {
    pub fn new(
        prime: BigUint,
        b: BigUint,
        range_checker_bus: usize,
        decomp: usize,
        limb_bits: usize,
        field_element_bits: usize,
    ) -> Self {
        let num_limbs = (prime.bits() + limb_bits - 1) / limb_bits;
        let check_carry = CheckCarryModToZeroSubAir::new(
            prime.clone(),
            limb_bits,
            range_checker_bus,
            decomp,
            field_element_bits,
        );

        EccAir {
            prime,
            b,
            limb_bits,
            num_limbs,
            check_carry,
            decomp,
        }
    }
}
