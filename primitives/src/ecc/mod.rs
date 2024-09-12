use num_bigint_dig::BigUint;

use crate::bigint::{CanonicalUint, LimbConfig};

pub mod air;
pub mod columns;
pub mod trace;
pub mod utils;

#[cfg(test)]
mod tests;

#[derive(Clone)]
pub struct EcPoint<T, C: LimbConfig> {
    pub x: CanonicalUint<T, C>,
    pub y: CanonicalUint<T, C>,
}

pub struct EcModularConfig {
    pub prime: BigUint,
    pub num_limbs: usize,
    pub limb_bits: usize,
}
