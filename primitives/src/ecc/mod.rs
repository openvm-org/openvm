use num_bigint_dig::BigUint;

use crate::bigint::{CanonicalUint, LimbConfig};

mod air;
mod columns;
pub mod trace;
mod utils;

pub use air::*;
pub use columns::*;

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
