use crate::bigint::{CanonicalUint, LimbConfig};

pub mod air;
pub mod columns;
pub mod trace;

#[cfg(test)]
mod tests;

pub struct EcPoint<T, C: LimbConfig> {
    pub x: CanonicalUint<T, C>,
    pub y: CanonicalUint<T, C>,
}
