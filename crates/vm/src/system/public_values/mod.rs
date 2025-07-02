use crate::{arch::VmAirWrapper, system::native_adapter::NativeAdapterAir};

mod columns;
/// Chip to publish custom public values from VM programs.
mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type PublicValuesAir = VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir>;
