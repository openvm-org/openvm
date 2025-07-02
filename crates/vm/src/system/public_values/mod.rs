use crate::{
    arch::VmAirWrapper,
    system::{native_adapter::NativeAdapterAir, public_values::core::PublicValuesCoreAir},
};

mod columns;
/// Chip to publish custom public values from VM programs.
pub mod core;

#[cfg(test)]
mod tests;

pub type PublicValuesAir = VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir>;
