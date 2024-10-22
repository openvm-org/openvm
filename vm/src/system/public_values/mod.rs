use crate::{
    arch::{VmAirWrapper, VmChipWrapper},
    kernels::adapters::native_adapter::{NativeAdapterAir, NativeAdapterChip},
    system::public_values::core::{PublicValuesCoreAir, PublicValuesCoreChip},
};

mod columns;
/// Chip to publish custom public values from VM programs.
mod core;
#[cfg(test)]
mod tests;

pub type PublicValuesAir = VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir>;
pub type PublicValuesChip<F> =
    VmChipWrapper<F, NativeAdapterChip<F, 2, 0>, PublicValuesCoreChip<F>>;
