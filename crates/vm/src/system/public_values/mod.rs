use core::PublicValuesStep;

use crate::{
    arch::{NewVmChipWrapper, VmAirWrapper},
    system::{
        native_adapter::{NativeAdapterAir, NativeAdapterStep},
        public_values::core::PublicValuesCoreAir,
    },
};

mod columns;
/// Chip to publish custom public values from VM programs.
pub mod core;

#[cfg(test)]
mod tests;

pub type PublicValuesAir = VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir>;
pub type PublicValuesStepWithAdapter<F> = PublicValuesStep<NativeAdapterStep<F, 2, 0>, F>;
pub type PublicValuesChip<F> = NewVmChipWrapper<F, PublicValuesAir, PublicValuesStepWithAdapter<F>>;
