use core::PublicValuesCoreStep;

use crate::system::native_adapter::{NativeAdapterAir, NativeAdapterStep};

mod columns;
/// Chip to publish custom public values from VM programs.
pub mod core;

#[cfg(test)]
mod tests;

pub type PublicValuesChip<F> =
    PublicValuesCoreStep<NativeAdapterAir<2, 0>, NativeAdapterStep<F, 2, 0>, F>;
