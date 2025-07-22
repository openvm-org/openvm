use std::sync::Arc;

use openvm_stark_backend::{
    config::StarkGenericConfig, prover::hal::ProverBackend, AirRef, ChipUsageGetter,
};
use stark_backend_gpu::types::DeviceAirProofRawInput;

#[macro_use]
extern crate derive_new;

pub mod dummy;
pub mod extensions;
pub mod mod_builder;
pub mod primitives;
pub mod system;
#[cfg(any(feature = "test-utils", test))]
pub mod testing;
mod utils;

pub use utils::*;
