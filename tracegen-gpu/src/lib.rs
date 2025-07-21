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

pub trait DeviceChip<SC: StarkGenericConfig, PB: ProverBackend>: ChipUsageGetter {
    fn air(&self) -> AirRef<SC>;
    fn generate_trace(&self) -> PB::Matrix;
    fn generate_device_air_proof_input(&self) -> DeviceAirProofRawInput<PB> {
        DeviceAirProofRawInput {
            cached_mains: vec![],
            common_main: Some(self.generate_trace()),
            public_values: vec![],
        }
    }
}

impl<SC: StarkGenericConfig, PB: ProverBackend, G: DeviceChip<SC, PB>> DeviceChip<SC, PB>
    for Arc<G>
{
    fn air(&self) -> AirRef<SC> {
        self.as_ref().air()
    }

    fn generate_trace(&self) -> PB::Matrix {
        self.as_ref().generate_trace()
    }
}
