use std::sync::Arc;

#[cfg(feature = "cuda")]
use cuda_backend_v2::GpuBackendV2;
use openvm_stark_backend::AirRef;
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use recursion_circuit::system::VerifierSubCircuit;
use stark_backend_v2::prover::CpuBackendV2;

use crate::circuit::{nonroot::NonRootTraceGenImpl, root::RootTraceGenImpl};

mod compression;
mod nonroot;
mod root;
mod utils;

pub use compression::*;
pub use nonroot::*;
pub use root::*;
pub use utils::*;

// TODO: move to stark-backend-v2
pub trait Circuit {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>>;
}

impl<C: Circuit> Circuit for Arc<C> {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        self.as_ref().airs()
    }
}

pub type NonRootCpuProver<const MAX_NUM_PROOFS: usize> =
    NonRootAggregationProver<CpuBackendV2, VerifierSubCircuit<MAX_NUM_PROOFS>, NonRootTraceGenImpl>;
pub type CompressionCpuProver =
    CompressionProver<CpuBackendV2, VerifierSubCircuit<1>, NonRootTraceGenImpl>;
pub type RootCpuProver = RootProver<CpuBackendV2, VerifierSubCircuit<1>, RootTraceGenImpl>;

#[cfg(feature = "cuda")]
pub type NonRootGpuProver<const MAX_NUM_PROOFS: usize> =
    NonRootAggregationProver<GpuBackendV2, VerifierSubCircuit<MAX_NUM_PROOFS>, NonRootTraceGenImpl>;
#[cfg(feature = "cuda")]
pub type CompressionGpuProver =
    CompressionProver<GpuBackendV2, VerifierSubCircuit<1>, NonRootTraceGenImpl>;
#[cfg(feature = "cuda")]
pub type RootGpuProver = RootProver<GpuBackendV2, VerifierSubCircuit<1>, RootTraceGenImpl>;
