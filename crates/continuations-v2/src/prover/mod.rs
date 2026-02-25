use std::sync::Arc;

#[cfg(feature = "cuda")]
use openvm_cuda_backend::GpuBackend;
use openvm_stark_backend::{prover::CpuBackend, AirRef};
use recursion_circuit::system::VerifierSubCircuit;

use crate::{
    circuit::{nonroot::NonRootTraceGenImpl, root::RootTraceGenImpl},
    SC,
};

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
    fn airs(&self) -> Vec<AirRef<SC>>;
}

impl<C: Circuit> Circuit for Arc<C> {
    fn airs(&self) -> Vec<AirRef<SC>> {
        self.as_ref().airs()
    }
}

pub type NonRootCpuProver<const MAX_NUM_PROOFS: usize> = NonRootAggregationProver<
    CpuBackend<SC>,
    VerifierSubCircuit<MAX_NUM_PROOFS>,
    NonRootTraceGenImpl,
>;
pub type CompressionCpuProver =
    CompressionProver<CpuBackend<SC>, VerifierSubCircuit<1>, NonRootTraceGenImpl>;
pub type RootCpuProver = RootProver<CpuBackend<SC>, VerifierSubCircuit<1>, RootTraceGenImpl>;

#[cfg(feature = "cuda")]
pub type NonRootGpuProver<const MAX_NUM_PROOFS: usize> =
    NonRootAggregationProver<GpuBackend, VerifierSubCircuit<MAX_NUM_PROOFS>, NonRootTraceGenImpl>;
#[cfg(feature = "cuda")]
pub type CompressionGpuProver =
    CompressionProver<GpuBackend, VerifierSubCircuit<1>, NonRootTraceGenImpl>;
#[cfg(feature = "cuda")]
pub type RootGpuProver = RootProver<GpuBackend, VerifierSubCircuit<1>, RootTraceGenImpl>;
