use std::sync::Arc;

#[cfg(feature = "cuda")]
use openvm_cuda_backend::GpuBackend;
use openvm_stark_backend::{prover::CpuBackend, AirRef};
use recursion_circuit::system::VerifierSubCircuit;

use crate::{
    circuit::{
        deferral::{
            aggregation::{nonroot::DeferralNonRootTraceGenImpl, root::DeferralRootTraceGenImpl},
            verify::DeferredVerifyTraceGenImpl,
        },
        nonroot::NonRootTraceGenImpl,
        root::RootTraceGenImpl,
    },
    SC,
};

mod compression;
mod deferral;
mod nonroot;
mod root;
mod utils;

pub use compression::*;
pub use deferral::*;
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
pub type DeferralVerifyCpuProver =
    DeferredVerifyProver<CpuBackend<SC>, VerifierSubCircuit<1>, DeferredVerifyTraceGenImpl>;
pub type DeferralNonRootCpuProver<const MAX_NUM_PROOFS: usize> = DeferralNonRootProver<
    CpuBackend<SC>,
    VerifierSubCircuit<MAX_NUM_PROOFS>,
    DeferralNonRootTraceGenImpl,
>;
pub type DeferralRootCpuProver =
    DeferralRootProver<CpuBackend<SC>, VerifierSubCircuit<1>, DeferralRootTraceGenImpl>;

#[cfg(feature = "cuda")]
pub type NonRootGpuProver<const MAX_NUM_PROOFS: usize> =
    NonRootAggregationProver<GpuBackend, VerifierSubCircuit<MAX_NUM_PROOFS>, NonRootTraceGenImpl>;
#[cfg(feature = "cuda")]
pub type CompressionGpuProver =
    CompressionProver<GpuBackend, VerifierSubCircuit<1>, NonRootTraceGenImpl>;
#[cfg(feature = "cuda")]
pub type RootGpuProver = RootProver<GpuBackend, VerifierSubCircuit<1>, RootTraceGenImpl>;
#[cfg(feature = "cuda")]
pub type DeferralVerifyGpuProver =
    DeferredVerifyProver<GpuBackend, VerifierSubCircuit<1>, DeferredVerifyTraceGenImpl>;
#[cfg(feature = "cuda")]
pub type DeferralNonRootGpuProver<const MAX_NUM_PROOFS: usize> = DeferralNonRootProver<
    GpuBackend,
    VerifierSubCircuit<MAX_NUM_PROOFS>,
    DeferralNonRootTraceGenImpl,
>;
#[cfg(feature = "cuda")]
pub type DeferralRootGpuProver =
    DeferralRootProver<GpuBackend, VerifierSubCircuit<1>, DeferralRootTraceGenImpl>;
