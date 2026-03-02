#[cfg(feature = "cuda")]
use openvm_cuda_backend::GpuBackend;
use openvm_stark_backend::prover::CpuBackend;
use recursion_circuit::system::VerifierSubCircuit;

use crate::{
    circuit::{
        deferral::{
            aggregation::{hook::DeferralRootTraceGenImpl, inner::DeferralNonRootTraceGenImpl},
            verify::DeferredVerifyTraceGenImpl,
        },
        inner::NonRootTraceGenImpl,
        root::RootTraceGenImpl,
    },
    SC,
};

mod compression;
mod deferral;
mod inner;
mod root;
mod utils;

pub use compression::*;
pub use deferral::*;
pub use inner::*;
pub use root::*;
pub use utils::*;

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
