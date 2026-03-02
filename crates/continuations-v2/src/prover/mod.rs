#[cfg(feature = "cuda")]
use openvm_cuda_backend::GpuBackend;
use openvm_stark_backend::prover::CpuBackend;
use recursion_circuit::system::VerifierSubCircuit;

use crate::{
    circuit::{
        deferral::{
            aggregation::{hook::DeferralHookTraceGenImpl, inner::DeferralInnerTraceGenImpl},
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
pub type DeferralInnerCpuProver<const MAX_NUM_PROOFS: usize> = DeferralInnerProver<
    CpuBackend<SC>,
    VerifierSubCircuit<MAX_NUM_PROOFS>,
    DeferralInnerTraceGenImpl,
>;
pub type DeferralHookCpuProver =
    DeferralHookProver<CpuBackend<SC>, VerifierSubCircuit<1>, DeferralHookTraceGenImpl>;

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
pub type DeferralInnerGpuProver<const MAX_NUM_PROOFS: usize> =
    DeferralInnerProver<GpuBackend, VerifierSubCircuit<MAX_NUM_PROOFS>, DeferralInnerTraceGenImpl>;
#[cfg(feature = "cuda")]
pub type DeferralHookGpuProver =
    DeferralHookProver<GpuBackend, VerifierSubCircuit<1>, DeferralHookTraceGenImpl>;
