use openvm_cpu_backend::CpuBackend;
#[cfg(feature = "cuda")]
use openvm_cuda_backend::GpuBackend;
#[cfg(feature = "cuda")]
use openvm_cuda_common::stream::GpuDeviceCtx;
use openvm_recursion_circuit::system::VerifierSubCircuit;

#[cfg(feature = "root-prover")]
use crate::circuit::root::RootTraceGenImpl;
use crate::{
    circuit::{
        deferral::{hook::DeferralHookTraceGenImpl, inner::DeferralInnerTraceGenImpl},
        inner::InnerTraceGenImpl,
    },
    SC,
};

mod deferral;
mod inner;
mod utils;

pub use deferral::*;
pub use inner::*;
pub use utils::*;

#[cfg(feature = "root-prover")]
mod root;
#[cfg(feature = "root-prover")]
pub use root::*;

pub type InnerCpuProver<const MAX_NUM_PROOFS: usize> =
    InnerAggregationProver<CpuBackend<SC>, VerifierSubCircuit<MAX_NUM_PROOFS>, InnerTraceGenImpl>;
#[cfg(feature = "cuda")]
pub type InnerGpuProver<const MAX_NUM_PROOFS: usize> = InnerAggregationProver<
    GpuBackend,
    VerifierSubCircuit<MAX_NUM_PROOFS>,
    InnerTraceGenImpl,
    GpuDeviceCtx,
>;

#[cfg(feature = "root-prover")]
pub type RootCpuProver = RootProver<VerifierSubCircuit<1>, RootTraceGenImpl>;
#[cfg(all(feature = "cuda", feature = "root-prover"))]
pub type RootGpuProver = RootProver<VerifierSubCircuit<1>, RootTraceGenImpl>;

pub type DeferralInnerCpuProver =
    DeferralInnerProver<CpuBackend<SC>, VerifierSubCircuit<2>, DeferralInnerTraceGenImpl>;
#[cfg(feature = "cuda")]
pub type DeferralInnerGpuProver =
    DeferralInnerProver<GpuBackend, VerifierSubCircuit<2>, DeferralInnerTraceGenImpl, GpuDeviceCtx>;

pub type DeferralHookCpuProver =
    DeferralHookProver<CpuBackend<SC>, VerifierSubCircuit<1>, DeferralHookTraceGenImpl>;
#[cfg(feature = "cuda")]
pub type DeferralHookGpuProver =
    DeferralHookProver<GpuBackend, VerifierSubCircuit<1>, DeferralHookTraceGenImpl, GpuDeviceCtx>;
