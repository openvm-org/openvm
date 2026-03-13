use openvm_cpu_backend::CpuBackend;
#[cfg(feature = "cuda")]
use openvm_cuda_backend::{BabyBearBn254Poseidon2GpuEngine, GpuBackend};
use openvm_recursion_circuit::system::VerifierSubCircuit;

use crate::{
    circuit::{
        deferral::{hook::DeferralHookTraceGenImpl, inner::DeferralInnerTraceGenImpl},
        inner::InnerTraceGenImpl,
        root::RootTraceGenImpl,
    },
    RootSC, SC,
};

mod deferral;
mod inner;
mod root;
mod utils;

pub use deferral::*;
pub use inner::*;
pub use root::*;
pub use utils::*;

pub type InnerCpuProver<const MAX_NUM_PROOFS: usize> =
    InnerAggregationProver<CpuBackend<SC>, VerifierSubCircuit<MAX_NUM_PROOFS>, InnerTraceGenImpl>;
pub type RootCpuProver = RootProver<CpuBackend<RootSC>, VerifierSubCircuit<1>, RootTraceGenImpl>;
pub type DeferralInnerCpuProver<const MAX_NUM_PROOFS: usize> = DeferralInnerProver<
    CpuBackend<SC>,
    VerifierSubCircuit<MAX_NUM_PROOFS>,
    DeferralInnerTraceGenImpl,
>;
pub type DeferralHookCpuProver =
    DeferralHookProver<CpuBackend<SC>, VerifierSubCircuit<1>, DeferralHookTraceGenImpl>;

#[cfg(feature = "cuda")]
pub type InnerGpuProver<const MAX_NUM_PROOFS: usize> =
    InnerAggregationProver<GpuBackend, VerifierSubCircuit<MAX_NUM_PROOFS>, InnerTraceGenImpl>;
#[cfg(feature = "cuda")]
pub type RootGpuProver = RootProver<
    <BabyBearBn254Poseidon2GpuEngine as openvm_stark_backend::StarkEngine>::PB,
    VerifierSubCircuit<1>,
    RootTraceGenImpl,
>;
#[cfg(feature = "cuda")]
pub type DeferralInnerGpuProver<const MAX_NUM_PROOFS: usize> =
    DeferralInnerProver<GpuBackend, VerifierSubCircuit<MAX_NUM_PROOFS>, DeferralInnerTraceGenImpl>;
#[cfg(feature = "cuda")]
pub type DeferralHookGpuProver =
    DeferralHookProver<GpuBackend, VerifierSubCircuit<1>, DeferralHookTraceGenImpl>;
