use std::sync::Arc;

#[cfg(feature = "cuda")]
use cuda_backend_v2::GpuBackendV2;
use eyre::Result;
use openvm_stark_backend::AirRef;
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use recursion_circuit::system::VerifierSubCircuit;
use stark_backend_v2::{
    DIGEST_SIZE, F, StarkWhirEngine,
    keygen::types::MultiStarkVerifyingKeyV2,
    proof::Proof,
    prover::{CpuBackendV2, ProverBackendV2, ProvingContextV2},
};

use crate::circuit::nonroot::NonRootTraceGen;

mod compression;
mod nonroot;
mod root;
mod utils;

pub use compression::*;
pub use nonroot::*;
pub use root::*;
pub use utils::*;

pub const DEFAULT_MAX_NUM_PROOFS: usize = 4;

pub trait AggregationProver<PB: ProverBackendV2> {
    /// Verifying key used to verify the result of agg_prove
    fn get_vk(&self) -> Arc<MultiStarkVerifyingKeyV2>;
    /// Commit of verifier circuit's cached trace
    fn get_cached_commit(&self, is_recursive: bool) -> PB::Commitment;
    fn generate_proving_ctx(
        &self,
        proofs: &[Proof],
        user_pv_commit: Option<[F; DIGEST_SIZE]>,
        is_recursive: bool,
    ) -> ProvingContextV2<PB>;
    fn agg_prove<E: StarkWhirEngine<PB = PB>>(
        &self,
        proofs: &[Proof],
        user_pv_commit: Option<[F; DIGEST_SIZE]>,
        is_recursive: bool,
    ) -> Result<Proof>;
}

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
    NonRootAggregationProver<CpuBackendV2, VerifierSubCircuit<MAX_NUM_PROOFS>, NonRootTraceGen>;
pub type CompressionCpuProver =
    CompressionProver<CpuBackendV2, VerifierSubCircuit<1>, NonRootTraceGen>;

#[cfg(feature = "cuda")]
pub type NonRootGpuProver<const MAX_NUM_PROOFS: usize> =
    NonRootAggregationProver<GpuBackendV2, VerifierSubCircuit<MAX_NUM_PROOFS>, NonRootTraceGen>;
#[cfg(feature = "cuda")]
pub type CompressionGpuProver =
    CompressionProver<GpuBackendV2, VerifierSubCircuit<1>, NonRootTraceGen>;
