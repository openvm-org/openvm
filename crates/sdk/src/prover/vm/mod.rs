use async_trait::async_trait;
use openvm_circuit::arch::{ContinuationVmProof, Streams};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    proof::Proof,
};

pub mod local;
pub mod types;

/// Prover for a specific exe in a specific continuation VM using a specific Stark config.
pub trait ContinuationVmProver<SC: StarkGenericConfig> {
    fn prove(&self, input: impl Into<Streams<Val<SC>>>) -> ContinuationVmProof<SC>;
}

/// Async prover for a specific exe in a specific continuation VM using a specific Stark config.
#[async_trait]
pub trait AsyncContinuationVmProver<SC: StarkGenericConfig> {
    async fn prove(
        &self,
        input: impl Into<Streams<Val<SC>>> + Send + Sync,
    ) -> ContinuationVmProof<SC>;
}

/// Prover for a specific exe in a specific single-segment VM using a specific Stark config.
pub trait SingleSegmentVmProver<SC: StarkGenericConfig> {
    fn prove(&self, input: impl Into<Streams<Val<SC>>>) -> Proof<SC>;
}

/// Async prover for a specific exe in a specific single-segment VM using a specific Stark config.
#[async_trait]
pub trait AsyncSingleSegmentVmProver<SC: StarkGenericConfig> {
    async fn prove(&self, input: impl Into<Streams<Val<SC>>> + Send + Sync) -> Proof<SC>;
}
