use std::collections::VecDeque;

use async_trait::async_trait;
use ax_stark_backend::{
    config::{Com, StarkGenericConfig, Val},
    prover::types::Proof,
};
use derivative::Derivative;
use serde::{Deserialize, Serialize};

use crate::system::memory::{tree::public_values::UserPublicValuesProof, CHUNK};

pub mod local;

#[derive(Serialize, Deserialize, Derivative)]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
pub struct ContinuationVmProof<SC: StarkGenericConfig> {
    pub per_segment: Vec<Proof<SC>>,
    pub user_public_values: UserPublicValuesProof<{ CHUNK }, Val<SC>>,
}

/// Prover for a specific exe in a specific continuation VM using a specific Stark config.
pub trait ContinuationVmProver<SC: StarkGenericConfig> {
    fn prove(&self, input: impl Into<VecDeque<Vec<Val<SC>>>>) -> ContinuationVmProof<SC>;
}

/// Async prover for a specific exe in a specific continuation VM using a specific Stark config.
#[async_trait]
pub trait AsyncContinuationVmProver<SC: StarkGenericConfig> {
    async fn prove(
        &self,
        input: impl Into<VecDeque<Vec<Val<SC>>>> + Send + Sync,
    ) -> ContinuationVmProof<SC>;
}

/// Prover for a specific exe in a specific single-segment VM using a specific Stark config.
pub trait SingleSegmentVmProver<SC: StarkGenericConfig> {
    fn prove(&self, input: impl Into<VecDeque<Vec<Val<SC>>>>) -> Proof<SC>;
}

/// Async prover for a specific exe in a specific single-segment VM using a specific Stark config.
#[async_trait]
pub trait AsyncSingleSegmentVmProver<SC: StarkGenericConfig> {
    async fn prove(&self, input: impl Into<VecDeque<Vec<Val<SC>>>> + Send + Sync) -> Proof<SC>;
}
