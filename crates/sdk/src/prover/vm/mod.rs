use std::sync::{mpsc, Arc};

use async_trait::async_trait;
use openvm_circuit::{
    arch::{ContinuationVmProof, Streams, VmConfig},
    system::program::trace::VmCommittedExe,
};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    proof::Proof,
    prover::types::ProofInput,
};
use openvm_stark_sdk::config::FriParameters;

use crate::prover::vm::types::VmProvingKey;

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

    /// Spawns a worker thread to execute segments and generate proof inputs
    fn spawn_trace_worker(
        &self,
        input_rx: mpsc::Receiver<Streams<Val<SC>>>,
        proof_tx: mpsc::SyncSender<(ProofInput<SC>, tracing::Span)>,
    ) -> std::thread::JoinHandle<()>;

    /// Spawns a worker thread to generate proofs from proof inputs
    fn spawn_prove_worker(
        &self,
        proof_input_rx: mpsc::Receiver<(ProofInput<SC>, tracing::Span)>,
        proof_tx: mpsc::SyncSender<Proof<SC>>,
    ) -> std::thread::JoinHandle<()>;
}

/// Async prover for a specific exe in a specific single-segment VM using a specific Stark config.
#[async_trait]
pub trait AsyncSingleSegmentVmProver<SC: StarkGenericConfig> {
    async fn prove(&self, input: impl Into<Streams<Val<SC>>> + Send + Sync) -> Proof<SC>;
}
