use async_trait::async_trait;

use crate::{
    config::StarkGenericConfig,
    prover::types::{Proof, ProofInput},
    verifier::VerificationError,
};

/// Async prover for a specific circuit using a specific Stark config.
#[async_trait]
pub trait AsyncCircuitProver<SC: StarkGenericConfig> {
    async fn prove(&self, proof_input: ProofInput<SC>) -> Proof<SC>;
}

/// Prover for a specific circuit using a specific Stark config.
pub trait CircuitProver<SC: StarkGenericConfig> {
    fn prove(&self, proof_input: ProofInput<SC>) -> Proof<SC>;
}

/// Verifier for a specific circuit using a specific Stark config.
pub trait CircuitVerifier<SC: StarkGenericConfig> {
    fn verify(&self, proof: &Proof<SC>) -> Result<(), VerificationError>;
}
