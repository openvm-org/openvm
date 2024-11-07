use async_trait::async_trait;
use p3_uni_stark::StarkGenericConfig;

use crate::{
    prover::types::{Proof, ProofInput},
    verifier::VerificationError,
};

/// Async prover for a specific STARK using a specific Stark config.
#[async_trait]
pub trait AsyncConcreteProver<SC: StarkGenericConfig> {
    async fn prove(&self, proof_input: ProofInput<SC>) -> Proof<SC>;
}

/// Prover for a specific STARK using a specific Stark config.
pub trait ConcreteProver<SC: StarkGenericConfig> {
    fn prove(&self, proof_input: ProofInput<SC>) -> Proof<SC>;
}

/// Verifier for a specific STARK using a specific Stark config.
pub trait ConcreteVerifier<SC: StarkGenericConfig> {
    fn verify(&self, proof: &Proof<SC>) -> Result<(), VerificationError>;
}
