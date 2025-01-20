use derivative::Derivative;
use openvm_native_compiler::ir::DIGEST_SIZE;
use openvm_stark_backend::{
    config::{Com, StarkGenericConfig, Val},
    prover::types::Proof,
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Derivative)]
#[serde(bound = "")]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
pub struct MinimalVmVerifierInput<SC: StarkGenericConfig> {
    /// App VM proof
    pub proof: Proof<SC>,
    pub public_values: Vec<Val<SC>>,
}

#[derive(Debug)]
pub struct MinimalVmVerifierPvs<T> {
    /// The commitment of the App VM executable.
    pub exe_commit: [T; DIGEST_SIZE],
    // /// The commitment of the leaf verifier program, which commits the VM config of App VM.
    // pub leaf_verifier_commit: [T; DIGEST_SIZE],
    /// Raw public values from App VM execution.
    pub public_values: Vec<T>,
}
