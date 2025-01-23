use derivative::Derivative;
use openvm_native_compiler::ir::DIGEST_SIZE;
use openvm_stark_backend::{
    config::{Com, StarkGenericConfig, Val},
    prover::types::Proof,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use static_assertions::assert_impl_all;

use crate::verifier::leaf::types::UserPublicValuesRootProof;

#[derive(Serialize, Deserialize, Derivative)]
#[serde(bound = "")]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
pub struct MinimalVmVerifierInput<SC: StarkGenericConfig> {
    /// App VM proof
    pub proof: Proof<SC>,
    /// Public values to expose directly
    pub public_values: Vec<Val<SC>>,
}
assert_impl_all!(MinimalVmVerifierInput<BabyBearPoseidon2Config>: Serialize, DeserializeOwned);

#[derive(Debug)]
pub struct MinimalVmVerifierPvs<T> {
    /// The commitment of the App VM executable.
    pub exe_commit: [T; DIGEST_SIZE],
    /// The commitment of the leaf verifier program, which commits the VM config of App VM.
    pub leaf_verifier_commit: [T; DIGEST_SIZE],
    /// Raw public values from App VM execution.
    pub public_values: Vec<T>,
}

impl<F: Copy> MinimalVmVerifierPvs<F> {
    pub fn flatten(self) -> Vec<F> {
        let mut ret = self.exe_commit.to_vec();
        ret.extend(self.leaf_verifier_commit);
        ret.extend(self.public_values);
        ret
    }
    pub fn from_flatten(flatten: Vec<F>) -> Self {
        let exe_commit = flatten[..DIGEST_SIZE].try_into().unwrap();
        let leaf_verifier_commit = flatten[DIGEST_SIZE..2 * DIGEST_SIZE].try_into().unwrap();
        let public_values = flatten[2 * DIGEST_SIZE..].to_vec();
        Self {
            exe_commit,
            leaf_verifier_commit,
            public_values,
        }
    }
}
