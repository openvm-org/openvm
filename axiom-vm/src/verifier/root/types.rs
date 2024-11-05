use std::array;

use ax_stark_sdk::ax_stark_backend::{
    config::{Com, StarkGenericConfig, Val},
    p3_field::PrimeField32,
    prover::types::Proof,
};
use axvm_native_compiler::ir::{Builder, Config, Felt, DIGEST_SIZE};
use derivative::Derivative;
use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub struct RootVmVerifierPvs<T> {
    pub exe_commit: [T; DIGEST_SIZE],
    pub leaf_verifier_commit: [T; DIGEST_SIZE],
    pub public_values: Vec<T>,
}

/// Input for the root VM verifier.
#[derive(Serialize, Deserialize, Derivative)]
#[serde(bound = "")]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
pub struct RootVmVerifierInput<SC: StarkGenericConfig> {
    /// The proofs of leaf verifier or internal verifier in the execution order.
    pub proofs: Vec<Proof<SC>>,
    /// Public values to expose directly
    pub public_values: Vec<Val<SC>>,
}

impl<F: PrimeField32> RootVmVerifierPvs<Felt<F>> {
    pub fn uninit<C: Config<F = F>>(builder: &mut Builder<C>, num_public_values: usize) -> Self {
        Self {
            exe_commit: array::from_fn(|_| builder.uninit()),
            leaf_verifier_commit: array::from_fn(|_| builder.uninit()),
            public_values: (0..num_public_values).map(|_| builder.uninit()).collect(),
        }
    }
}

impl<F: Copy> RootVmVerifierPvs<F> {
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
