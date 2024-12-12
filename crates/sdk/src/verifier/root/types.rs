use std::array;

use derivative::Derivative;
use openvm_native_compiler::ir::{Builder, Config, Felt, DIGEST_SIZE};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Config,
    openvm_stark_backend::{
        config::{Com, StarkGenericConfig, Val},
        p3_field::PrimeField32,
        prover::types::Proof,
    },
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use static_assertions::assert_impl_all;

#[derive(Debug)]
pub struct RootVmVerifierPvs<T> {
    /// The commitment of the App VM executable.
    pub exe_commit: [T; DIGEST_SIZE],
    /// The commitment of the leaf verifier program, which commits the VM config of App VM.
    pub leaf_verifier_commit: [T; DIGEST_SIZE],
    /// Raw public values from App VM execution.
    pub public_values: Vec<T>,
}

/// Input for the root VM verifier.
/// Note: Root verifier is proven in Root SC, but it usually verifies proofs in SC. So
/// usually only RootVmVerifierInput<SC> is needed.
#[derive(Serialize, Deserialize, Derivative)]
#[serde(bound = "")]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
pub struct RootVmVerifierInput<SC: StarkGenericConfig> {
    /// The proofs of leaf verifier or internal verifier in the execution order.
    pub proofs: Vec<Proof<SC>>,
    /// Public values to expose directly
    pub public_values: Vec<Val<SC>>,
}
assert_impl_all!(RootVmVerifierInput<BabyBearPoseidon2Config>: Serialize, DeserializeOwned);

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
