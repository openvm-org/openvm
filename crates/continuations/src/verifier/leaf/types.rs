use derivative::Derivative;
use openvm_circuit::{
    arch::ContinuationVmProof, system::memory::merkle::public_values::UserPublicValuesProof,
};
use openvm_native_compiler::ir::DIGEST_SIZE;
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Config,
    openvm_stark_backend::{
        config::{Com, StarkGenericConfig, Val},
        proof::Proof,
    },
    p3_baby_bear::BabyBear,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use static_assertions::assert_impl_all;

/// Input for the leaf VM verifier.
#[derive(Serialize, Deserialize, Derivative)]
#[serde(bound = "")]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
pub struct LeafVmVerifierInput<SC: StarkGenericConfig> {
    /// The proofs of the VM execution segments in the execution order.
    pub proofs: Vec<Proof<SC>>,
    /// The public values root proof. Leaf VM verifier only needs this when verifying the last
    /// segment.
    pub public_values_root_proof: Option<UserPublicValuesRootProof<Val<SC>>>,
}
assert_impl_all!(LeafVmVerifierInput<BabyBearPoseidon2Config>: Serialize, DeserializeOwned);

/// Proof that the merkle root of public values is in the memory state. Can be extracted from
/// `openvm_circuit::system::memory::public_values::UserPublicValuesProof`.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct UserPublicValuesRootProof<F> {
    /// Sibling hashes for proving the merkle root of public values. For a specific VM, the path
    /// is constant. So we don't need the boolean which indicates if a node is a left child or
    /// right child.
    pub sibling_hashes: Vec<[F; DIGEST_SIZE]>,
    pub public_values_commit: [F; DIGEST_SIZE],
}
assert_impl_all!(UserPublicValuesRootProof<BabyBear>: Serialize, DeserializeOwned);

impl<SC: StarkGenericConfig> LeafVmVerifierInput<SC> {
    pub fn chunk_continuation_vm_proof(proof: &ContinuationVmProof<SC>, chunk: usize) -> Vec<Self> {
        let ContinuationVmProof {
            per_segment,
            user_public_values,
        } = proof;
        let mut ret: Vec<Self> = per_segment
            .chunks(chunk)
            .map(|proof| Self {
                proofs: proof.to_vec(),
                public_values_root_proof: None,
            })
            .collect();
        ret.last_mut().unwrap().public_values_root_proof =
            Some(UserPublicValuesRootProof::extract(user_public_values));
        ret
    }
}

impl<F: Clone> UserPublicValuesRootProof<F> {
    pub fn extract(pvs_proof: &UserPublicValuesProof<{ DIGEST_SIZE }, F>) -> Self {
        Self {
            sibling_hashes: pvs_proof.proof.clone(),
            public_values_commit: pvs_proof.public_values_commit.clone(),
        }
    }
}
