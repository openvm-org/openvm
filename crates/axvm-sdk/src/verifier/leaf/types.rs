use ax_stark_sdk::{
    ax_stark_backend::{
        config::{Com, StarkGenericConfig, Val},
        prover::types::Proof,
    },
    config::baby_bear_poseidon2::BabyBearPoseidon2Config,
    p3_baby_bear::BabyBear,
};
use axvm_circuit::system::memory::tree::public_values::UserPublicValuesProof;
use axvm_native_compiler::ir::DIGEST_SIZE;
use derivative::Derivative;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use static_assertions::assert_impl_all;

use crate::prover::vm::ContinuationVmProof;

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
/// `axvm-circuit::system::memory::public_values::UserPublicValuesProof`.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct UserPublicValuesRootProof<F> {
    /// Sibling hashes for proving the merkle root of public values. For a specific VM, the path
    /// is constant. So we don't need the boolean which indicates if a node is a left child or right
    /// child.
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
            sibling_hashes: pvs_proof
                .proof
                .clone()
                .into_iter()
                .map(|(_, hash)| hash)
                .collect(),
            public_values_commit: pvs_proof.public_values_commit.clone(),
        }
    }
}
