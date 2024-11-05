use std::{array, borrow::BorrowMut};

use ax_stark_sdk::ax_stark_backend::{
    config::{Com, StarkGenericConfig, Val},
    p3_field::PrimeField32,
    prover::types::Proof,
};
use axvm_circuit::{
    circuit_derive::AlignedBorrow,
    system::{
        connector::VmConnectorPvs,
        memory::{merkle::MemoryMerklePvs, tree::public_values::UserPublicValuesProof},
    },
};
use axvm_native_compiler::ir::{Builder, Config, Felt, DIGEST_SIZE};
use derivative::Derivative;
use serde::{Deserialize, Serialize};

#[derive(Debug, AlignedBorrow)]
#[repr(C)]
pub struct LeafVmVerifierPvs<T> {
    pub app_commit: [T; DIGEST_SIZE],
    pub connector: VmConnectorPvs<T>,
    pub memory: MemoryMerklePvs<T, DIGEST_SIZE>,
    pub public_values_commit: [T; DIGEST_SIZE],
}

/// Input for the leaf VM verifier.
#[derive(Serialize, Deserialize, Derivative)]
#[serde(bound = "")]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
pub struct LeafVmVerifierInput<SC: StarkGenericConfig> {
    /// The proofs of the execution segments in the execution order.
    pub proofs: Vec<Proof<SC>>,
    /// The public values root proof. Leaf VM verifier only needs this when verifying the last
    /// segment.
    pub public_values_root_proof: Option<UserPublicValuesRootProof<Val<SC>>>,
}

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

impl<F: PrimeField32> LeafVmVerifierPvs<Felt<F>> {
    pub(crate) fn uninit<C: Config<F = F>>(builder: &mut Builder<C>) -> Self {
        Self {
            app_commit: array::from_fn(|_| builder.uninit()),
            connector: VmConnectorPvs {
                initial_pc: builder.uninit(),
                final_pc: builder.uninit(),
                exit_code: builder.uninit(),
                is_terminate: builder.uninit(),
            },
            memory: MemoryMerklePvs {
                initial_root: array::from_fn(|_| builder.uninit()),
                final_root: array::from_fn(|_| builder.uninit()),
            },
            public_values_commit: array::from_fn(|_| builder.uninit()),
        }
    }
}

impl<F: Default + Clone> LeafVmVerifierPvs<Felt<F>> {
    pub fn flatten(self) -> Vec<Felt<F>> {
        let mut v = vec![Felt(0, Default::default()); LeafVmVerifierPvs::<u8>::width()];
        *v.as_mut_slice().borrow_mut() = self;
        v
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
