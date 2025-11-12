use std::sync::Arc;

use eyre::Result;
use openvm_circuit::system::memory::merkle::public_values::UserPublicValuesProof;
use stark_backend_v2::{
    DIGEST_SIZE, F, SystemParams, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof,
};

mod nonroot;
mod root;

pub use nonroot::*;
pub use root::*;

const MAX_NUM_PROOFS: usize = 4;

#[derive(Clone, Copy, Debug)]
pub struct AggregationVerifier<const LEAF_NUM_CHILDREN: usize, const INTERNAL_NUM_CHILDREN: usize> {
    pub leaf_system_params: SystemParams,
    pub internal_system_params: SystemParams,
    pub root_system_params: SystemParams,
}

impl<const LEAF_NUM_CHILDREN: usize, const INTERNAL_NUM_CHILDREN: usize>
    AggregationVerifier<LEAF_NUM_CHILDREN, INTERNAL_NUM_CHILDREN>
{
    pub fn verify(
        &self,
        app_vk: Arc<MultiStarkVerifyingKeyV2>,
        proofs: &[Proof],
        user_pvs_proof: UserPublicValuesProof<DIGEST_SIZE, F>,
    ) -> Result<Proof> {
        // Verify app-layer proofs and generate leaf-layer proofs
        let leaf_verifier =
            NonRootVerifier::<LEAF_NUM_CHILDREN>::new(app_vk, self.leaf_system_params);
        let leaf_proofs = proofs
            .chunks(LEAF_NUM_CHILDREN)
            .map(|proofs| leaf_verifier.verify(proofs, Some(user_pvs_proof.public_values_commit)))
            .collect::<Result<Vec<_>>>()?;

        // Verify leaf-layer proofs and generate internal-for-leaf-layer proofs
        let internal_0_verifier = NonRootVerifier::<INTERNAL_NUM_CHILDREN>::new(
            leaf_verifier.vk.clone(),
            self.internal_system_params,
        );
        let mut internal_proofs = leaf_proofs
            .chunks(INTERNAL_NUM_CHILDREN)
            .map(|proofs| internal_0_verifier.verify(proofs, None))
            .collect::<Result<Vec<_>>>()?;

        // Verify internal-for-leaf-layer proofs and generate internal-recursive-layer proofs
        let internal_1_verifier = NonRootVerifier::<INTERNAL_NUM_CHILDREN>::new(
            internal_0_verifier.vk.clone(),
            self.internal_system_params,
        );
        internal_proofs = internal_proofs
            .chunks(INTERNAL_NUM_CHILDREN)
            .map(|proofs| internal_1_verifier.verify(proofs, None))
            .collect::<Result<Vec<_>>>()?;

        // Recursively verify internal-layer proofs until only 1 remains
        while internal_proofs.len() > 1 {
            internal_proofs = internal_proofs
                .chunks(INTERNAL_NUM_CHILDREN)
                .map(|proofs| internal_1_verifier.verify(proofs, None))
                .collect::<Result<Vec<_>>>()?;
        }

        // Verify final internal-layer proof and return final root proof
        let root_verifier =
            RootVerifier::new(internal_1_verifier.vk.clone(), self.root_system_params);
        root_verifier.verify(&internal_proofs[0], user_pvs_proof)
    }
}
