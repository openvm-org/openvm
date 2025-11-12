use std::sync::Arc;

use eyre::Result;
use openvm_circuit::system::memory::merkle::public_values::UserPublicValuesProof;
use stark_backend_v2::{
    DIGEST_SIZE, F, SystemParams, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof,
};

#[derive(Clone)]
pub struct RootVerifier {
    pub child_vk: Arc<MultiStarkVerifyingKeyV2>,
    // pub pk: Arc<MultiStarkProvingKeyV2>,
}

impl RootVerifier {
    pub fn new(child_vk: Arc<MultiStarkVerifyingKeyV2>, _system_params: SystemParams) -> Self {
        // TODO[stephen]: Generate MultiStarkProvingKeyV2 using root pv AIRs
        Self { child_vk }
    }

    pub fn verify(
        &self,
        proof: &Proof,
        _user_pvs_proof: UserPublicValuesProof<DIGEST_SIZE, F>,
    ) -> Result<Proof> {
        // TODO[stephen]: Verify proofs and generate proof of verification/continuation correctness
        Ok(proof.clone())
    }
}
