use std::sync::Arc;

use eyre::Result;
use stark_backend_v2::{
    keygen::types::{MultiStarkProvingKeyV2, MultiStarkVerifyingKeyV2, SystemParams},
    proof::Proof,
};

#[derive(Clone)]
pub struct RootVerifier {
    pub child_vk: Arc<MultiStarkVerifyingKeyV2>,
    pub pk: Arc<MultiStarkProvingKeyV2>,
}

impl RootVerifier {
    pub fn new(_child_vk: Arc<MultiStarkVerifyingKeyV2>, _system_params: SystemParams) -> Self {
        // TODO[stephen]: Generate MultiStarkProvingKeyV2 for NUM_CHILDREN proofs
        todo!()
    }

    pub fn get_vk(&self) -> MultiStarkVerifyingKeyV2 {
        self.pk.get_vk()
    }

    pub fn verify(&self, _proof: &Proof) -> Result<Proof> {
        // TODO[stephen]: Verify proofs and generate proof of verification/continuation correctness
        todo!()
    }
}
