use std::sync::Arc;

use eyre::Result;
use stark_backend_v2::{
    keygen::types::{MultiStarkProvingKeyV2, MultiStarkVerifyingKeyV2, SystemParams},
    proof::Proof,
};

#[derive(Clone)]
pub struct NonRootVerifier<const NUM_CHILDREN: usize> {
    pub child_vk: Arc<MultiStarkVerifyingKeyV2>,
    pub pk: Arc<MultiStarkProvingKeyV2>,
}

impl<const NUM_CHILDREN: usize> NonRootVerifier<NUM_CHILDREN> {
    pub fn new(_child_vk: Arc<MultiStarkVerifyingKeyV2>, _system_params: SystemParams) -> Self {
        // TODO[stephen]: Generate MultiStarkProvingKeyV2 for NUM_CHILDREN proofs
        todo!()
    }

    pub fn get_vk(&self) -> MultiStarkVerifyingKeyV2 {
        self.pk.get_vk()
    }

    pub fn verify(&self, proofs: &[Proof]) -> Result<Proof> {
        assert!(proofs.len() <= NUM_CHILDREN);
        // TODO[stephen]: Verify proofs and generate proof of verification/continuation correctness
        todo!()
    }
}
