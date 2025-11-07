use std::sync::Arc;

use eyre::Result;
use stark_backend_v2::{SystemParams, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};

mod nonroot;
mod root;

pub use nonroot::*;
pub use root::*;

#[derive(Clone, Copy, Debug)]
pub struct AggregationVerifier<const LEAF_NUM_CHILDREN: usize, const INTERNAL_NUM_CHILDREN: usize> {
    pub leaf_system_params: SystemParams,
    pub internal_system_params: SystemParams,
    pub root_system_params: SystemParams,
}

impl<const LEAF_NUM_CHILDREN: usize, const INTERNAL_NUM_CHILDREN: usize>
    AggregationVerifier<LEAF_NUM_CHILDREN, INTERNAL_NUM_CHILDREN>
{
    pub fn verify(&self, proofs: &[Proof], app_vk: Arc<MultiStarkVerifyingKeyV2>) -> Result<Proof> {
        // Verify app-layer proofs and generate leaf-layer proofs
        let leaf_verifier =
            NonRootVerifier::<LEAF_NUM_CHILDREN>::new(app_vk, self.leaf_system_params);
        let leaf_proofs = proofs
            .chunks(LEAF_NUM_CHILDREN)
            .map(|proofs| leaf_verifier.verify(proofs))
            .collect::<Result<Vec<_>>>()?;

        // Verify leaf-layer proofs and generate internal-for-leaf-layer proofs
        let internal_0_verifier = NonRootVerifier::<INTERNAL_NUM_CHILDREN>::new(
            Arc::new(leaf_verifier.get_vk()),
            self.internal_system_params,
        );
        let mut internal_proofs = leaf_proofs
            .chunks(INTERNAL_NUM_CHILDREN)
            .map(|proofs| internal_0_verifier.verify(proofs))
            .collect::<Result<Vec<_>>>()?;

        // Verify internal-for-leaf-layer proofs and generate internal-recursive-layer proofs
        let internal_1_verifier = NonRootVerifier::<INTERNAL_NUM_CHILDREN>::new(
            Arc::new(internal_0_verifier.get_vk()),
            self.internal_system_params,
        );
        internal_proofs = internal_proofs
            .chunks(INTERNAL_NUM_CHILDREN)
            .map(|proofs| internal_1_verifier.verify(proofs))
            .collect::<Result<Vec<_>>>()?;

        // Recursively verify internal-layer proofs until only 1 remains
        while internal_proofs.len() > 1 {
            // TODO[stephen]: add some debug that ensures the internal verifier vk stabilizes
            internal_proofs = internal_proofs
                .chunks(INTERNAL_NUM_CHILDREN)
                .map(|proofs| internal_1_verifier.verify(proofs))
                .collect::<Result<Vec<_>>>()?;
        }

        // Verify final internal-layer proof and return final root proof
        let root_verifier = RootVerifier::new(
            Arc::new(internal_1_verifier.get_vk()),
            self.root_system_params,
        );
        root_verifier.verify(&internal_proofs[0])
    }
}
