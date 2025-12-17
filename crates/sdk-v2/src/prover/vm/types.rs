use std::sync::Arc;

use derivative::Derivative;
use serde::{Deserialize, Serialize};
use stark_backend_v2::{SystemParams, keygen::types::MultiStarkProvingKeyV2};

/// Proving key for a specific VM.
#[derive(Serialize, Deserialize, Derivative)]
pub struct VmProvingKey<VC> {
    pub vm_config: VC,
    pub vm_pk: Arc<MultiStarkProvingKeyV2>,
}

impl<VC> VmProvingKey<VC> {
    pub fn get_params(&self) -> SystemParams {
        self.vm_pk.params.clone()
    }
}
