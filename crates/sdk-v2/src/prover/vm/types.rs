use std::sync::Arc;

use derivative::Derivative;
use openvm_stark_backend::{keygen::types::MultiStarkProvingKey, SystemParams};
use serde::{Deserialize, Serialize};

/// Proving key for a specific VM.
#[derive(Serialize, Deserialize, Derivative)]
pub struct VmProvingKey<VC> {
    pub vm_config: VC,
    pub vm_pk: Arc<MultiStarkProvingKey<crate::SC>>,
}

impl<VC> VmProvingKey<VC> {
    pub fn get_params(&self) -> SystemParams {
        self.vm_pk.params.clone()
    }
}
