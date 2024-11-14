use ax_stark_backend::{
    config::{Com, StarkGenericConfig},
    keygen::types::MultiStarkProvingKey,
};
use ax_stark_sdk::config::FriParameters;
use derivative::Derivative;
use serde::{Deserialize, Serialize};

use crate::arch::VmConfig;

///Proving key for a specific VM.
#[derive(Serialize, Deserialize, Derivative)]
#[serde(bound(
    serialize = "MultiStarkProvingKey<SC>: Serialize",
    deserialize = "MultiStarkProvingKey<SC>: Deserialize<'de>"
))]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
pub struct VmProvingKey<SC: StarkGenericConfig> {
    pub fri_params: FriParameters,
    pub vm_config: VmConfig,
    pub vm_pk: MultiStarkProvingKey<SC>,
}
