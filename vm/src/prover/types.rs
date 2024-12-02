use ax_stark_backend::{
    config::{Com, StarkGenericConfig},
    keygen::types::MultiStarkProvingKey,
};
use ax_stark_sdk::config::FriParameters;
use derivative::Derivative;
use serde::{Deserialize, Serialize};

///Proving key for a specific VM.
#[derive(Serialize, Deserialize, Derivative)]
#[serde(bound(
    serialize = "MultiStarkProvingKey<SC>: Serialize, VmConfig: Serialize",
    deserialize = "MultiStarkProvingKey<SC>: Deserialize<'de>, VmConfig: Deserialize<'de>"
))]
#[derivative(Clone(bound = "Com<SC>: Clone, VmConfig: Clone"))]
pub struct VmProvingKey<SC: StarkGenericConfig, VmConfig> {
    pub fri_params: FriParameters,
    pub vm_config: VmConfig,
    pub vm_pk: MultiStarkProvingKey<SC>,
}
