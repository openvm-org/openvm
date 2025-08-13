use derivative::Derivative;
use openvm_stark_backend::{
    config::{Com, StarkGenericConfig},
    keygen::types::MultiStarkProvingKey,
};
use openvm_stark_sdk::config::FriParameters;
use serde::{Deserialize, Serialize};

/// Proving key for a specific VM.
#[derive(Serialize, Deserialize, Derivative)]
#[serde(bound(
    serialize = "MultiStarkProvingKey<SC>: Serialize, VC: Serialize",
    deserialize = "MultiStarkProvingKey<SC>: Deserialize<'de>, VC: Deserialize<'de>"
))]
#[derivative(Clone(bound = "Com<SC>: Clone, VC: Clone"))]
pub struct VmProvingKey<SC: StarkGenericConfig, VC> {
    pub fri_params: FriParameters,
    pub vm_config: VC,
    pub vm_pk: MultiStarkProvingKey<SC>,
}
