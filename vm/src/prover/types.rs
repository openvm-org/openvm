use ax_stark_backend::{
    config::{Com, StarkGenericConfig, Val},
    keygen::types::MultiStarkProvingKey,
};
use ax_stark_sdk::config::FriParameters;
use derivative::Derivative;
use p3_field::PrimeField32;
use serde::{Deserialize, Serialize};

use crate::arch::VmGenericConfig;

///Proving key for a specific VM.
#[derive(Serialize, Deserialize, Derivative)]
#[serde(bound(
    serialize = "MultiStarkProvingKey<SC>: Serialize, VmConfig: Serialize",
    deserialize = "MultiStarkProvingKey<SC>: Deserialize<'de>, VmConfig: Deserialize<'de>"
))]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
pub struct VmProvingKey<SC: StarkGenericConfig, VmConfig: VmGenericConfig<Val<SC>>>
where
    Val<SC>: PrimeField32,
{
    pub fri_params: FriParameters,
    pub vm_config: VmConfig,
    pub vm_pk: MultiStarkProvingKey<SC>,
}
