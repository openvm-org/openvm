use std::sync::Arc;

use derivative::Derivative;
use openvm_stark_backend::{
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    proof::Proof,
    StarkProtocolConfig,
};
use serde::{Deserialize, Serialize};

mod hook;
mod inner;
pub use hook::*;
pub use inner::*;

#[derive(Derivative, Serialize, Deserialize)]
#[derivative(Clone(bound = ""))]
#[serde(bound = "")]
pub struct DeferralCircuitProverKey<SC: StarkProtocolConfig> {
    pub base_pk: Arc<MultiStarkProvingKey<SC>>,
    pub aux: Vec<u8>,
}

pub trait DeferralCircuitProver<SC: StarkProtocolConfig> {
    fn get_vk(&self) -> Arc<MultiStarkVerifyingKey<SC>>;
    fn get_pk(&self) -> Arc<DeferralCircuitProverKey<SC>>;

    fn from_pk(encoded_pk: DeferralCircuitProverKey<SC>) -> Self
    where
        Self: Sized;

    fn prove(&self, input_bytes: &[u8]) -> Proof<SC>;
    fn get_def_idx(&self) -> usize;
}
