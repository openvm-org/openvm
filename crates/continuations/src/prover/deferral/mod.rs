use std::sync::Arc;

use openvm_stark_backend::{
    keygen::types::MultiStarkVerifyingKey, proof::Proof, StarkProtocolConfig,
};

mod hook;
mod inner;
pub use hook::*;
pub use inner::*;

pub trait DeferralCircuitProver<SC: StarkProtocolConfig> {
    fn get_vk(&self) -> Arc<MultiStarkVerifyingKey<SC>>;
    fn prove(&self, input_bytes: &[u8]) -> Proof<SC>;
}
